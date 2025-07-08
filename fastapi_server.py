import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel
import io
import time

# --- Model Definition (from convert_to_torchscript.py) ---
class NestedHashLayer(nn.Module):
    def __init__(self, feature_dim: int, hidden_size, bit_list: list[int]):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]

        self.hash_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.max_bit)
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(bit) for bit in self.bit_list])

    def forward(self, x):
        full_output = self.hash_head(x)
        outputs_bits = [full_output[:, :length] for length in self.bit_list]
        outputs = [F.normalize(ln(output), p=2, dim=1) for output, ln in zip(outputs_bits, self.layer_norms)]
        return tuple(outputs) # Return as tuple for TorchScript compatibility

class DeepHashingModelForInference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hparams = type('HParams', (object,), config)()

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        self.vision_model = backbone.vision_model
        self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim,
                                   self.hparams.bit_list)

    def forward(self, images):
        features = self.vision_model(images).last_hidden_state.mean(dim=1)
        outputs = self.nhl(features)
        return outputs # This will be a tuple from NestedHashLayer

# --- Config (from config.py) ---
config = {
    "dataset_name": "hyunlord/query_image_anchor_positive_large_384",
    "cache_dir": "./.cache",
    "model_name": "google/siglip2-base-patch16-384",
    "hash_hidden_dim": 512,
    "margin": 0.242047,
    "lambda_ortho": 0.197038,
    "lambda_lcs": 1.137855,
    "lambda_cons": 0.1,
    "lambda_quant": 0.01,
    "batch_groups": 4,
    "images_per_group": 10,
    "image_size": 384,
    "learning_rate": 0.000020,
    "epochs": 50,
    "num_workers": 28,
    "seed": 42,
    "bit_list": [8, 16, 32, 48, 64, 128]
}

# --- FastAPI App ---
app = FastAPI()

# Global model variable
model = None
device = None

@app.on_event("startup")
async def load_model():
    global model, device
    # Determine device (CPU or GPU)
    # For CPU only: device = torch.device("cpu")
    # For GPU if available, else CPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    # Load the checkpoint using the OriginalDeepHashingModel (LightningModule)
    # This is needed because load_from_checkpoint expects the LightningModule structure
    from pytorch_lightning import LightningModule # Import here to avoid circular dependency if model.py is separate
    class OriginalDeepHashingModel(LightningModule):
        def __init__(self, config):
            super().__init__()
            self.save_hyperparameters(config)
            backbone = AutoModel.from_pretrained(self.hparams.model_name)
            self.vision_model = backbone.vision_model
            self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim,
                                       self.hparams.bit_list)
            self.bit_importance_ema_dict = {}
            self.ema_decay = 0.99
        def forward(self, images):
            features = self.vision_model(images).last_hidden_state.mean(dim=1)
            outputs = self.nhl(features)
            return outputs
        def training_step(self, batch, batch_idx):
            return torch.tensor(0.0)
        def validation_step(self, batch, batch_idx):
            return torch.tensor(0.0)
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.001)


    checkpoint_path = "/hanmail/users/rexxa.som/jupyter/my_checkpoints3/last.ckpt"
    lightning_model = OriginalDeepHashingModel.load_from_checkpoint(checkpoint_path, config=config, map_location='cpu')
    lightning_model.eval()
    
    # Create the inference-only model and load state dict
    model = DeepHashingModelForInference(config)
    model.load_state_dict(lightning_model.state_dict())
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "ready", "model_loaded": True, "device": str(device)}
    return {"status": "loading", "model_loaded": False}

@app.post("/v2/models/{model_name}/versions/{model_version}/infer")
async def infer(model_name: str, model_version: str, request_body: bytes = File(...)):
    if model_name != config["model_name"] or model_version != "1":
        raise HTTPException(status_code=404, detail="Model not found")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Triton client sends binary data directly.
    # The request_body will contain the raw bytes of the numpy array.
    # We need to reconstruct the numpy array from bytes.
    # This assumes the client sends a single 'images' input.
    
    # For simplicity, assuming the client sends a numpy array directly as bytes
    # In a real scenario, you might need a more robust parsing (e.g., using a custom Pydantic model
    # or a more complex request body if multiple inputs/outputs are involved).
    try:
        # Reconstruct numpy array from bytes
        # This requires knowing the shape and dtype from the client side.
        # For this benchmark, we know it's (N, 3, IMAGE_SIZE, IMAGE_SIZE) FP32
        input_np = np.frombuffer(request_body, dtype=np.float32).reshape(-1, 3, config["image_size"], config["image_size"])
        
        # Move to torch tensor and device
        input_tensor = torch.from_numpy(input_np).to(device)

        # Perform inference
        with torch.no_grad():
            # Measure pure model inference time
            inference_start_time = time.time()
            outputs_tuple = model(input_tensor)
            inference_end_time = time.time()
            model_inference_latency_ms = (inference_end_time - inference_start_time) * 1000

        # Prepare response in a format compatible with tritonclient.http.InferResult
        # Triton client expects a specific JSON structure for HTTP/REST API
        # For benchmarking, we just need to ensure the output is correctly generated.
        
        # Convert tuple of tensors to list of numpy arrays
        output_nps = [output.cpu().numpy() for output in outputs_tuple]

        # Construct a simplified response for benchmarking purposes
        # The benchmark client only checks for successful response, not content.
        # If you need to return actual data, you'd structure it like Triton's JSON output.
        
        # Example of a minimal valid response for tritonclient.http
        # This is not a full Triton REST API response, but enough for the client to parse.
        # For this benchmark, we just need to return something that doesn't error out.
        
        # For the benchmark client, we just need a successful HTTP 200 response.
        # The client's `response.as_numpy()` will try to parse the binary data.
        # So, we need to return the binary data directly.
        
        # Ensure the output is the 128-bit hash, as requested by the client.
        # The client requests f"output_{BIT_LIST[-1]}_bit" which is "output_128_bit"
        # The model returns a tuple of tensors in order of BIT_LIST.
        # So, outputs_tuple[-1] corresponds to the 128-bit hash.
        
        # Convert the 128-bit output tensor to numpy bytes
        output_128_bit_np = outputs_tuple[-1].cpu().numpy()
        
        # Return the raw bytes of the numpy array.
        # The client's `response.as_numpy()` expects this.
        return output_128_bit_np.tobytes()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    # To run on CPU:
    # uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
    
    # To run on GPU (if available):
    # uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
    # The load_model() function will automatically detect CUDA if available.
    
    # For direct execution (e.g., for testing):
    uvicorn.run(app, host="0.0.0.0", port=8000)
