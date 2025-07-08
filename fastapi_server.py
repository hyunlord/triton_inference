import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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
from base64 import b64decode, b64encode # 추가
import ujson as json # 추가

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
    "model_name": "google/siglip2-base-patch16-384", # This must remain for AutoModel.from_pretrained
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

# Define the name the model is deployed under in Triton/FastAPI
DEPLOYED_MODEL_NAME = "deep_hashing" # New constant for the deployed model name

# Global model variable
model = None
device = None
model_ready = False

@app.on_event("startup")
async def load_model():
    global model, device, model_ready
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    from pytorch_lightning import LightningModule
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
        def training_step(self, batch, batch_idx): return torch.tensor(0.0)
        def validation_step(self, batch, batch_idx): return torch.tensor(0.0)
        def configure_optimizers(self): return torch.optim.AdamW(self.parameters(), lr=0.001)

    checkpoint_path = "/hanmail/users/rexxa.som/jupyter/my_checkpoints3/last.ckpt" # Ensure this path is correct

    try:
        lightning_model = OriginalDeepHashingModel.load_from_checkpoint(checkpoint_path, config=config, map_location='cpu')
        lightning_model.eval()
        
        model = DeepHashingModelForInference(config)
        model.load_state_dict(lightning_model.state_dict())
        model.eval()
        model.to(device)
        model_ready = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_ready = False

# Triton-compatible health check endpoints
@app.get("/v2/health/ready")
async def health_ready():
    print(f"DEBUG: /v2/health/ready endpoint hit. model_ready={model_ready}")
    if model_ready:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Model not ready")

@app.get("/v2/models/{model_name}/ready")
async def model_ready_check(model_name: str):
    print(f"DEBUG: /v2/models/{model_name}/ready endpoint hit. model_name={model_name}, config_model_name={config['model_name']}, deployed_model_name={DEPLOYED_MODEL_NAME}, model_ready={model_ready}")
    if model_name == DEPLOYED_MODEL_NAME and model_ready:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail=f"Model '{model_name}' not ready")

@app.post("/v2/models/{model_name}/versions/{model_version}/infer")
async def infer(model_name: str, model_version: str, request_body: dict): # request_body: dict로 변경
    print(f"DEBUG: /v2/models/{model_name}/versions/{model_version}/infer endpoint hit.")
    print(f"DEBUG: model_name={model_name}, model_version={model_version}")
    print(f"DEBUG: Expected model_name={DEPLOYED_MODEL_NAME}, expected model_version=1")

    if model_name != DEPLOYED_MODEL_NAME or model_version != "1":
        print("DEBUG: Model name or version mismatch detected!")
        raise HTTPException(status_code=404, detail="Model not found")
    if not model_ready:
        print("DEBUG: Model not ready flag is False!")
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # JSON 페이로드에서 Base64 인코딩된 데이터 추출
        # Triton client의 JSON 요청 구조를 따름
        input_data_json = request_body["inputs"][0]
        encoded_data = input_data_json["data"][0] # Assuming single data element in list
        input_bytes = b64decode(encoded_data)
        
        # NumPy 배열로 재구성
        input_np = np.frombuffer(input_bytes, dtype=np.float32).reshape(input_data_json["shape"])
        
        input_tensor = torch.from_numpy(input_np).to(device)

        with torch.no_grad():
            outputs_tuple = model(input_tensor)

        output_128_bit_np = outputs_tuple[-1].cpu().numpy()
        
        # 출력을 Base64 인코딩하여 JSON 응답으로 반환
        encoded_output = b64encode(output_128_bit_np.tobytes()).decode('utf-8')
        
        # Triton client가 파싱할 수 있는 JSON 응답 형식
        response_json = {
            "model_name": DEPLOYED_MODEL_NAME,
            "model_version": model_version,
            "outputs": [
                {
                    "name": f"output_{config['bit_list'][-1]}_bit", # 128-bit output
                    "datatype": "FP32",
                    "shape": list(output_128_bit_np.shape),
                    "data": [encoded_output]
                }
            ]
        }
        return response_json

    except Exception as e:
        print(f"DEBUG: Inference error in infer endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)