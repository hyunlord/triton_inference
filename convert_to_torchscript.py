import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel

# NestedHashLayer는 그대로 유지
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
        return outputs

# config.py의 config 딕셔너리를 여기에 포함
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

# 모델 로드 및 TorchScript 변환
checkpoint_path = "/hanmail/users/rexxa.som/jupyter/my_checkpoints3/lask.ckpt"
output_path = "/hanmail/users/rexxa.som/github/triton_inference/triton_models/deep_hashing/1/model.pt"

# 1. 체크포인트를 로드하기 위한 원래 DeepHashingModel (LightningModule) 정의
#    load_from_checkpoint가 이 클래스 정의를 필요로 합니다.
class OriginalDeepHashingModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        # 추론 시에는 gradient_checkpointing이 필요 없으므로 주석 처리하거나 제거
        # backbone.config.gradient_checkpointing = True
        # backbone.gradient_checkpointing_enable()
        self.vision_model = backbone.vision_model
        self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim,
                                   self.hparams.bit_list)
        # 추론 시에는 필요 없는 속성들
        self.bit_importance_ema_dict = {}
        self.ema_decay = 0.99

    def forward(self, images):
        features = self.vision_model(images).last_hidden_state.mean(dim=1)
        outputs = self.nhl(features)
        return outputs

    # load_from_checkpoint가 필요로 할 수 있는 최소한의 더미 메소드들
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)
    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)
    def configure_optimizers(self):
        # 옵티마이저가 필요 없지만, LightningModule의 요구사항을 충족하기 위해 더미 반환
        return torch.optim.AdamW(self.parameters(), lr=0.001)


# 2. 추론 전용 DeepHashingModel (nn.Module) 정의
#    이 모델은 LightningModule의 Trainer 관련 속성 없이 순수하게 추론 로직만 가집니다.
class DeepHashingModelForInference(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hparams를 수동으로 설정하여 OriginalDeepHashingModel의 동작을 모방
        self.hparams = type('HParams', (object,), config)()

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        self.vision_model = backbone.vision_model
        self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim,
                                   self.hparams.bit_list)

    def forward(self, images):
        features = self.vision_model(images).last_hidden_state.mean(dim=1)
        outputs = self.nhl(features)
        return tuple(outputs)


# 3. 체크포인트를 OriginalDeepHashingModel에 로드
lightning_model = OriginalDeepHashingModel.load_from_checkpoint(checkpoint_path, config=config, map_location='cpu')
lightning_model.eval()
lightning_model.to('cpu')

# 4. 추론 전용 모델 인스턴스 생성 및 학습된 가중치 로드
inference_model = DeepHashingModelForInference(config)
inference_model.load_state_dict(lightning_model.state_dict()) # Lightning 모델의 가중치를 로드
inference_model.eval()
inference_model.to('cpu')

# 더미 입력 생성
dummy_input = torch.randn(1, 3, config["image_size"], config["image_size"], device='cpu')

# 5. 추론 전용 모델을 TorchScript로 트레이스
traced_model = torch.jit.trace(inference_model, dummy_input)

# 6. TorchScript 모델 저장
traced_model.save(output_path)

print(f"Model successfully converted to TorchScript and saved at: {output_path}")