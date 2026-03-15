# ====================================================================================================================
# Architecture: [FAU + rPPG] -> Time-Aware Projection -> VideoMAE (LoRA) -> Attention Pooling -> Classifier
# ====================================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel
from peft import LoraConfig, get_peft_model
from src.backbones.fau_encoder import FAUEncoder
from src.backbones.rppg_encoder import RPPGEncoder
from src.backbones.pos import PositionalEncoding
from src.pooler.attn_pooler import AttentionPooler

class DeepfakeDetector(nn.Module):
    def __init__(self,
                 videomae_model_name: str ='MCG-NJU/videomae-base',
                 backbone_fau: str = 'swin_transformer_tiny',
                 num_au_classes: int = 8,
                 num_frames: int = 16,
                 au_ckpt_path: str = None,
                 phys_ckpt_path: str = None,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 lora_cfg: dict = None):
        super().__init__()

        self.au_encoder = FAUEncoder(num_classes=num_au_classes, backbone=backbone_fau)
        if au_ckpt_path:
            print(f"Loading AU Checkpoint: {au_ckpt_path}")
            self.au_encoder.load_pretrained(au_ckpt_path)
            for par in self.au_encoder.parameters():
                par.requires_grad=False
        self.phys_encoder = RPPGEncoder(frames=num_frames)
        if phys_ckpt_path:
            print(f"Loading Phys Checkpoint: {phys_ckpt_path}")
            self.phys_encoder.load_pretrained(phys_ckpt_path)
            for par in self.phys_encoder.parameters():
                par.requires_grad=False

        print(f"Loading VideoMAE: {videomae_model_name}")
        self.videomae = VideoMAEModel.from_pretrained(videomae_model_name)
        if lora_cfg:
            print("Applying LoRA...")
            peft_config = LoraConfig(**lora_cfg)
            self.videomae = get_peft_model(self.videomae, peft_config)
            self.videomae.print_trainable_parameters()

        hidden_size = self.videomae.config.hidden_size

        self.au_proj = nn.Linear(self.au_encoder.out_channels, hidden_size)
        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, hidden_size)

        self.segment_embed = nn.Embedding(2, hidden_size)
        self.pos = PositionalEncoding(hidden_size)

        self.attn_pooler = AttentionPooler(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, x_video, return_info=False):
        """
        x_video: [B, 3, T=16, 224, 224]
        """
        B, C, T, H, W = x_video.shape
        device = x_video.device

        x_au_input = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw, cl, cl_edge= self.au_encoder(x_au_input)
        tokens_au = self.au_proj(au_raw)

        tokens_au = tokens_au.view(B, T, -1, tokens_au.shape[-1])
        Num_AU = tokens_au.shape[2]

        tokens_au = tokens_au.permute(0, 2, 1, 3).reshape(B * Num_AU, T, -1)
        tokens_au = self.pos(tokens_au)
        tokens_au = tokens_au.view(B, Num_AU, T, -1).permute(0, 2, 1, 3)
        tokens_au = tokens_au + self.segment_embed(torch.tensor(0, device=device))

        tokens_au = tokens_au.flatten(1, 2)

        rPPG, phys_raw = self.phys_encoder(x_video)
        tokens_phys = self.phys_proj(phys_raw)
        tokens_phys = self.pos(tokens_phys)
        tokens_phys = tokens_phys + self.segment_embed(torch.tensor(1, device=device))
        combined_embeddings = torch.cat([tokens_au, tokens_phys], dim=1)
        outputs = self.videomae.encoder(combined_embeddings)
        last_hidden_state = outputs.last_hidden_state
        features, attn_weights = self.attn_pooler(last_hidden_state)
        logits = self.classifier(self.dropout(self.norm(features)))

        if return_info:
            return {"logits": logits,
                    "attn_weights": attn_weights,
                    "au_embeddings": tokens_au.detach(), "phys_embeddings": tokens_phys.detach(),
                    "au_logits": cl.detach(), "rPPG": rPPG.detach()}
        return logits

# === ЗАПУСК ===
if __name__ == '__main__':
    try:
        print("Инициализация модели...")
        # Укажи тут свои параметры
        model = DeepfakeDetector(num_frames=128,
            videomae_model_name='MCG-NJU/videomae-base',
            num_au_classes=12,
            lora_cfg={
                "inference_mode": False,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["query", "value", "key", "dense"]
            }
        )

        dummy_input = torch.randn(1, 3, 128, 224, 224)
        print("Запуск forward pass...")
        result = model(dummy_input, return_info=True)
        logits, attn_weights, au_embeddings, phys_embeddings,au_logits, rPPG  = (result["logits"], result["attn_weights"], result["au_embeddings"],
         result["phys_embeddings"], result["au_logits"], result["rPPG"])
        print(f"✅ Успех!")
        print(f"Logits shape: {logits.shape} (должно быть [2, 2])")
        print(f"Attention Weights shape: {attn_weights.shape} (должно быть [2, Total_Tokens])")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()