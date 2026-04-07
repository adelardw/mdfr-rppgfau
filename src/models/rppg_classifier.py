# ====================================================================================================================
# Architecture: rPPG Encoder -> Temporal PE -> Attention Pooling -> Classifier
# ====================================================================================================================

import torch
import torch.nn as nn
from src.backbones.rppg_encoder import RPPGEncoder
from src.pooler.attn_pooler import AttentionPooler


class RPPGClassifier(nn.Module):
    def __init__(self,
                 num_frames: int = 32,
                 phys_ckpt_path: str = None,
                 num_classes: int = 2,
                 embed_dim: int = 512,
                 dropout: float = 0.1,
                 full_train: bool = False):
        super().__init__()

        self.phys_encoder = RPPGEncoder(frames=num_frames)
        if phys_ckpt_path:
            print(f"Loading Phys Checkpoint: {phys_ckpt_path}")
            self.phys_encoder.load_pretrained(phys_ckpt_path)
            for par in self.phys_encoder.parameters():
                par.requires_grad = full_train

        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, embed_dim)

        self.attn_pooler = AttentionPooler(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_video, return_info=False):
        """
        x_video: [B, 3, T, H, W]
        """
        # === rPPG branch (video-level) ===
        rPPG, phys_raw = self.phys_encoder(x_video)  # phys_raw: [B, T, 64]
        tokens_phys = self.phys_proj(phys_raw)         # [B, T, embed_dim]

        # === Aggregation + classification ===
        features, attn_weights = self.attn_pooler(tokens_phys)
        logits = self.classifier(self.dropout(self.norm(features)))

        if return_info:
            return {
                "logits": logits,
                "attn_weights": attn_weights,
                "rPPG": rPPG.detach(),
            }
        return logits


if __name__ == '__main__':
    try:
        print("Initializing RPPGClassifier...")
        model = RPPGClassifier(num_frames=32, embed_dim=512)

        dummy_input = torch.randn(2, 3, 32, 224, 224)
        print("Forward pass...")
        result = model(dummy_input, return_info=True)
        print(f"Logits: {result['logits'].shape}")
        print(f"Attn weights: {result['attn_weights'].shape}")
        print(f"rPPG: {result['rPPG'].shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
