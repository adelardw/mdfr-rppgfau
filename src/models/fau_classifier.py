# ====================================================================================================================
# Architecture: FAU Encoder (frame-level) -> Attention Pooling -> Classifier
# Input: video [B, 3, T, H, W]
# ====================================================================================================================

import torch
import torch.nn as nn
from src.backbones.fau_encoder import FAUEncoder
from src.pooler.attn_pooler import AttentionPooler


class FAUClassifier(nn.Module):
    def __init__(self,
                 backbone_fau: str = 'swin_transformer_tiny',
                 num_au_classes: int = 12,
                 num_frames: int = 32,
                 au_ckpt_path: str = None,
                 num_classes: int = 2,
                 embed_dim: int = 512,
                 dropout: float = 0.1,
                 full_train: bool = False):
        super().__init__()

        self.au_encoder = FAUEncoder(num_classes=num_au_classes, backbone=backbone_fau)
        if au_ckpt_path:
            print(f"Loading AU Checkpoint: {au_ckpt_path}")
            self.au_encoder.load_pretrained(au_ckpt_path)
            for par in self.au_encoder.parameters():
                par.requires_grad = full_train

        self.au_proj = nn.Linear(self.au_encoder.out_channels, embed_dim)

        self.attn_pooler = AttentionPooler(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_video, return_info=False):
        """
        x_video: [B, 3, T, H, W]
        """
        B, C, T, H, W = x_video.shape

        # Process all frames independently
        x_frames = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw, cl, cl_edge = self.au_encoder(x_frames)   # [B*T, num_au, D]
        tokens_au = self.au_proj(au_raw)                    # [B*T, num_au, embed_dim]

        # Flatten time and AU dims: [B, T*num_au, embed_dim]
        tokens_au = tokens_au.reshape(B, -1, tokens_au.shape[-1])

        features, attn_weights = self.attn_pooler(tokens_au)
        logits = self.classifier(self.dropout(self.norm(features)))

        if return_info:
            return {
                "logits": logits,
                "attn_weights": attn_weights,
                "au_logits": cl.detach(),
            }
        return logits


if __name__ == '__main__':
    try:
        print("Initializing FAUClassifier...")
        model = FAUClassifier(num_frames=32, num_au_classes=12, embed_dim=512)

        dummy_input = torch.randn(2, 3, 32, 224, 224)
        print("Forward pass...")
        result = model(dummy_input, return_info=True)
        print(f"Logits: {result['logits'].shape}")
        print(f"Attn weights: {result['attn_weights'].shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
