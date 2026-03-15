# ====================================================================================================================
# Architecture: [FAU + rPPG] -> Per-AU Temporal PE -> TransformerDecoder (Q-Former) -> Attention Pooling -> Classifier
# ====================================================================================================================

import torch
import torch.nn as nn
from src.backbones.fau_encoder import FAUEncoder
from src.backbones.rppg_encoder import RPPGEncoder
from src.backbones.pos import PositionalEncoding
from src.pooler.attn_pooler import AttentionPooler


class DeepfakeDetector(nn.Module):
    def __init__(self,
                 backbone_fau: str = 'swin_transformer_tiny',
                 num_au_classes: int = 12,
                 num_frames: int = 128,
                 au_ckpt_path: str = None,
                 phys_ckpt_path: str = None,
                 num_classes: int = 2,
                 embed_dim: int = 512,
                 num_queries: int = 32,
                 num_decoder_layers: int = 6,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.au_encoder = FAUEncoder(num_classes=num_au_classes, backbone=backbone_fau)
        if au_ckpt_path:
            print(f"Loading AU Checkpoint: {au_ckpt_path}")
            self.au_encoder.load_pretrained(au_ckpt_path)
            for par in self.au_encoder.parameters():
                par.requires_grad = False

        self.phys_encoder = RPPGEncoder(frames=num_frames)
        if phys_ckpt_path:
            print(f"Loading Phys Checkpoint: {phys_ckpt_path}")
            self.phys_encoder.load_pretrained(phys_ckpt_path)
            for par in self.phys_encoder.parameters():
                par.requires_grad = False

        self.au_proj = nn.Linear(self.au_encoder.out_channels, embed_dim)
        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, embed_dim)

        self.segment_embed = nn.Embedding(2, embed_dim)
        self.pos = PositionalEncoding(embed_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=2048, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.attn_pooler = AttentionPooler(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_video, return_info=False):
        """
        x_video: [B, 3, T, 224, 224]
        """
        B, C, T, H, W = x_video.shape
        device = x_video.device

        # === FAU branch (frame-level) ===
        x_au_input = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw, cl, cl_edge = self.au_encoder(x_au_input)
        tokens_au = self.au_proj(au_raw)

        # Per-AU temporal PE: each AU gets its own temporal trajectory
        tokens_au = tokens_au.view(B, T, -1, tokens_au.shape[-1])
        Num_AU = tokens_au.shape[2]
        tokens_au = tokens_au.permute(0, 2, 1, 3).reshape(B * Num_AU, T, -1)
        tokens_au = self.pos(tokens_au)
        tokens_au = tokens_au.view(B, Num_AU, T, -1).permute(0, 2, 1, 3)
        tokens_au = tokens_au + self.segment_embed(torch.tensor(0, device=device))
        tokens_au = tokens_au.flatten(1, 2)

        # === rPPG branch (video-level) ===
        rPPG, phys_raw = self.phys_encoder(x_video)
        tokens_phys = self.phys_proj(phys_raw)
        tokens_phys = self.pos(tokens_phys)
        tokens_phys = tokens_phys + self.segment_embed(torch.tensor(1, device=device))

        # === Fusion: learnable queries cross-attend to multimodal tokens ===
        kv_memory = torch.cat([tokens_au, tokens_phys], dim=1)
        queries = self.query_embed.expand(B, -1, -1)
        decoded_queries = self.decoder(tgt=queries, memory=kv_memory)

        # === Aggregation + classification ===
        features, attn_weights = self.attn_pooler(decoded_queries)
        logits = self.classifier(self.dropout(self.norm(features)))

        if return_info:
            return {
                "logits": logits,
                "attn_weights": attn_weights,
                "au_embeddings": tokens_au.detach(),
                "phys_embeddings": tokens_phys.detach(),
                "au_logits": cl.detach(),
                "rPPG": rPPG.detach()
            }
        return logits


if __name__ == '__main__':
    try:
        print("Initializing model...")
        model = DeepfakeDetector(
            num_frames=128,
            num_au_classes=12,
            embed_dim=512
        )

        dummy_input = torch.randn(1, 3, 128, 224, 224)
        print("Forward pass...")
        result = model(dummy_input, return_info=True)
        print(f"Logits: {result['logits'].shape}")
        print(f"Attn weights: {result['attn_weights'].shape}")
        print(f"rPPG: {result['rPPG'].shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
