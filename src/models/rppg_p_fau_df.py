# ====================================================================================================================
# Deepfake-trained encoders variant of rppg_p_fau.py.
# rPPG = DeepFakesON-Phys (Celeb-DF v2 fine-tune, see src/backbones_df/rppg_df.py)
# FAU  = OpenGraphAU      (41-AU hybrid pretrain,  see src/backbones_df/fau_df.py)
# Both encoders are intended to be **frozen** (full_train=False); only the
# Q-Former fusion + classification heads are trained on the deepfake task.
# ====================================================================================================================

import torch
import torch.nn as nn

from src.backbones_df.fau_df import FAUEncoderDF
from src.backbones_df.rppg_df import RPPGEncoderDF
from src.backbones.pos import PositionalEncoding
from src.pooler.attn_pooler import AttentionPooler


class DeepfakeDetectorDF(nn.Module):
    def __init__(
        self,
        backbone_fau: str = "swin_transformer_base",
        num_au_main_classes: int = 27,
        num_au_sub_classes: int = 14,
        num_frames: int = 128,
        au_ckpt_path: str | None = None,
        phys_ckpt_path: str | None = None,
        num_classes: int = 2,
        embed_dim: int = 512,
        num_queries: int = 32,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        full_train: bool = False,
        num_gender_classes: int = 0,
        num_ethnicity_classes: int = 0,
        num_emotion_classes: int = 0,
    ):
        super().__init__()

        if isinstance(full_train, str):
            full_train = full_train.strip().lower() in ("1", "true", "yes", "y")
        else:
            full_train = bool(full_train)

        # ── FAU branch (OpenGraphAU, frozen by default) ──────────────────
        self.au_encoder = FAUEncoderDF(
            backbone=backbone_fau,
            num_main_classes=num_au_main_classes,
            num_sub_classes=num_au_sub_classes,
        )
        if au_ckpt_path:
            print(f"Loading AU (OpenGraphAU) Checkpoint: {au_ckpt_path}")
            self.au_encoder.load_pretrained(au_ckpt_path)
        for par in self.au_encoder.parameters():
            par.requires_grad = full_train

        # ── rPPG branch (DeepFakesON-Phys, frozen by default) ────────────
        self.phys_encoder = RPPGEncoderDF(frames=num_frames)
        if phys_ckpt_path:
            print(f"Loading Phys (DeepFakesON-Phys) Checkpoint: {phys_ckpt_path}")
            self.phys_encoder.load_pretrained(phys_ckpt_path)
        for par in self.phys_encoder.parameters():
            par.requires_grad = full_train

        # ── Projections ──────────────────────────────────────────────────
        self.au_proj   = nn.Linear(self.au_encoder.out_channels, embed_dim)
        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, embed_dim)

        self.segment_embed = nn.Embedding(2, embed_dim)
        self.pos = PositionalEncoding(embed_dim)

        # ── Q-Former fusion head ─────────────────────────────────────────
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=2048, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.attn_pooler = AttentionPooler(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # ── Auxiliary heads ──────────────────────────────────────────────
        self.gender_head    = nn.Linear(embed_dim, num_gender_classes)    if num_gender_classes    > 0 else None
        self.ethnicity_head = nn.Linear(embed_dim, num_ethnicity_classes) if num_ethnicity_classes > 0 else None
        self.emotion_head   = nn.Linear(embed_dim, num_emotion_classes)   if num_emotion_classes   > 0 else None

    def forward(self, x_video, return_info=True):
        """
        x_video: [B, 3, T, H, W]   (224×224 RGB; the rPPG encoder resizes
                                    internally to 36×36).
        """
        B, C, T, H, W = x_video.shape
        device = x_video.device

        # === FAU branch (frame-level, 224×224) ===
        x_au_input = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw, au_logits = self.au_encoder(x_au_input)        # [B*T, 27, D_au], [B*T, 41]
        tokens_au = self.au_proj(au_raw)                       # [B*T, 27, embed_dim]

        # Per-AU temporal positional encoding: same trick as rppg_p_fau.py —
        # each AU gets its own trajectory through time.
        tokens_au = tokens_au.view(B, T, -1, tokens_au.shape[-1])    # [B, T, 27, D]
        Num_AU = tokens_au.shape[2]
        tokens_au = tokens_au.permute(0, 2, 1, 3).reshape(B * Num_AU, T, -1)
        tokens_au = self.pos(tokens_au)
        tokens_au = tokens_au.view(B, Num_AU, T, -1).permute(0, 2, 1, 3)
        tokens_au = tokens_au + self.segment_embed(torch.tensor(0, device=device))
        tokens_au = tokens_au.flatten(1, 2)                        # [B, T*27, D]

        # === rPPG branch (video-level) ===
        rPPG, phys_raw = self.phys_encoder(x_video)             # [B, T], [B, T, 128]
        tokens_phys = self.phys_proj(phys_raw)                  # [B, T, embed_dim]
        tokens_phys = self.pos(tokens_phys)
        tokens_phys = tokens_phys + self.segment_embed(torch.tensor(1, device=device))

        # === Fusion: learnable queries cross-attend to multimodal tokens ===
        kv_memory = torch.cat([tokens_au, tokens_phys], dim=1)
        queries = self.query_embed.expand(B, -1, -1)
        decoded_queries = self.decoder(tgt=queries, memory=kv_memory)

        # === Aggregation + classification ===
        features, attn_weights = self.attn_pooler(decoded_queries)
        normed = self.dropout(self.norm(features))
        logits = self.classifier(normed)

        if return_info:
            out = {
                "logits": logits,
                "embedding": normed,
                "attn_weights": attn_weights,
                "au_embeddings": tokens_au,
                "phys_embeddings": tokens_phys,
                "au_logits": au_logits,        # 41-class AU scores (aux supervision optional)
                "rPPG": rPPG,                  # per-frame fake-likelihood from DF-Phys
                "au_raw": au_raw,
                "phys_raw": phys_raw,
            }
            if self.gender_head is not None:
                out["gender_logits"] = self.gender_head(normed)
            if self.ethnicity_head is not None:
                out["ethnicity_logits"] = self.ethnicity_head(normed)
            if self.emotion_head is not None:
                out["emotion_logits"] = self.emotion_head(normed)
            return out
        return logits


if __name__ == "__main__":
    try:
        print("Initializing DF detector...")
        model = DeepfakeDetectorDF(
            backbone_fau="resnet50",
            num_frames=64,
            embed_dim=512,
        )
        dummy = torch.randn(1, 3, 64, 224, 224)
        result = model(dummy, return_info=True)
        print(f"Logits: {result['logits'].shape}")
        print(f"AU logits: {result['au_logits'].shape}")
        print(f"rPPG: {result['rPPG'].shape}")
        print(f"phys_raw: {result['phys_raw'].shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
