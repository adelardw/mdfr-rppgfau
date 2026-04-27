# ====================================================================================================================
# VideoMAe as Feature Extractor
# ====================================================================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel
from src.backbones.fau_encoder import FAUEncoder
from src.backbones.rppg_encoder import RPPGEncoder
from src.backbones.pos import PositionalEncoding
from peft import LoraConfig, get_peft_model

class DeepfakeDetector(nn.Module):
    def __init__(self,
                 videomae_model_name: str ='MCG-NJU/videomae-base',
                 backbone_fau: str = 'resnet50',
                 num_au_classes: int = 8,
                 au_ckpt_path: str | None = './src/backbones/MEGraphAU/checkpoints/MEFARG_resnet50_DISFA_fold2.pth',
                 phys_ckpt_path: str | None = './src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth',
                 embed_dim: int = 512,
                 num_queries: int = 32,
                 num_decoder_layers: int = 6,
                 nhead: int = 8,
                 dropout:int = 0.1,
                 lora_cfg: dict | None = None):
        super().__init__()

        self.au_encoder = FAUEncoder(num_classes=num_au_classes, backbone=backbone_fau)
        if au_ckpt_path:
            self.au_encoder.load_pretrained(au_ckpt_path)
        self.phys_encoder = RPPGEncoder(frames=16)

        if phys_ckpt_path:
            self.phys_encoder.load_pretrained(phys_ckpt_path)

        print(f"Loading VideoMAE: {videomae_model_name}...")
        self.videomae = VideoMAEModel.from_pretrained(videomae_model_name)

        if lora_cfg:
            peft_config = LoraConfig(**lora_cfg)
            self.videomae = get_peft_model(self.videomae, peft_config)
        else:
            for param in self.videomae.parameters():
                param.requires_grad = False

        self.mae_proj = nn.Linear(self.videomae.config.hidden_size, embed_dim)
        self.au_proj = nn.Linear(self.au_encoder.out_channels, embed_dim)
        self.phys_proj = nn.Linear(self.phys_encoder.out_channels, embed_dim)
        self.segment_embed = nn.Embedding(3, embed_dim)

        self.pos = PositionalEncoding(embed_dim)
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=2048, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, x_video):
        """
        x_video: [B, 3, T=16, 224, 224] - Входной видео-клип
        """
        B, C, T, H, W = x_video.shape
        device = x_video.device

        mae_input = x_video.permute(0, 2, 1, 3, 4)
        mae_out = self.videomae(mae_input).last_hidden_state

        tokens_mae = self.mae_proj(mae_out)

        x_au_input = x_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        au_raw, _, _ = self.au_encoder(x_au_input)
        au_raw = au_raw.view(B, T, -1, au_raw.shape[-1])

        au_flat = au_raw.flatten(1, 2)

        tokens_au = self.au_proj(au_flat)

        _, phys_raw = self.phys_encoder(x_video)
        tokens_phys = self.phys_proj(phys_raw)


        tokens_mae = tokens_mae + self.segment_embed(torch.tensor(0, device=device))
        tokens_au = tokens_au + self.segment_embed(torch.tensor(1, device=device))
        tokens_phys = tokens_phys + self.segment_embed(torch.tensor(2, device=device))

        tokens_mae = self.pos(tokens_mae)
        tokens_au = self.pos(tokens_au)
        tokens_phys = self.pos(tokens_phys)

        kv_memory = torch.cat([tokens_mae, tokens_au, tokens_phys], dim=1)

        queries = self.query_embed.expand(B, -1, -1)

        decoded_queries = self.decoder(tgt=queries, memory=kv_memory)
        out_features = decoded_queries.mean(dim=1)
        logits = self.classifier(self.norm(out_features))
        return logits


if __name__ == '__main__':
    model = DeepfakeDetector(num_au_classes=8, embed_dim=512)

    dummy_input = torch.randn(2, 3, 16, 224, 224)

    print("Запуск forward pass...")
    try:
        output = model(dummy_input)
        print(f"✅ Успех! Размер выхода: {output.shape}") # [2, 2]
    except Exception as e:
        print(f"❌ Ошибка: {e}")