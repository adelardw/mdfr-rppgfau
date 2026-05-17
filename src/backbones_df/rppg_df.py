"""DeepFakesON-Phys encoder (rPPG branch for deepfake detection).

Reimplements the BiDAlab DeepFakesON-Phys network in PyTorch. The original
release is a Keras `.h5` whose topology we extracted from `model_config`:

  Motion branch (input_1 = normalized frame difference)
      Layer1_2  Conv3→32  3×3 valid tanh
      Layer2_2  Conv32→32 3×3 valid tanh
      Layer3_2  AvgPool 2×2
      Conv1x1_1 Conv32→1  1×1 tanh       (attention 1, BN, dropout)
      Layer4_2  Conv1→32  3×3 valid tanh  (lifts attention back to 32 ch)
      Layer5_2  Conv32→64 3×3 valid tanh
      Layer6_2  AvgPool 2×2
      Conv1x1_2 Conv64→1  1×1 tanh       (attention 2, BN)

  Appearance branch (input_2 = raw frame, both 36×36 RGB)
      Layer1    Conv3→32  3×3 valid tanh
      Layer2    Conv32→32 3×3 valid tanh
      Layer3    AvgPool 2×2
      ⊗ att1                              (broadcast multiply, dropout 0.25)
      Layer4    Conv32→32 3×3 valid tanh
      Layer5    Conv32→64 3×3 valid tanh
      Layer6    AvgPool 2×2
      ⊗ att2                              (broadcast multiply, dropout 0.25)
      Flatten → Dense 2304→32 tanh        (Layer8, penultimate features)
                Dense   32→1  sigmoid     (Layer9, real/fake score)

The motion branch *generates* the attention; the appearance branch *uses*
it. Output is a single fake-likelihood per frame.

Returned tensors
----------------
forward(video) -> (rPPG_like, phys_raw)
    rPPG_like : [B, T]      per-frame DF score (sigmoid output)
    phys_raw  : [B, T, 32]  per-frame penultimate features (consumed by
                            the Q-Former fusion head)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class _WidthBN(nn.Module):
    """BatchNorm along the trailing W axis of an NCHW tensor.

    Replicates `tf.keras.layers.BatchNormalization(axis=-1)` when the model
    is configured with `data_format='channels_first'` — which the BiDAlab
    DeepFakesON-Phys release does. In that combination Keras normalizes
    *per-width-column*, producing one (gamma, beta, mean, var) tuple per W
    position rather than per channel.
    """

    def __init__(self, num_features: int, eps: float = 1e-3,
                 momentum: float = 0.01):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))
        # PyTorch BN uses a scalar tensor here; we match that to make the
        # converter's saved `bn{1,2}.num_batches_tracked` load cleanly.
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        if self.training:
            mean = x.mean(dim=(0, 1, 2))                          # [W]
            var  = x.var(dim=(0, 1, 2), unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var,  alpha=self.momentum)
                self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var
        shape = (1, 1, 1, -1)
        x = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return x * self.weight.view(shape) + self.bias.view(shape)


class _DeepFakesOnPhysCAN(nn.Module):
    """Exact PyTorch port of DeepFakesON-Phys (Layer1..Layer9 + BN)."""

    INPUT_SIZE = 36
    PENULTIMATE_DIM = 32

    def __init__(self):
        super().__init__()
        # ── Motion branch (att-generator) ──────────────────────────────
        self.Layer1_2    = nn.Conv2d(3,  32, 3)
        self.Layer2_2    = nn.Conv2d(32, 32, 3)
        self.Layer3_2    = nn.AvgPool2d(2)
        self.Conv1x1_1   = nn.Conv2d(32, 1, 1)
        # 16x16 attention map after Layer3_2; Keras BN normalizes per W column.
        self.bn1         = _WidthBN(16)
        self.Dropout1_2  = nn.Dropout(0.25)
        self.Layer4_2    = nn.Conv2d(1,  32, 3)
        self.Layer5_2    = nn.Conv2d(32, 64, 3)
        self.Layer6_2    = nn.AvgPool2d(2)
        self.Conv1x1_2   = nn.Conv2d(64, 1, 1)
        # 6x6 attention map after Layer6_2.
        self.bn2         = _WidthBN(6)

        # ── Appearance branch (att-consumer) ───────────────────────────
        self.Layer1   = nn.Conv2d(3,  32, 3)
        self.Layer2   = nn.Conv2d(32, 32, 3)
        self.Layer3   = nn.AvgPool2d(2)
        self.Dropout1 = nn.Dropout(0.25)
        self.Layer4   = nn.Conv2d(32, 32, 3)
        self.Layer5   = nn.Conv2d(32, 64, 3)
        self.Layer6   = nn.AvgPool2d(2)
        self.Dropout2 = nn.Dropout(0.25)

        # ── Classifier head (Layer8 penultimate → Layer9 binary) ──────
        # Spatial trace 36 → conv-conv → 32 → pool → 16 → conv-conv → 12
        # → pool → 6.  Final feature map is 6×6×64 = 2304.
        self.Layer8   = nn.Linear(6 * 6 * 64, self.PENULTIMATE_DIM)
        self.Dropout3 = nn.Dropout(0.5)
        self.Layer9   = nn.Linear(self.PENULTIMATE_DIM, 1)

    def forward(self, motion: torch.Tensor, raw: torch.Tensor):
        """
        motion : [N, 3, 36, 36]   normalized ratio-difference frame
        raw    : [N, 3, 36, 36]   z-normalized raw frame
        Returns (score [N, 1] in [0,1], penultimate [N, 32]).
        """
        # Motion branch
        m = torch.tanh(self.Layer1_2(motion))
        m = torch.tanh(self.Layer2_2(m))
        m = self.Layer3_2(m)
        att1 = self.bn1(torch.tanh(self.Conv1x1_1(m)))      # [N, 1, 16, 16]
        m = self.Dropout1_2(att1)
        m = torch.tanh(self.Layer4_2(m))
        m = torch.tanh(self.Layer5_2(m))
        m = self.Layer6_2(m)
        att2 = self.bn2(torch.tanh(self.Conv1x1_2(m)))      # [N, 1, 6, 6]

        # Appearance branch (multiplied by motion-derived attention)
        r = torch.tanh(self.Layer1(raw))
        r = torch.tanh(self.Layer2(r))
        r = self.Layer3(r)
        r = self.Dropout1(r * att1)
        r = torch.tanh(self.Layer4(r))
        r = torch.tanh(self.Layer5(r))
        r = self.Layer6(r)
        r = self.Dropout2(r * att2)                          # [N, 64, 6, 6]

        # Head
        flat  = r.flatten(1)                                 # [N, 2304]
        feat  = torch.tanh(self.Layer8(flat))                # [N, 32] penultimate
        score = torch.sigmoid(self.Layer9(self.Dropout3(feat)))
        return score, feat


class RPPGEncoderDF(nn.Module):
    """Wraps the DF-CAN above with the preprocessing pipeline from BiDAlab's
    `vid_to_deepframes_rawframes.py` (36×36 resize, ratio diff, z-norm)."""

    INPUT_SIZE = _DeepFakesOnPhysCAN.INPUT_SIZE
    DIFF_EPS   = 1e-6
    NORM_EPS   = 0.1

    def __init__(self, frames: int = 128):
        super().__init__()
        self.frames = frames
        self.model = _DeepFakesOnPhysCAN()
        self.out_channels = _DeepFakesOnPhysCAN.PENULTIMATE_DIM   # = 32

    # ── Weight loading ────────────────────────────────────────────────────

    def load_pretrained(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Файл не найден: {checkpoint_path}")
            return False

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        if all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"✅ Веса загружены из {checkpoint_path}")
        if missing:
            print(f"🔹 Missing: {len(missing)} keys (first 5: {missing[:5]})")
        if unexpected:
            print(f"🔹 Unexpected: {len(unexpected)} keys (first 5: {unexpected[:5]})")
        return True

    # ── Pre-processing (mirrors vid_to_deepframes_rawframes.py) ───────────

    def _preprocess(self, video: torch.Tensor):
        """
        video: [B, 3, T, H, W]
        Returns (motion, raw), each [B*T, 3, 36, 36], z-normalized per video.
        """
        B, C, T, H, W = video.shape

        # Spatial resize to 36×36 (per frame).
        x = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(
            x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="area" if H >= self.INPUT_SIZE else "bilinear",
        )
        x = x.view(B, T, C, self.INPUT_SIZE, self.INPUT_SIZE)
        x = x.permute(0, 2, 1, 3, 4).contiguous()           # [B, 3, T, 36, 36]

        # Ratio difference along time, zero-pad final frame.
        x_next = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
        diff = (x_next - x) / (x_next + x + self.DIFF_EPS)

        # Per-video z-normalization (channel-wise stats over T,H,W).
        def _znorm(t: torch.Tensor) -> torch.Tensor:
            mean = t.mean(dim=(2, 3, 4), keepdim=True)
            std  = t.std(dim=(2, 3, 4), keepdim=True)
            return (t - mean) / (std + self.NORM_EPS)

        raw_n  = _znorm(x)
        diff_n = _znorm(diff)

        # Per-frame view: [B*T, 3, 36, 36] each.
        motion = diff_n.permute(0, 2, 1, 3, 4).reshape(
            B * T, C, self.INPUT_SIZE, self.INPUT_SIZE)
        raw    = raw_n.permute(0, 2, 1, 3, 4).reshape(
            B * T, C, self.INPUT_SIZE, self.INPUT_SIZE)
        return motion, raw

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, video: torch.Tensor):
        """
        video : [B, 3, T, H, W]
        Returns
        -------
        rPPG_like : [B, T]      per-frame fake-likelihood (sigmoid)
        phys_raw  : [B, T, 32]  per-frame penultimate features
        """
        B = video.size(0)
        T = video.size(2)
        motion, raw = self._preprocess(video)
        score, feat = self.model(motion, raw)
        rppg_like = score.view(B, T)
        phys_raw  = feat.view(B, T, -1)
        return rppg_like, phys_raw


if __name__ == "__main__":
    enc = RPPGEncoderDF(frames=64)
    dummy = torch.randn(2, 3, 64, 224, 224)
    rppg, feats = enc(dummy)
    print("rppg_like:", rppg.shape, "  phys_raw:", feats.shape,
          "  out_channels:", enc.out_channels)
