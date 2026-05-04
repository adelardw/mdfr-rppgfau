"""
Auxiliary supervised losses for the rPPG/FAU encoders, computed on top of the
DeepfakeDetector outputs (`au_logits`, `au_logits_edge`, `rPPG`, ...).

Goal: keep the encoders solving their *original* domain tasks while they are
also fine-tuned for deepfake detection, so they don't collapse into a generic
DF-tuned backbone. Pair this with anchor distillation in the model — distill
preserves features, aux-supervised preserves *behavior* on the source task.

Usage (from a Lightning module):

    self.aux = AuxSupervisedLoss(
        fau_weight=0.5,
        rppg_weight=0.2,
        fau_loss_type="bce",         # AU is multi-label binary by convention
        rppg_loss_type="neg_pearson" # standard rPPG metric
    )
    ...
    aux_losses = self.aux(output, meta)   # {"fau": ..., "rppg": ..., "total": ...}
    total = total + aux_losses["total"]

Either branch is silently skipped when the corresponding label is missing
(meta key absent or set to None) — safe to enable before all labels exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── FAU ──────────────────────────────────────────────────────────────────────

class FAUAuxLoss(nn.Module):
    """
    Loss on the FAU encoder's classification heads (`au_logits`, `au_logits_edge`).

    AU annotation is per-frame multi-label binary in the standard datasets
    (BP4D, DISFA): label[b, t, k] in {0, 1, -1=unknown}. We default to BCE with
    masking; -1 entries don't contribute.

    Logits are produced per-frame inside the model with shape [B*T, K] (or
    similar). The label tensor passed in is expected to flatten to match.
    """

    def __init__(self, loss_type: str = "bce", use_edge: bool = True,
                 edge_weight: float = 0.5):
        super().__init__()
        assert loss_type in ("bce", "ce")
        self.loss_type = loss_type
        self.use_edge = use_edge
        self.edge_weight = edge_weight

    @staticmethod
    def _flatten_match(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Bring labels to logits' leading shape (e.g. [B,T,K] -> [B*T,K]).
        if labels.dim() == logits.dim() + 1:
            labels = labels.flatten(0, 1)
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(f"FAU label/logits batch mismatch: {labels.shape} vs {logits.shape}")
        return logits, labels

    def _bce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, labels = self._flatten_match(logits, labels)
        mask = labels >= 0
        if mask.sum() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())

    def _ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, labels = self._flatten_match(logits, labels)
        mask = labels >= 0
        if mask.sum() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], labels[mask].long())

    def forward(self, output: dict, labels: torch.Tensor | None) -> torch.Tensor:
        if labels is None or "au_logits" not in output:
            return output["logits"].sum() * 0.0  # zero with grad-safe shape

        fn = self._bce if self.loss_type == "bce" else self._ce
        loss = fn(output["au_logits"], labels)
        if self.use_edge and "au_logits_edge" in output:
            loss = loss + self.edge_weight * fn(output["au_logits_edge"], labels)
        return loss


# ── rPPG ─────────────────────────────────────────────────────────────────────

class NegPearsonLoss(nn.Module):
    """
    1 - Pearson correlation between predicted and ground-truth rPPG signals.
    Standard supervision for rPPG networks (Yu et al., PhysNet).

    Inputs of shape [B, T] (or [B, 1, T]). Samples whose label is all-NaN or
    explicitly masked (per-sample mask) are dropped.
    """

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-8)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)

        # Per-sample validity: drop NaN/all-zero or externally masked rows.
        valid = ~torch.isnan(target).any(dim=-1)
        if mask is not None:
            valid = valid & mask.bool()
        if valid.sum() == 0:
            return pred.sum() * 0.0

        p = self._norm(pred[valid])
        t = self._norm(target[valid])
        # Pearson on length-T zero-mean unit-std signals = mean(p*t).
        r = (p * t).mean(dim=-1)
        return (1.0 - r).mean()


class RPPGAuxLoss(nn.Module):
    """
    Loss on the rPPG encoder output. Two flavors:

      - "neg_pearson": supervised against a ground-truth rPPG/PPG signal.
      - "bandpass":    self-supervised — penalize energy outside the
                       physiologic heart-rate band (0.7-2.5 Hz at 30fps).
                       Useful when no GT signal exists; it enforces that
                       the predicted rPPG actually looks like a pulse.
    """

    def __init__(self, loss_type: str = "neg_pearson",
                 fps: float = 30.0,
                 hr_band_hz: tuple[float, float] = (0.7, 2.5)):
        super().__init__()
        assert loss_type in ("neg_pearson", "bandpass")
        self.loss_type = loss_type
        self.fps = fps
        self.hr_band_hz = hr_band_hz
        self._neg_pearson = NegPearsonLoss()

    def _bandpass(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        x = pred - pred.mean(dim=-1, keepdim=True)
        T = x.size(-1)
        spec = torch.fft.rfft(x, dim=-1)
        power = (spec.real ** 2 + spec.imag ** 2)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(power.device)
        in_band = (freqs >= self.hr_band_hz[0]) & (freqs <= self.hr_band_hz[1])
        # Maximize fraction of energy inside the HR band → minimize 1 - frac.
        in_e = power[..., in_band].sum(dim=-1)
        all_e = power.sum(dim=-1) + 1e-8
        return (1.0 - in_e / all_e).mean()

    def forward(self, output: dict,
                target: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if "rPPG" not in output:
            return output["logits"].sum() * 0.0
        pred = output["rPPG"]

        if self.loss_type == "neg_pearson":
            if target is None:
                return pred.sum() * 0.0
            return self._neg_pearson(pred, target, mask)
        return self._bandpass(pred)


# ── Combined wrapper ─────────────────────────────────────────────────────────

class AuxSupervisedLoss(nn.Module):
    """
    Convenience module that bundles FAU + rPPG aux losses with their weights.

    Forward signature:
        aux(output, meta) -> {"fau": Tensor, "rppg": Tensor, "total": Tensor}

    `meta` is a dict (the same one the lightning module already passes around),
    expected (optional) keys:
        meta["au_labels"]        — multi-label AU tensor
        meta["rppg_signal"]      — GT rPPG signal [B, T]
        meta["rppg_mask"]        — bool [B], optional per-sample validity
    """

    def __init__(self,
                 fau_weight: float = 0.5,
                 rppg_weight: float = 0.2,
                 fau_loss_type: str = "bce",
                 rppg_loss_type: str = "neg_pearson",
                 fps: float = 30.0):
        super().__init__()
        self.fau_weight = fau_weight
        self.rppg_weight = rppg_weight
        self.fau = FAUAuxLoss(loss_type=fau_loss_type) if fau_weight > 0 else None
        self.rppg = (RPPGAuxLoss(loss_type=rppg_loss_type, fps=fps)
                     if rppg_weight > 0 else None)

    def forward(self, output: dict, meta: dict | None = None) -> dict:
        meta = meta or {}
        zero = output["logits"].sum() * 0.0
        fau_l = self.fau(output, meta.get("au_labels")) if self.fau is not None else zero
        rppg_l = (self.rppg(output, meta.get("rppg_signal"), meta.get("rppg_mask"))
                  if self.rppg is not None else zero)
        total = self.fau_weight * fau_l + self.rppg_weight * rppg_l
        return {"fau": fau_l, "rppg": rppg_l, "total": total}
