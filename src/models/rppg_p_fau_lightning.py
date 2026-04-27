import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from src.models.rppg_p_fau import DeepfakeDetector
from src.loss.contrastive import InfoNCEConsistencyLoss


def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy ignoring samples where label == -1 (unknown)."""
    mask = labels >= 0
    if mask.sum() == 0:
        return logits.sum() * 0.0
    w = weight.to(logits.device) if weight is not None else None
    return F.cross_entropy(logits[mask], labels[mask], weight=w)


class FauRPPGDeepFakeRecognizer(pl.LightningModule):
    """
    Multi-task deepfake detector with:
      - Uncertainty Weighting (Kendall et al. 2018) for task balancing.
        log_var[i] = log(σ_i²); loss_i = exp(-log_var[i]) * L_i + 0.5 * log_var[i]
        Active (trainable) only when aux heads exist; otherwise fixed at 0 → no scaling.
      - Class-weighted CE per task to handle within-task imbalance (e.g. ethnicity).
        Weights computed externally from training data and passed as class_weights dict.
    """

    def __init__(
        self,
        model_params: dict,
        lr: float = 1e-5,
        encoder_lr: float = 1e-5,
        weight_decay: float = 1e-3,
        T_max: int = 10,
        num_classes: int = 2,
        class_weights: dict | None = None,
        # kept for backward-compat with old configs — replaced by uncertainty weighting
        gender_loss_weight: float = 0.3,
        ethnicity_loss_weight: float = 0.3,
        emotion_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = DeepfakeDetector(**model_params)

        # ── Class-weighted CE losses ──────────────────────────────────────────
        # Weights stored as buffers so they move to the right device automatically.
        def _reg_cw(attr, key):
            w = class_weights.get(key) if class_weights else None
            self.register_buffer(attr, w.float() if w is not None else None,
                                 persistent=False)

        _reg_cw("cw_main",      "main")
        _reg_cw("cw_gender",    "gender")
        _reg_cw("cw_ethnicity", "ethnicity")
        _reg_cw("cw_emotion",   "emotion")

        # ── Uncertainty weighting ─────────────────────────────────────────────
        # 4 slots: [main, gender, ethnicity, emotion]
        # Trainable only when aux heads exist (otherwise fixed zeros → no scaling).
        num_aux = sum(
            1 for k in ("num_gender_classes", "num_ethnicity_classes", "num_emotion_classes")
            if model_params.get(k, 0) > 0
        )
        if num_aux > 0:
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            self.register_buffer("log_vars", torch.zeros(4))

        # ── Primary metrics ───────────────────────────────────────────────────
        self.train_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_rec  = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_auc  = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.val_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_rec  = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_auc  = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        # ── Aux head metrics ──────────────────────────────────────────────────
        num_gender    = model_params.get("num_gender_classes", 0)
        num_ethnicity = model_params.get("num_ethnicity_classes", 0)
        num_emotion   = model_params.get("num_emotion_classes", 0)

        if num_gender > 0:
            self.train_gender_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_gender)
            self.val_gender_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_gender)
        if num_ethnicity > 0:
            self.train_ethnicity_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_ethnicity)
            self.val_ethnicity_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_ethnicity)
        if num_emotion > 0:
            self.train_emotion_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_emotion)
            self.val_emotion_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_emotion)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _uncertainty_loss(log_var: torch.Tensor, task_loss: torch.Tensor) -> torch.Tensor:
        """Kendall et al. 2018: exp(-s)*L + 0.5*s  where s = log(σ²)."""
        return torch.exp(-log_var) * task_loss + 0.5 * log_var

    @staticmethod
    def _unpack_batch(batch):
        x, targets = batch
        if isinstance(targets, dict):
            return x, targets["label"], targets
        return x, targets, {"label": targets}

    def forward(self, x):
        return self.model(x, return_info=False)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y, meta = self._unpack_batch(batch)
        output = self.model(x, return_info=True)
        logits = output["logits"]

        loss_main = _masked_ce(logits, y, self.cw_main)
        total = self._uncertainty_loss(self.log_vars[0], loss_main)

        if "gender_logits" in output:
            g_labels = meta.get("gender", torch.full_like(y, -1))
            if isinstance(g_labels, torch.Tensor):
                g_loss = _masked_ce(output["gender_logits"], g_labels, self.cw_gender)
                total = total + self._uncertainty_loss(self.log_vars[1], g_loss)
                self.log("train_gender_loss", g_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
                valid = g_labels >= 0
                if valid.any() and hasattr(self, "train_gender_acc"):
                    self.log("train_gender_acc",
                             self.train_gender_acc(output["gender_logits"][valid], g_labels[valid]),
                             prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        if "ethnicity_logits" in output:
            e_labels = meta.get("ethnicity", torch.full_like(y, -1))
            if isinstance(e_labels, torch.Tensor):
                e_loss = _masked_ce(output["ethnicity_logits"], e_labels, self.cw_ethnicity)
                total = total + self._uncertainty_loss(self.log_vars[2], e_loss)
                self.log("train_ethnicity_loss", e_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
                valid = e_labels >= 0
                if valid.any() and hasattr(self, "train_ethnicity_acc"):
                    self.log("train_ethnicity_acc",
                             self.train_ethnicity_acc(output["ethnicity_logits"][valid], e_labels[valid]),
                             prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        if "emotion_logits" in output:
            em_labels = meta.get("emotion", torch.full_like(y, -1))
            if isinstance(em_labels, torch.Tensor):
                em_loss = _masked_ce(output["emotion_logits"], em_labels, self.cw_emotion)
                total = total + self._uncertainty_loss(self.log_vars[3], em_loss)
                self.log("train_emotion_loss", em_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
                valid = em_labels >= 0
                if valid.any() and hasattr(self, "train_emotion_acc"):
                    self.log("train_emotion_acc",
                             self.train_emotion_acc(output["emotion_logits"][valid], em_labels[valid]),
                             prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self.log("train_loss", total,     prog_bar=True,  on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_ce",   loss_main, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        # Log learned uncertainty params
        self.log("lv_main",      self.log_vars[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lv_gender",    self.log_vars[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lv_ethnicity", self.log_vars[2], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("lv_emotion",   self.log_vars[3], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        probs = F.softmax(logits, dim=1)
        self.log("train_acc",  self.train_acc(logits, y),  prog_bar=True,  on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_f1",   self.train_f1(logits, y),   prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_auc",  self.train_auc(probs, y),   prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_prec", self.train_prec(logits, y), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rec",  self.train_rec(logits, y),  prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return total

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        x, y, meta = self._unpack_batch(batch)
        output = self.model(x, return_info=True)
        logits = output["logits"]

        val_loss = _masked_ce(logits, y, self.cw_main)

        if "gender_logits" in output:
            g_labels = meta.get("gender", torch.full_like(y, -1))
            if isinstance(g_labels, torch.Tensor):
                g_loss = _masked_ce(output["gender_logits"], g_labels, self.cw_gender)
                val_loss = val_loss + self._uncertainty_loss(self.log_vars[1], g_loss)
                valid = g_labels >= 0
                if valid.any() and hasattr(self, "val_gender_acc"):
                    self.log("val_gender_acc",
                             self.val_gender_acc(output["gender_logits"][valid], g_labels[valid]),
                             prog_bar=False, sync_dist=True)

        if "ethnicity_logits" in output:
            e_labels = meta.get("ethnicity", torch.full_like(y, -1))
            if isinstance(e_labels, torch.Tensor):
                e_loss = _masked_ce(output["ethnicity_logits"], e_labels, self.cw_ethnicity)
                val_loss = val_loss + self._uncertainty_loss(self.log_vars[2], e_loss)
                valid = e_labels >= 0
                if valid.any() and hasattr(self, "val_ethnicity_acc"):
                    self.log("val_ethnicity_acc",
                             self.val_ethnicity_acc(output["ethnicity_logits"][valid], e_labels[valid]),
                             prog_bar=False, sync_dist=True)

        if "emotion_logits" in output:
            em_labels = meta.get("emotion", torch.full_like(y, -1))
            if isinstance(em_labels, torch.Tensor):
                em_loss = _masked_ce(output["emotion_logits"], em_labels, self.cw_emotion)
                val_loss = val_loss + self._uncertainty_loss(self.log_vars[3], em_loss)
                valid = em_labels >= 0
                if valid.any() and hasattr(self, "val_emotion_acc"):
                    self.log("val_emotion_acc",
                             self.val_emotion_acc(output["emotion_logits"][valid], em_labels[valid]),
                             prog_bar=False, sync_dist=True)

        probs = F.softmax(logits, dim=1)
        self.log("val_loss", val_loss,               prog_bar=True,  sync_dist=True)
        self.log("val_acc",  self.val_acc(logits, y), prog_bar=True,  sync_dist=True)
        self.log("val_f1",   self.val_f1(logits, y),  prog_bar=True,  sync_dist=True)
        self.log("val_prec", self.val_prec(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_rec",  self.val_rec(logits, y),  prog_bar=False, sync_dist=True)
        self.log("val_auc",  self.val_auc(probs, y),   prog_bar=True,  sync_dist=True)

        return val_loss

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        encoder_params = (list(self.model.au_encoder.parameters())
                          + list(self.model.phys_encoder.parameters()))
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        trainable_enc = [p for p in encoder_params if p.requires_grad]

        param_groups = [{"params": other_params, "lr": self.hparams.lr}]
        if trainable_enc:
            param_groups.append({"params": trainable_enc, "lr": self.hparams.encoder_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max, eta_min=1e-8)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"},
        }
