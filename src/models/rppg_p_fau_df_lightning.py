import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

from src.models.rppg_p_fau_df import DeepfakeDetectorDF
from src.loss.contrastive import InfoNCEConsistencyLoss, SupConLoss, BatchHardTripletLoss


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


class FauRPPGDeepFakeRecognizerDF(pl.LightningModule):
    """Lightning wrapper around DeepfakeDetectorDF.

    Mirrors FauRPPGDeepFakeRecognizer (rppg_p_fau_lightning.py) but built on
    the DF-trained backbones in src.backbones_df:
      - rPPG = DeepFakesON-Phys  (CelebDF v2 fine-tune)
      - FAU  = OpenGraphAU       (41 AU hybrid pretrain)

    Defaults to **encoders frozen** (full_train=False inside the model). The
    Q-Former + classifier are the only trainable parts in that mode.
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
        gender_loss_weight: float = 0.3,
        ethnicity_loss_weight: float = 0.3,
        emotion_loss_weight: float = 0.3,
        metric_loss_type: str = "triplet",
        metric_loss_weight: float = 0.2,
        triplet_margin: float = 0.3,
        supcon_temperature: float = 0.1,
        memory_bank_size: int = 256,
        embed_dim: int = 512,
        embeddings_dump_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self._dump_dir = embeddings_dump_dir
        if embeddings_dump_dir:
            os.makedirs(embeddings_dump_dir, exist_ok=True)
        self._dump_buf: dict[str, list[torch.Tensor]] = {
            "embedding": [], "fau": [], "phys": [], "rppg": [], "label": [],
        }

        self.model = DeepfakeDetectorDF(**model_params)

        if metric_loss_type == "supcon":
            self.metric_loss = SupConLoss(temperature=supcon_temperature)
        elif metric_loss_type == "triplet":
            self.metric_loss = BatchHardTripletLoss(margin=triplet_margin)
        else:
            self.metric_loss = None

        D = model_params.get("embed_dim", embed_dim)
        self._mb_size = memory_bank_size
        if memory_bank_size > 0 and self.metric_loss is not None:
            self.register_buffer("mb_feats",  torch.zeros(memory_bank_size, D))
            self.register_buffer("mb_labels", torch.full((memory_bank_size,), -1, dtype=torch.long))
            self.register_buffer("mb_ptr",    torch.zeros(1, dtype=torch.long))
            self.register_buffer("mb_filled", torch.zeros(1, dtype=torch.long))

        def _reg_cw(attr, key):
            w = class_weights.get(key) if class_weights else None
            self.register_buffer(attr, w.float() if w is not None else None,
                                 persistent=False)

        _reg_cw("cw_main",      "main")
        _reg_cw("cw_gender",    "gender")
        _reg_cw("cw_ethnicity", "ethnicity")
        _reg_cw("cw_emotion",   "emotion")

        num_aux = sum(
            1 for k in ("num_gender_classes", "num_ethnicity_classes", "num_emotion_classes")
            if model_params.get(k, 0) > 0
        )
        if num_aux > 0:
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            self.register_buffer("log_vars", torch.zeros(4))

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

    # ── Helpers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _mb_enqueue(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        if self._mb_size <= 0 or self.metric_loss is None:
            return
        valid = labels >= 0
        if valid.sum() == 0:
            return
        f = feats[valid].detach()
        y = labels[valid].detach()
        K = self._mb_size
        n = f.size(0)
        ptr = int(self.mb_ptr.item())
        idx = (torch.arange(n, device=f.device) + ptr) % K
        self.mb_feats[idx]  = f.to(self.mb_feats.dtype)
        self.mb_labels[idx] = y
        self.mb_ptr[0] = (ptr + n) % K
        self.mb_filled[0] = torch.clamp(self.mb_filled + n, max=K)

    def _metric_with_bank(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._mb_size <= 0 or int(self.mb_filled.item()) == 0:
            return self.metric_loss(feats, labels)
        K = int(self.mb_filled.item())
        bank_f = self.mb_feats[:K]
        bank_y = self.mb_labels[:K]
        all_f = torch.cat([feats, bank_f.to(feats.dtype)], dim=0)
        all_y = torch.cat([labels, bank_y.to(labels.device)], dim=0)
        return self.metric_loss(all_f, all_y)

    @staticmethod
    def _uncertainty_loss(log_var: torch.Tensor, task_loss: torch.Tensor) -> torch.Tensor:
        return torch.exp(-log_var) * task_loss + 0.5 * log_var

    @staticmethod
    def _unpack_batch(batch):
        x, targets = batch
        if isinstance(targets, dict):
            return x, targets["label"], targets
        return x, targets, {"label": targets}

    def forward(self, x):
        return self.model(x, return_info=False)

    # ── Training ──────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y, meta = self._unpack_batch(batch)
        output = self.model(x, return_info=True)
        logits = output["logits"]

        loss_main = _masked_ce(logits, y, self.cw_main)
        total = self._uncertainty_loss(self.log_vars[0], loss_main)

        if (self.metric_loss is not None
                and "embedding" in output
                and self.hparams.metric_loss_weight > 0):
            emb = output["embedding"]
            m_loss = self._metric_with_bank(emb, y)
            total = total + self.hparams.metric_loss_weight * m_loss
            self._mb_enqueue(emb, y)
            self.log("train_metric", m_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

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

        if self._dump_dir:
            with torch.no_grad():
                au = output["au_raw"]
                if au.dim() == 3:
                    au = au.view(y.size(0), -1, au.size(-2), au.size(-1)).mean(dim=(1, 2))
                else:
                    au = au.view(y.size(0), -1).mean(dim=1, keepdim=True)
                ph = output["phys_raw"]
                ph = ph.mean(dim=1) if ph.dim() == 3 else ph
                rppg = output["rPPG"].view(y.size(0), -1)
                self._dump_buf["embedding"].append(output["embedding"].detach().cpu())
                self._dump_buf["fau"].append(au.detach().cpu())
                self._dump_buf["phys"].append(ph.detach().cpu())
                self._dump_buf["rppg"].append(rppg.detach().cpu())
                self._dump_buf["label"].append(y.detach().cpu())

        self.log("train_loss", total,     prog_bar=True,  on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_ce",   loss_main, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
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

    def on_train_epoch_end(self) -> None:
        if not self._dump_dir or not self._dump_buf["label"]:
            return
        rank = getattr(self.trainer, "global_rank", 0)
        out = {k: torch.cat(v, dim=0) for k, v in self._dump_buf.items()}
        path = os.path.join(
            self._dump_dir, f"epoch_{self.current_epoch:04d}_rank{rank}.pt"
        )
        torch.save(out, path)
        for v in self._dump_buf.values():
            v.clear()

    # ── Validation ────────────────────────────────────────────────────────

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
        self.log("val_loss", val_loss,                prog_bar=True,  sync_dist=True)
        self.log("val_acc",  self.val_acc(logits, y),  prog_bar=True,  sync_dist=True)
        self.log("val_f1",   self.val_f1(logits, y),   prog_bar=True,  sync_dist=True)
        self.log("val_prec", self.val_prec(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_rec",  self.val_rec(logits, y),  prog_bar=False, sync_dist=True)
        self.log("val_auc",  self.val_auc(probs, y),   prog_bar=True,  sync_dist=True)

        return val_loss

    # ── Optimizer ─────────────────────────────────────────────────────────

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
