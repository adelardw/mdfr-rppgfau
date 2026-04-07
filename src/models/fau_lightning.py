import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from src.models.fau_classifier import FAUClassifier


class FAUDeepFakeRecognizer(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
        weight_decay: float = 1e-3,
        T_max: int = 100,
        num_classes: int = 2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FAUClassifier(**model_params)

        self.criterion_ce = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.train_rec = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_prec = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_rec = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, return_info=True)
        logits = output["logits"]

        loss_ce = self.criterion_ce(logits, y)

        self.log("train_loss", loss_ce, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        probs = F.softmax(logits, dim=1)

        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_f1", self.train_f1(logits, y), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_auc", self.train_auc(probs, y), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_prec", self.train_prec(logits, y), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_rec", self.train_rec(logits, y), prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss_ce

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, return_info=False)
        val_loss = self.criterion_ce(logits, y)
        probs = F.softmax(logits, dim=1)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc(logits, y), prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1(logits, y), prog_bar=True, sync_dist=True)
        self.log("val_prec", self.val_prec(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_rec", self.val_rec(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_auc", self.val_auc(probs, y), prog_bar=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        encoder_params = list(self.model.au_encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]

        trainable_encoder_params = [p for p in encoder_params if p.requires_grad]

        param_groups = [
            {"params": other_params, "lr": self.hparams.lr},
        ]
        if trainable_encoder_params:
            param_groups.append(
                {"params": trainable_encoder_params, "lr": self.hparams.encoder_lr},
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max, eta_min=1e-8)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
