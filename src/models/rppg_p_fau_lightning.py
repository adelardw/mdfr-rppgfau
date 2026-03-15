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


class FauRPPGDeepFakeRecognizer(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        T_max: int = 10,
        num_classes: int = 2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepfakeDetector(**model_params)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nce = InfoNCEConsistencyLoss()

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
        #nce_logits = output["nce_logits"]
        attn_weights = output["attn_weights"]

        loss_ce = self.criterion_ce(logits, y)

        #loss_nce = self.criterion_nce(nce_logits, attn_weights)

        total_loss = loss_ce # + (self.hparams.lambda_nce * loss_nce)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ce", loss_ce, prog_bar=False, on_step=True, on_epoch=True)
        #self.log("train_nce", loss_nce, prog_bar=False, on_step=True, on_epoch=True)

        probs = F.softmax(logits, dim=1)

        acc = self.train_acc(logits, y)
        f1 = self.train_f1(logits, y)
        prec = self.train_prec(logits, y)
        rec = self.train_rec(logits, y)
        auc = self.train_auc(probs, y)

        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_auc", auc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_prec", prec, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_rec", rec, prog_bar=False, on_step=False, on_epoch=True)


        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, return_info=False)
        logits = output
        val_loss = self.criterion_ce(logits, y)
        probs = F.softmax(logits, dim=1)

        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)
        prec = self.val_prec(logits, y)
        rec = self.val_rec(logits, y)
        auc = self.val_auc(probs, y)


        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_prec", prec, prog_bar=False)
        self.log("val_rec", rec, prog_bar=False)
        self.log("val_auc", auc, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

