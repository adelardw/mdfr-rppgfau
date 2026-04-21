import json
import os
import torch
import torch.nn.functional as F
import typer
from omegaconf import OmegaConf
from typing import List, Optional
from dotenv import load_dotenv
from torch.utils.data import DataLoader, ConcatDataset, Subset

from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer
from src.data.dataset import VideoFolderDataset
from src.data.meta_dataset import MetaVideoDataset
from src.data.transforms import VideoTransform
from src.data.processor import FaceDetector, Processor
from src.data.split import concat_cluster_split

from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

load_dotenv()
app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_MODEL_CFG = {
    "backbone_fau": "swin_transformer_tiny",
    "num_frames": 128,
    "au_ckpt_path": "./src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth",
    "phys_ckpt_path": "./src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth",
    "num_classes": 2,
    "dropout": 0.3,
    "num_au_classes": 12,
    "embed_dim": 512,
    "num_queries": 32,
    "num_decoder_layers": 6,
    "nhead": 8,
    "num_gender_classes": 0,
    "num_ethnicity_classes": 0,
    "num_emotion_classes": 0,
}


@app.command()
def evaluate(
    config_path: str = typer.Option(..., "--config_name", "-c", help="Путь к .yaml конфигу"),
    ckpt_path: str = typer.Option(..., "--ckpt_path", "-ckpt", help="Путь к .ckpt чекпоинту"),
    dataset_paths: Optional[List[str]] = typer.Option(
        None, "--dataset_path", "-d",
        help="Папки с обучающими датасетами — воспроизводит cluster split из train.py"
    ),
    eval_dataset_paths: Optional[List[str]] = typer.Option(
        None, "--eval_dataset_path", "-ed",
        help="Отдельные папки для оценки (оцениваются целиком, без split)"
    ),
    meta_csv_paths: Optional[List[str]] = typer.Option(
        None, "--meta_csv_path", "-mc",
        help="CSV-файлы с метаданными (filename, target, gender, ethnicity, emotion)"
    ),
    root_dir: str = typer.Option("", "--root_dir", "-rd", help="Корневая папка для разрешения путей из CSV"),
    video_col: str = typer.Option("filename", "--video_col"),
    label_col: str = typer.Option("target", "--label_col"),
    gender_col: Optional[str] = typer.Option("gender", "--gender_col"),
    ethnicity_col: Optional[str] = typer.Option("ethnicity", "--ethnicity_col"),
    emotion_col: Optional[str] = typer.Option("emotion", "--emotion_col"),
    split: str = typer.Option("test", "--split", "-s", help="val или test (только для cluster split режима с -d)"),
    batch_size: int = typer.Option(16, "--batch_size", "-bs"),
    num_workers: int = typer.Option(4, "--num_workers", "-nw"),
    save_path: Optional[str] = typer.Option(None, "--save_path", "-o", help="Сохранить результаты в JSON"),
):
    """
    Evaluate a trained FauRPPGDeepFakeRecognizer.

    Три режима (выберите один):
      Cluster split (как train.py):  -d /path1 -d /path2  [--split val|test]
      Отдельный eval датасет:        -ed /test_path
      CSV:                           -mc eval.csv
    """
    if split not in ("val", "test"):
        typer.echo("--split должен быть 'val' или 'test'")
        raise typer.Exit(1)

    if not os.path.exists(ckpt_path):
        typer.echo(f"Чекпоинт не найден: {ckpt_path}")
        raise FileNotFoundError(ckpt_path)

    if not os.path.exists(config_path):
        config_base_path = os.getenv("EXPERIMENTS_CFG_FOLDER", "")
        config_path = os.path.join(config_base_path, config_path)
        if not os.path.exists(config_path):
            raise Exception(f"Config not found: {config_path}")

    typer.echo(f"Config:     {config_path}")
    typer.echo(f"Checkpoint: {ckpt_path}")

    file_config = OmegaConf.load(config_path)
    final_config = OmegaConf.merge(OmegaConf.create(DEFAULT_MODEL_CFG), file_config.model_params)
    model_cfg = OmegaConf.to_container(final_config, resolve=True)

    num_frames = model_cfg["num_frames"]
    num_classes = model_cfg.get("num_classes", 2)
    typer.echo(f"num_frames={num_frames}, num_classes={num_classes}")

    # Face detector + val transform — identical pipeline to train.py
    val_transform = Processor(
        transform=VideoTransform(size=(224, 224), training=False),
        detector=FaceDetector(margin=20, device="cpu"),
    )

    meta_kwargs = dict(
        frames_per_video=num_frames,
        root_dir=root_dir,
        video_col=video_col,
        label_col=label_col,
        gender_col=gender_col,
        ethnicity_col=ethnicity_col,
        emotion_col=emotion_col,
    )

    class_names = None

    if meta_csv_paths:
        typer.echo(f"Режим: CSV ({len(meta_csv_paths)} файлов)")
        datasets = [
            MetaVideoDataset(p, video_transform=val_transform, **meta_kwargs)
            for p in meta_csv_paths if os.path.exists(p)
        ]
        if not datasets:
            raise Exception("Ни один CSV-датасет не загружен.")
        eval_ds = ConcatDataset(datasets)

    elif eval_dataset_paths:
        typer.echo(f"Режим: отдельный eval датасет ({len(eval_dataset_paths)} папок)")
        datasets = []
        for path in eval_dataset_paths:
            if os.path.exists(path):
                typer.echo(f"  Загрузка: {path}")
                datasets.append(VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames))
            else:
                typer.echo(f"  Пропуск (не найден): {path}")
        if not datasets:
            raise Exception("Ни один eval датасет не загружен.")
        eval_ds = ConcatDataset(datasets)
        class_names = getattr(datasets[0], "classes", None)

    elif dataset_paths:
        typer.echo(f"Режим: cluster split (воспроизведение train.py, split={split})")
        source_datasets, mirror_datasets = [], []
        for path in dataset_paths:
            if os.path.exists(path):
                typer.echo(f"  Загрузка: {path}")
                source_datasets.append(VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames))
                mirror_datasets.append(VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames))
            else:
                typer.echo(f"  Пропуск (не найден): {path}")
        if not source_datasets:
            raise Exception("Ни один датасет не загружен.")
        class_names = getattr(source_datasets[0], "classes", None)

        typer.echo("Запуск KMeans cluster split...")
        _, val_idx, test_idx = concat_cluster_split(
            source_datasets, n_clusters=3, seed=42, n_feature_frames=4,
            processor=val_transform,
        )
        indices = val_idx if split == "val" else test_idx
        eval_ds = Subset(ConcatDataset(mirror_datasets), indices)
        typer.echo(f"Split={split} → {len(eval_ds)} примеров")

    else:
        raise typer.BadParameter("Укажите -d, -ed или -mc")

    typer.echo(f"Итого для оценки: {len(eval_ds)} примеров")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    typer.echo("Загрузка чекпоинта...")
    lit_model = FauRPPGDeepFakeRecognizer.load_from_checkpoint(
        ckpt_path, model_params=model_cfg, map_location="cpu"
    )
    lit_model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    typer.echo(f"Device: {device}")
    lit_model = lit_model.to(device)

    acc_m   = MulticlassAccuracy(num_classes=num_classes).to(device)
    f1_m    = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    prec_m  = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    rec_m   = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    auroc_m = MulticlassAUROC(num_classes=num_classes).to(device)
    cm_m    = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    pc_acc  = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    pc_f1   = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    pc_prec = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    pc_rec  = MulticlassRecall(num_classes=num_classes, average=None).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    typer.echo(f"\nЗапуск inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            x, targets = batch
            y = targets["label"].to(device) if isinstance(targets, dict) else targets.to(device)
            x = x.to(device)

            logits = lit_model(x)
            probs = F.softmax(logits, dim=1)

            total_loss += criterion(logits, y).item()

            acc_m.update(logits, y)
            f1_m.update(logits, y)
            prec_m.update(logits, y)
            rec_m.update(logits, y)
            auroc_m.update(probs, y)
            cm_m.update(logits, y)
            pc_acc.update(logits, y)
            pc_f1.update(logits, y)
            pc_prec.update(logits, y)
            pc_rec.update(logits, y)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(eval_loader):
                typer.echo(f"  [{batch_idx + 1}/{len(eval_loader)}]")

    avg_loss = total_loss / max(len(eval_loader), 1)
    acc  = acc_m.compute().item()
    f1   = f1_m.compute().item()
    prec = prec_m.compute().item()
    rec  = rec_m.compute().item()
    auc  = auroc_m.compute().item()
    cm   = cm_m.compute().cpu().numpy()
    pca  = pc_acc.compute()
    pcf  = pc_f1.compute()
    pcp  = pc_prec.compute()
    pcr  = pc_rec.compute()

    cn = class_names if class_names else [str(i) for i in range(num_classes)]

    typer.echo("\n" + "=" * 50)
    typer.echo("  EVALUATION RESULTS")
    typer.echo("=" * 50)
    typer.echo(f"  Loss:       {avg_loss:.4f}")
    typer.echo(f"  Accuracy:   {acc:.4f}")
    typer.echo(f"  F1 (macro): {f1:.4f}")
    typer.echo(f"  Precision:  {prec:.4f}")
    typer.echo(f"  Recall:     {rec:.4f}")
    typer.echo(f"  AUROC:      {auc:.4f}")

    typer.echo("\n--- Per-class ---")
    per_class = {}
    for i, name in enumerate(cn):
        typer.echo(f"  [{name}]  acc={pca[i]:.4f}  f1={pcf[i]:.4f}  prec={pcp[i]:.4f}  rec={pcr[i]:.4f}")
        per_class[name] = {"acc": pca[i].item(), "f1": pcf[i].item(), "prec": pcp[i].item(), "rec": pcr[i].item()}

    typer.echo("\n--- Confusion Matrix ---")
    typer.echo("          " + "  ".join(f"{n:>8}" for n in cn))
    for i, name in enumerate(cn):
        row = "  ".join(f"{int(cm[i][j]):>8}" for j in range(num_classes))
        typer.echo(f"  {name:>8}  {row}")
    typer.echo("=" * 50)

    if save_path:
        results = {
            "checkpoint": ckpt_path,
            "config": config_path,
            "num_samples": len(eval_ds),
            "loss": avg_loss,
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
            "auroc": auc,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"\nРезультаты сохранены: {save_path}")


if __name__ == "__main__":
    app()
