import typer
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import List, Optional
from dotenv import load_dotenv
from torch.utils.data import DataLoader, ConcatDataset

from src.models.rppg_lightning import RPPGDeepFakeRecognizer
from src.data.dataset import VideoFolderDataset, split_dataset
from src.data.transforms import VideoTransform

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


@app.command()
def evaluate(
    config_path: str = typer.Option(
        ...,
        "--config_name", "-c",
        help="Путь к .yaml конфигу",
    ),
    dataset_paths: List[str] = typer.Option(
        ...,
        "--dataset_path", "-d",
        help="Пути к папкам с датасетами (можно указывать несколько раз)",
    ),
    ckpt_path: str = typer.Option(
        ...,
        "--ckpt_path", "-ckpt",
        help="Путь к .ckpt чекпоинту обученной модели",
    ),
    split: str = typer.Option(
        "val",
        "--split", "-s",
        help="Какой сплит использовать: val или test",
    ),
    batch_size: int = typer.Option(16, "--batch_size", "-bs", help="Размер батча"),
    num_workers: int = typer.Option(4, "--num_workers", "-nw", help="Кол-во воркеров"),
):
    """
    Evaluate a trained RPPGDeepFakeRecognizer on the val or test split.
    """
    if split not in ("val", "test"):
        typer.echo("--split должен быть 'val' или 'test'")
        raise typer.Exit(1)

    if not os.path.exists(ckpt_path):
        typer.echo(f"Чекпоинт не найден: {ckpt_path}")
        raise FileNotFoundError(ckpt_path)

    if not os.path.exists(config_path):
        config_base_path = os.getenv("EXPERIMENTS_CFG_FOLDER")
        config_path = os.path.join(config_base_path, config_path)
        if not os.path.exists(config_path):
            raise Exception(f"Config not found in {config_base_path}")

    typer.echo(f"Config: {config_path}")
    typer.echo(f"Checkpoint: {ckpt_path}")
    typer.echo(f"Split: {split}")

    default_model_cfg = OmegaConf.create({
        "num_frames": 32,
        "phys_ckpt_path": "./src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth",
        "num_classes": 2,
        "dropout": 0.3,
        "embed_dim": 512,
        "full_train": True,
    })

    file_config = OmegaConf.load(config_path)
    model_cfg = file_config.model_params

    final_config = OmegaConf.merge(default_model_cfg, model_cfg)
    model_cfg = OmegaConf.to_container(final_config, resolve=True)

    num_frames = model_cfg["num_frames"]
    num_classes = model_cfg.get("num_classes", 2)
    typer.echo(f"num_frames={num_frames}, num_classes={num_classes}")

    val_transform = VideoTransform(size=(224, 224), training=False)

    datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            typer.echo(f"Загрузка датасета: {path}")
            datasets.append(
                VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames)
            )
        else:
            typer.echo(f"Путь не найден, пропускаю: {path}")

    if not datasets:
        raise Exception("Ни один датасет не был загружен.")

    full_dataset = ConcatDataset(datasets)
    base_classes = getattr(datasets[0], "classes", None)
    typer.echo(f"Classes: {base_classes}")
    typer.echo(f"Total videos: {len(full_dataset)}")

    _, val_ds, test_ds = split_dataset(
        full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    eval_ds = val_ds if split == "val" else test_ds
    typer.echo(f"Eval split size: {len(eval_ds)}")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    lit_model = RPPGDeepFakeRecognizer.load_from_checkpoint(ckpt_path, map_location="cpu")
    lit_model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    typer.echo(f"Device: {device}")
    lit_model = lit_model.to(device)

    metrics = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes).to(device),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "precision": MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "recall": MulticlassRecall(num_classes=num_classes, average="macro").to(device),
        "auroc": MulticlassAUROC(num_classes=num_classes).to(device),
        "confusion_matrix": MulticlassConfusionMatrix(num_classes=num_classes).to(device),
    }

    per_class_acc = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    per_class_prec = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    per_class_rec = MulticlassRecall(num_classes=num_classes, average=None).to(device)

    total_loss = 0.0
    num_batches = 0
    criterion = torch.nn.CrossEntropyLoss()

    typer.echo(f"\nЗапуск inference на {len(eval_ds)} примерах...")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(eval_loader):
            x = x.to(device)
            y = y.to(device)

            logits = lit_model(x)
            probs = F.softmax(logits, dim=1)

            loss = criterion(logits, y)
            total_loss += loss.item()
            num_batches += 1

            for m in metrics.values():
                if isinstance(m, MulticlassAUROC):
                    m.update(probs, y)
                else:
                    m.update(logits, y)

            per_class_acc.update(logits, y)
            per_class_f1.update(logits, y)
            per_class_prec.update(logits, y)
            per_class_rec.update(logits, y)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(eval_loader):
                typer.echo(f"  batch {batch_idx + 1}/{len(eval_loader)}")

    avg_loss = total_loss / max(num_batches, 1)

    typer.echo("\n" + "=" * 50)
    typer.echo(f"  rPPG EVALUATION RESULTS ({split} split)")
    typer.echo("=" * 50)
    typer.echo(f"  Loss:      {avg_loss:.4f}")
    typer.echo(f"  Accuracy:  {metrics['accuracy'].compute().item():.4f}")
    typer.echo(f"  F1 (macro):{metrics['f1'].compute().item():.4f}")
    typer.echo(f"  Precision: {metrics['precision'].compute().item():.4f}")
    typer.echo(f"  Recall:    {metrics['recall'].compute().item():.4f}")
    typer.echo(f"  AUROC:     {metrics['auroc'].compute().item():.4f}")

    class_names = base_classes if base_classes else [str(i) for i in range(num_classes)]
    pc_acc = per_class_acc.compute()
    pc_f1 = per_class_f1.compute()
    pc_prec = per_class_prec.compute()
    pc_rec = per_class_rec.compute()

    typer.echo("\n--- Per-class ---")
    for i, name in enumerate(class_names):
        typer.echo(
            f"  [{name}]  acc={pc_acc[i]:.4f}  f1={pc_f1[i]:.4f}  "
            f"prec={pc_prec[i]:.4f}  rec={pc_rec[i]:.4f}"
        )

    cm = metrics["confusion_matrix"].compute().cpu().numpy()
    typer.echo("\n--- Confusion Matrix ---")
    header = "          " + "  ".join(f"{n:>8}" for n in class_names)
    typer.echo(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{int(cm[i][j]):>8}" for j in range(num_classes))
        typer.echo(f"  {name:>8}  {row}")

    typer.echo("=" * 50)


if __name__ == "__main__":
    app()
