import typer
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset, Subset
import os
import torch
import lightning as pl
from typing import List, Optional
from collections import Counter
from dotenv import load_dotenv
from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer
from src.data.dataset import VideoFolderDataset, split_dataset
from src.data.meta_dataset import MetaVideoDataset
from src.data.transforms import VideoTransform
from src.data.processor import FaceDetector, Processor
from src.data.split import concat_cluster_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def _class_weights(values: list, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: w_i = N / (k * n_i). Handles missing classes gracefully."""
    counts = Counter(v for v in values if v >= 0)
    total = sum(counts.values())
    w = torch.ones(n_classes)
    for c, cnt in counts.items():
        if 0 <= c < n_classes:
            w[c] = total / (n_classes * cnt)
    return w

app = typer.Typer(pretty_exceptions_show_locals=False)
                
load_dotenv()

@app.command()
def train(
    config_path: str = typer.Option(
        ...,
        "--config_name", "-c",
        help="Имя .yaml конфига (.yaml)",
    ),
    num_workers: int = typer.Option(4, '--num_workers', "-nw",help='Кол-во воркеров для лоадеров'),
    dataset_paths: Optional[List[str]] = typer.Option(
        None,
        "--dataset_path", "-d",
        help="Пути к папкам с датасетами (через запятую или несколько -d)"
    ),
    meta_csv_paths: Optional[List[str]] = typer.Option(
        None,
        "--meta_csv_path", "-mc",
        help="Пути к CSV-файлам с метаданными (filename, target, gender, ethnicity, emotion)"
    ),
    root_dir: str = typer.Option("", "--root_dir", "-rd", help="Корневая папка для разрешения относительных путей из CSV"),
    video_col: str = typer.Option("filename", "--video_col", help="Колонка с путём к видео в CSV"),
    label_col: str = typer.Option("target", "--label_col", help="Колонка с бинарной меткой (fake/real или 0/1) в CSV"),
    gender_col: Optional[str] = typer.Option("gender", "--gender_col", help="Колонка gender (None — не используется)"),
    ethnicity_col: Optional[str] = typer.Option("ethnicity", "--ethnicity_col", help="Колонка ethnicity (None — не используется)"),
    emotion_col: Optional[str] = typer.Option("emotion", "--emotion_col", help="Колонка emotion (None — не используется)"),
    batch_size: int = typer.Option(32, "--batch_size", "-bs",
                                   help='Размер батча'),

    val_dataset_paths: Optional[List[str]] = typer.Option(
        None,
        "--val_dataset_path", "-vd",
        help="Пути к папкам с датасетами для val/test (если не указано — используется тренировочный)"
    ),
    load_from_pretrain: Optional[str] = typer.Option(None, "--load_from_pretrain", "-r", help="Путь к .ckpt файлу для возобновления")
):
    """
    Запуск обучения модели FauRPPGDeepFakeRecognizer на нескольких датасетах.

    Режим 1 (папки): -d /path1 -d /path2
    Режим 2 (CSV):   -mc train.csv -mc extra.csv
    """

    use_meta = bool(meta_csv_paths)

    if use_meta:
        # Expand comma-separated CSV paths
        expanded_csv = []
        for p in meta_csv_paths:
            for part in p.split(","):
                part = part.strip()
                if part:
                    expanded_csv.append(part)
        meta_csv_paths = expanded_csv
        typer.echo(f"📋 Режим: CSV-датасет ({len(meta_csv_paths)} файлов)")
        for i, p in enumerate(meta_csv_paths):
            typer.echo(f"   [{i+1}] {p}")
    else:
        if not dataset_paths:
            raise typer.BadParameter("Укажите --dataset_path (-d) или --meta_csv_path (-mc)")
        # Expand comma-separated folder paths
        expanded_paths = []
        for p in dataset_paths:
            for part in p.split(","):
                part = part.strip()
                if part:
                    expanded_paths.append(part)
        dataset_paths = expanded_paths
        typer.echo(f"📋 Датасеты для обучения ({len(dataset_paths)}):")
        for i, p in enumerate(dataset_paths):
            typer.echo(f"   [{i+1}] {p}")

    if val_dataset_paths:
        typer.echo(f"📋 Датасеты для val/test ({len(val_dataset_paths)}):")
        for i, p in enumerate(val_dataset_paths):
            typer.echo(f"   [{i+1}] {p}")
    else:
        typer.echo("📋 Val/test датасет: из тренировочного (по умолчанию)")

    if load_from_pretrain is not None:
        if not os.path.exists(load_from_pretrain):
            typer.echo(f"❌ Ошибка: Чекпоинт не найден: {load_from_pretrain}")
            raise FileNotFoundError(load_from_pretrain)
        typer.echo(f"🔄 Будет выполнено возобновление из: {load_from_pretrain}")

    if not os.path.exists(config_path):
        config_base_path = os.getenv('EXPERIMENTS_CFG_FOLDER')
        typer.echo(f'[CONFIG BASE PATH] base_path={config_base_path}')
        config_path = os.path.join(config_base_path, config_path)

        if not os.path.exists(config_path):
            raise Exception(f'Config not found in {config_base_path}')

    typer.echo(f"📂 Загрузка конфигурации из: {config_path}")

    default_model_cfg = OmegaConf.create({
        'backbone_fau': 'swin_transformer_tiny',
        'num_frames':  128,
        'au_ckpt_path':  './src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth',
        'phys_ckpt_path': './src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth',
        'num_classes':  2,
        'dropout':  0.3,
        'num_au_classes': 12,
        'embed_dim': 512,
        'num_queries': 32,
        'num_decoder_layers': 6,
        'nhead': 8,
        'num_gender_classes': 0,
        'num_ethnicity_classes': 0,
        'num_emotion_classes': 0,
    })

    file_config = OmegaConf.load(config_path)

    model_cfg = file_config.model_params
    train_cfg = file_config.train_params
    trainer_cfg = file_config.trainer_params

    final_config = OmegaConf.merge(default_model_cfg, model_cfg)
    model_cfg = OmegaConf.to_container(final_config, resolve=True)

    num_frames = model_cfg['num_frames']
    typer.echo(f"📐 num_frames={num_frames}")

    detector = FaceDetector(margin=20, device="cpu")
    train_transform = Processor(
        transform=VideoTransform(size=(224, 224), training=True),
        detector=detector,
    )
    val_transform = Processor(
        transform=VideoTransform(size=(224, 224), training=False),
        detector=detector,
    )

    if use_meta:
        # ---- CSV-based multi-task datasets ----
        meta_kwargs = dict(
            frames_per_video=num_frames,
            root_dir=root_dir,
            video_col=video_col,
            label_col=label_col,
            gender_col=gender_col,
            ethnicity_col=ethnicity_col,
            emotion_col=emotion_col,
        )
        train_datasets = []
        for csv_path in meta_csv_paths:
            if os.path.exists(csv_path):
                typer.echo(f"📦 Загрузка CSV датасета: {csv_path}")
                train_datasets.append(MetaVideoDataset(csv_path, video_transform=train_transform, **meta_kwargs))
            else:
                typer.echo(f"⚠️ CSV не найден и будет пропущен: {csv_path}")

        if not train_datasets:
            raise Exception("Ни один CSV-датасет не был загружен. Проверьте пути.")

        full_train_dataset = ConcatDataset(train_datasets)
        typer.echo(f"Total train videos (CSV): {len(full_train_dataset)}")

        # For val/test transforms we create a second copy (same CSV, different transform)
        val_datasets_mirror = [
            MetaVideoDataset(csv_path, video_transform=val_transform, **meta_kwargs)
            for csv_path in meta_csv_paths if os.path.exists(csv_path)
        ]
        full_val_dataset = ConcatDataset(val_datasets_mirror)

        typer.echo("🔍 Running cluster-based split on CSV dataset…")
        train_idx, val_idx, test_idx = concat_cluster_split(
            train_datasets, n_clusters=3, seed=42, n_feature_frames=4
        )

        train_ds = Subset(full_train_dataset, train_idx)
        val_ds   = Subset(full_val_dataset,   val_idx)
        test_ds  = Subset(full_val_dataset,   test_idx)
        typer.echo(f"Cluster split → Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    else:
        # ---- Folder-based datasets (original flow) ----
        train_datasets = []
        for path in dataset_paths:
            if os.path.exists(path):
                typer.echo(f"📦 Загрузка тренировочного датасета из: {path}")
                train_datasets.append(VideoFolderDataset(path, video_transform=train_transform, frames_per_video=num_frames))
            else:
                typer.echo(f"⚠️ Путь не найден и будет пропущен: {path}")

        if not train_datasets:
            raise Exception("Ни один датасет не был загружен. Проверьте пути.")

        full_train_dataset = ConcatDataset(train_datasets)
        base_classes = getattr(train_datasets[0], 'classes', 'Unknown')
        typer.echo(f"Classes (from first DS): {base_classes}")
        typer.echo(f"Total train videos: {len(full_train_dataset)}")

        if val_dataset_paths:
            val_datasets = []
            for path in val_dataset_paths:
                if os.path.exists(path):
                    typer.echo(f"📦 Загрузка val/test датасета из: {path}")
                    val_datasets.append(VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames))
                else:
                    typer.echo(f"⚠️ Путь не найден и будет пропущен: {path}")

            if not val_datasets:
                raise Exception("Ни один val/test датасет не был загружен. Проверьте пути.")

            full_val_dataset = ConcatDataset(val_datasets)
            typer.echo(f"Total val/test videos: {len(full_val_dataset)}")

            train_ds = full_train_dataset
            val_ds, test_ds, _ = split_dataset(full_val_dataset, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
            typer.echo(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        else:
            full_val_dataset = ConcatDataset([
                VideoFolderDataset(path, video_transform=val_transform, frames_per_video=num_frames)
                for path in dataset_paths if os.path.exists(path)
            ])

            typer.echo("🔍 Running cluster-based split (KMeans on visual features)…")
            train_idx, val_idx, test_idx = concat_cluster_split(
                train_datasets, n_clusters=3, seed=42, n_feature_frames=4
            )

            train_ds = Subset(full_train_dataset, train_idx)
            val_ds   = Subset(full_val_dataset,   val_idx)
            test_ds  = Subset(full_val_dataset,   test_idx)
            typer.echo(f"Cluster split → Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True)


    # ── Class weights (only for meta CSV mode) ───────────────────────────────
    class_weights = None
    if use_meta and train_datasets:
        all_labels     = [int(lbl) for ds in train_datasets for _, lbl in ds.samples]
        all_gender     = [v for ds in train_datasets for v in ds._gender]
        all_ethnicity  = [v for ds in train_datasets for v in ds._ethnicity]
        all_emotion    = [v for ds in train_datasets for v in ds._emotion]

        ng = model_cfg.get("num_gender_classes", 0)
        ne = model_cfg.get("num_ethnicity_classes", 0)
        nem = model_cfg.get("num_emotion_classes", 0)

        class_weights = {"main": _class_weights(all_labels, 2)}
        if ng  > 0: class_weights["gender"]    = _class_weights(all_gender,    ng)
        if ne  > 0: class_weights["ethnicity"] = _class_weights(all_ethnicity, ne)
        if nem > 0: class_weights["emotion"]   = _class_weights(all_emotion,   nem)

        typer.echo("⚖️  Class weights:")
        for k, w in class_weights.items():
            typer.echo(f"   {k}: {w.numpy().round(3).tolist()}")

    lit_model = FauRPPGDeepFakeRecognizer(
        model_params=model_cfg,
        class_weights=class_weights,
        **train_cfg)

    print("\n🔍 CHECKING TRAINABLE PARAMS:")
    trainable_layers = []
    for name, param in lit_model.model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
        else:
            print(f'ПУПУПУУУУ ХУЕТА: {name, param.requires_grad}')

    typer.echo(f"Trainable layers ({len(trainable_layers)}):")
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath='checkpoints/',
    #    filename='best-{epoch:02d}-{val_auc:.4f}',
    #    monitor='val_auc',
    #    mode='max',
    #    save_top_k=1,
    #    save_last=True,
    #    verbose=True
    #)


    early_stop_callback = EarlyStopping(
        monitor='val_auc',
        min_delta=0.001,
        patience=15,
        verbose=True,
        mode='max'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [early_stop_callback, lr_monitor]

    trainer = pl.Trainer(callbacks=callbacks,strategy="ddp_find_unused_parameters_true",
                         **trainer_cfg)
    typer.echo("🚀 Запуск обучения Lightning...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=load_from_pretrain)


if __name__ == "__main__":
    app()