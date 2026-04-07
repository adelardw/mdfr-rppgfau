# MDFr-rPPGpFAU - multimodal deepfake recognition based on rppg (video) and fau (image) encoders.

[English](#english) | [Русский](#русский)

---

## English

MDF is a research repository for **multimodal deepfake detection**. The model combines:

- **FAU-based frame-level facial features** (Swin Transformer + GNN)
- **rPPG-based video-level physiological features** (PhysNet)
- **Transformer Decoder (Q-Former)** for cross-modal fusion with learnable queries
- **Attention Pooling + MLP head** for final binary classification (**REAL / FAKE**)

## Architecture

![MDF architecture](docs/architecture.png)

The model processes a video through two complementary branches:

1. **Frame-level branch (FAU):** Extracts facial action unit features from individual frames using a Swin-T backbone with a GNN module (MEGraphAU). Each AU receives its own temporal positional encoding, preserving per-AU temporal dynamics. A segment embedding (0) distinguishes FAU tokens.

2. **Video-level branch (rPPG):** Extracts physiological signals (pulse-related variation) over the full video using PhysNet. Features are projected and enriched with sinusoidal positional encoding. A segment embedding (1) distinguishes rPPG tokens.

Both branches are concatenated and fed as **memory** into a **Transformer Decoder (Q-Former)** with 6 layers and 8 attention heads. A set of **32 learnable query embeddings** cross-attend to the multimodal tokens, producing a fused representation.

The output queries are aggregated by an **Attention Pooler** (3-layer MLP with softmax-weighted averaging), passed through LayerNorm + Dropout, and classified by an **MLP head** into REAL or FAKE.

### Key design choices

- **Per-AU Temporal PE:** Each facial action unit gets its own temporal positional encoding trajectory, allowing the model to track temporal dynamics per AU independently.
- **Segment embeddings:** Distinguish FAU tokens from rPPG tokens within the shared sequence.
- **Q-Former fusion:** Learnable queries cross-attend to both modalities, letting the model discover which multimodal patterns are most discriminative.
- **Frozen encoders (optional):** FAU and rPPG encoders can be frozen or fine-tuned (`full_train` flag).

### Model parameters

| Parameter | Value |
|---|---|
| FAU backbone | Swin Transformer Tiny |
| rPPG backbone | PhysNet |
| Embedding dim | 512 |
| Num queries | 32 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Dropout | 0.3 |
| Num AU classes | 12 |

## Datasets

The model has been trained and evaluated on three public deepfake datasets:

| Dataset | Description |
|---|---|
| **FF++** (FaceForensics++) | Face-swapped videos with multiple manipulation methods |
| **CelebDF** (Celeb-DeepFake) | High-quality celebrity deepfake videos |
| **VCDF-X** | AI-generated face content |

Data split: 70% train / 15% val / 15% test (configurable).

## Ablation Study — Cross-Dataset Evaluation

Models trained on one dataset and evaluated on all three. Metrics: Accuracy / F1 (macro) / AUROC.

> **Methodological note:** All results are reported on the **val split** (10% of the target dataset, random seed=42). For in-domain experiments the checkpoint was selected via early stopping on this same val split, which introduces a mild optimistic bias (standard practice). For cross-dataset experiments the remaining 90% of the target dataset is unused — evaluation is based on ~600–1400 samples depending on the dataset size, which is sufficient for stable estimates but should be kept in mind when interpreting small differences between numbers.

### Accuracy

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8316** | 0.7616 | 0.5479 |
| **CelebDF** | 0.6464 | **0.9342** | 0.4944 |
| **VCDF-X** | 0.4842 | 0.4875 | **0.9269** |
| **Mix (all)** | 0.8057 | **0.9728** | **0.9131** |

### F1 (macro)

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8622** | 0.7320 | 0.5489 |
| **CelebDF** | 0.3914 | **0.9602** | 0.2585 |
| **VCDF-X** | 0.3140 | 0.1692 | **0.9055** |
| **Mix (all)** | 0.8077 | **0.9809** | **0.9073** |

### AUROC

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.9758** | 0.8166 | 0.5768 |
| **CelebDF** | 0.7538 | **0.9999** | 0.3458 |
| **VCDF-X** | 0.4497 | 0.5445 | **0.9799** |
| **Mix (all)** | 0.9351 | **0.9981** | **0.9752** |

### Full metrics per experiment

<details>
<summary>FF++ → FF++</summary>

| Metric | Value |
|---|---:|
| Loss | 0.1478 |
| Accuracy | 0.8316 |
| F1 (macro) | 0.8622 |
| Precision | 0.9033 |
| Recall | 0.8316 |
| AUROC | 0.9758 |

Per-class: crop_img acc=0.9807, f1=0.9634 | real acc=0.6825, f1=0.7611
</details>

<details>
<summary>FF++ → CelebDF</summary>

| Metric | Value |
|---|---:|
| Loss | 0.3493 |
| Accuracy | 0.7616 |
| F1 (macro) | 0.7320 |
| Precision | 0.7116 |
| Recall | 0.7616 |
| AUROC | 0.8166 |

Per-class: crop_img acc=0.9049, f1=0.9238 | real acc=0.6184, f1=0.5402
</details>

<details>
<summary>FF++ → VCDF-X</summary>

| Metric | Value |
|---|---:|
| Loss | 1.2221 |
| Accuracy | 0.5479 |
| F1 (macro) | 0.5489 |
| Precision | 0.5540 |
| Recall | 0.5479 |
| AUROC | 0.5768 |

Per-class: fake acc=0.7828, f1=0.7551 | real acc=0.3130, f1=0.3427
</details>

<details>
<summary>VCDF-X → VCDF-X</summary>

| Metric | Value |
|---|---:|
| Loss | 0.2047 |
| Accuracy | 0.9269 |
| F1 (macro) | 0.9055 |
| Precision | 0.8915 |
| Recall | 0.9269 |
| AUROC | 0.9799 |

Per-class: fake acc=0.9028, f1=0.9387 | real acc=0.9511, f1=0.8722
</details>

<details>
<summary>VCDF-X → FF++</summary>

| Metric | Value |
|---|---:|
| Loss | 2.3390 |
| Accuracy | 0.4842 |
| F1 (macro) | 0.3140 |
| Precision | 0.4896 |
| Recall | 0.4842 |
| AUROC | 0.4497 |

Per-class: crop_img acc=0.2541, f1=0.3898 | real acc=0.7143, f1=0.2381
</details>

<details>
<summary>VCDF-X → CelebDF</summary>

| Metric | Value |
|---|---:|
| Loss | 2.7093 |
| Accuracy | 0.4875 |
| F1 (macro) | 0.1692 |
| Precision | 0.4792 |
| Recall | 0.4875 |
| AUROC | 0.5445 |

Per-class: crop_img acc=0.0672, f1=0.1244 | real acc=0.9079, f1=0.2140
</details>

<details>
<summary>CelebDF → FF++</summary>

| Metric | Value |
|---|---:|
| Loss | 3.2892 |
| Accuracy | 0.6464 |
| F1 (macro) | 0.3914 |
| Precision | 0.5987 |
| Recall | 0.6464 |
| AUROC | 0.7538 |

Per-class: crop_img acc=0.2928, f1=0.4530 | real acc=1.0000, f1=0.3298
</details>

<details>
<summary>CelebDF → CelebDF</summary>

| Metric | Value |
|---|---:|
| Loss | 0.0446 |
| Accuracy | 0.9342 |
| F1 (macro) | 0.9602 |
| Precision | 0.9908 |
| Recall | 0.9342 |
| AUROC | 0.9999 |

Per-class: crop_img acc=1.0000, f1=0.9908 | real acc=0.8684, f1=0.9296
</details>

<details>
<summary>CelebDF → VCDF-X</summary>

| Metric | Value |
|---|---:|
| Loss | 3.8568 |
| Accuracy | 0.4944 |
| F1 (macro) | 0.2585 |
| Precision | 0.4681 |
| Recall | 0.4944 |
| AUROC | 0.3458 |

Per-class: fake acc=0.0352, f1=0.0667 | real acc=0.9535, f1=0.4503
</details>

<details>
<summary>Mix (all) → FF++</summary>

| Metric | Value |
|---|---:|
| Loss | 0.2222 |
| Accuracy | 0.8057 |
| F1 (macro) | 0.8077 |
| Precision | 0.8098 |
| Recall | 0.8057 |
| AUROC | 0.9351 |

Per-class: crop_img acc=0.9448, f1=0.9434 | real acc=0.6667, f1=0.6720
</details>

<details>
<summary>Mix (all) → CelebDF</summary>

| Metric | Value |
|---|---:|
| Loss | 0.0368 |
| Accuracy | 0.9728 |
| F1 (macro) | 0.9809 |
| Precision | 0.9894 |
| Recall | 0.9728 |
| AUROC | 0.9981 |

Per-class: crop_img acc=0.9981, f1=0.9953 | real acc=0.9474, f1=0.9664
</details>

<details>
<summary>Mix (all) → VCDF-X</summary>

| Metric | Value |
|---|---:|
| Loss | 0.1849 |
| Accuracy | 0.9131 |
| F1 (macro) | 0.9073 |
| Precision | 0.9022 |
| Recall | 0.9131 |
| AUROC | 0.9752 |

Per-class: fake acc=0.9338, f1=0.9436 | real acc=0.8924, f1=0.8711
</details>

### Observations

- Each model achieves strong in-domain performance (0.83–0.93 accuracy).
- Single-dataset models generalize poorly to unseen domains — VCDF-X and CelebDF in particular have very different artifact distributions.
- **FF++** model generalizes reasonably to CelebDF (0.76 acc, 0.82 AUROC) but fails on VCDF-X (0.55 acc).
- **Mix training** (FF++ + CelebDF + VCDF-X) achieves the best overall generalization across all three domains: 0.81/0.97/0.91 accuracy and 0.94/0.998/0.975 AUROC respectively.
- The gap between Mix and single-dataset on CelebDF is especially large: 0.97 vs 0.93 (CelebDF-only).

## Repository structure

```text
src/
  models/         # model definitions
  backbones/      # FAU (MEGraphAU) and rPPG (PhysNet) backbones
  data/           # dataset classes and transforms
  config/         # training configs
  experiments/    # experiment configs (YAML)
  train.py        # training entrypoint
  eval.py         # GradCAM visualization
```

## Setup

### 1. Environment

```bash
bash env.sh
```

### 2. FAU weights

Download separately and place at:
```text
src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth
```

### 3. rPPG weights

Weights from the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) repository:
```text
src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth
```

## Training

Training is implemented in **PyTorch Lightning**.

```bash
python src/train.py --config src/experiments/base_config.yml --data_dir <path_to_dataset>
```

Key training parameters:
- Optimizer: AdamW (lr=1e-4 main, lr=1e-5 encoders)
- Scheduler: CosineAnnealingLR
- Early stopping on `val_auc` (patience=15)
- Gradient accumulation: 2 batches
- Max epochs: 1000

## Notes

- This is a **research codebase**, not a production package.
- Some components depend on external model weights.
- The architecture diagram source is available at `docs/architecture.drawio`.

## Citation

If you use this repository, please cite the project page or contact the author directly.

---

## Русский

MDF — исследовательский репозиторий для задачи **детекции дипфейков на основе мультимодальных признаков**. Модель объединяет:

- **FAU-признаки на уровне кадров** (Swin Transformer + GNN)
- **rPPG-признаки на уровне видео** (PhysNet)
- **Transformer Decoder (Q-Former)** для кросс-модального слияния с обучаемыми запросами
- **Attention Pooling + MLP head** для бинарной классификации (**REAL / FAKE**)

## Архитектура

![Архитектура MDF](docs/architecture.png)

Модель обрабатывает видео через две ветки:

1. **Ветка уровня кадров (FAU):** Извлекает признаки единиц действия лица (Action Units) из отдельных кадров с помощью Swin-T и графовой нейросети (MEGraphAU). Каждый AU получает собственное временное позиционное кодирование для отслеживания динамики. Сегментное вложение (0) помечает FAU-токены.

2. **Ветка уровня видео (rPPG):** Извлекает физиологические сигналы (вариации пульса) по всему видео с помощью PhysNet. Признаки проецируются и дополняются синусоидальным позиционным кодированием. Сегментное вложение (1) помечает rPPG-токены.

Обе ветки конкатенируются и подаются как **memory** в **Transformer Decoder (Q-Former)** с 6 слоями и 8 головами внимания. **32 обучаемых запроса (queries)** выполняют кросс-внимание к мультимодальным токенам, формируя объединённое представление.

Выходные запросы агрегируются **Attention Pooler** (3-слойный MLP с softmax-взвешиванием), проходят через LayerNorm + Dropout и классифицируются **MLP-головой** в REAL или FAKE.

### Ключевые решения

- **Per-AU Temporal PE:** Каждая единица действия лица получает собственную траекторию позиционного кодирования.
- **Сегментные вложения:** Различают FAU-токены и rPPG-токены в общей последовательности.
- **Q-Former слияние:** Обучаемые запросы выполняют кросс-внимание к обеим модальностям.
- **Заморозка энкодеров (опционально):** FAU и rPPG энкодеры могут быть заморожены или дообучены (флаг `full_train`).

## Датасеты

Модель обучена и протестирована на трёх датасетах:

| Датасет | Описание |
|---|---|
| **FF++** (FaceForensics++) | Face-swap видео с несколькими методами манипуляции |
| **CelebDF** (Celeb-DeepFake) | Высококачественные дипфейки знаменитостей |
| **VCDF-X** | AI-генерированный контент с лицами |

Разбиение: 70% train / 15% val / 15% test (настраивается).

## Ablation Study — Кросс-датасетная оценка

Модели обучены на одном датасете и протестированы на всех трёх. Метрики: Accuracy / F1 (macro) / AUROC.

> **Методологическое примечание:** Все результаты получены на **val split** (10% целевого датасета, random seed=42). Для in-domain экспериментов чекпоинт выбирался через early stopping на этом же val split — это вносит лёгкий оптимистичный bias (стандартная практика). Для cross-dataset экспериментов оставшиеся 90% целевого датасета не используются — оценка основана на ~600–1400 примерах в зависимости от датасета, чего достаточно для стабильных оценок, но стоит учитывать при интерпретации небольших расхождений между числами.

### Accuracy

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8316** | 0.7616 | 0.5479 |
| **CelebDF** | 0.6464 | **0.9342** | 0.4944 |
| **VCDF-X** | 0.4842 | 0.4875 | **0.9269** |
| **Смесь (все)** | 0.8057 | **0.9728** | **0.9131** |

### F1 (macro)

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8622** | 0.7320 | 0.5489 |
| **CelebDF** | 0.3914 | **0.9602** | 0.2585 |
| **VCDF-X** | 0.3140 | 0.1692 | **0.9055** |
| **Смесь (все)** | 0.8077 | **0.9809** | **0.9073** |

### AUROC

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.9758** | 0.8166 | 0.5768 |
| **CelebDF** | 0.7538 | **0.9999** | 0.3458 |
| **VCDF-X** | 0.4497 | 0.5445 | **0.9799** |
| **Смесь (все)** | 0.9351 | **0.9981** | **0.9752** |

### Наблюдения

- Каждая модель показывает высокие результаты на своём домене (0.83–0.93 accuracy).
- Модели, обученные на одном датасете, плохо переносятся на другие домены — особенно VCDF-X и CelebDF имеют очень разные распределения артефактов.
- Модель **FF++** более-менее переносится на CelebDF (0.76 acc, 0.82 AUROC), но плохо работает на VCDF-X (0.55 acc).
- **Обучение на смеси** (FF++ + CelebDF + VCDF-X) даёт наилучшую генерализацию по всем трём доменам: 0.81 / 0.97 / 0.91 по accuracy и 0.94 / 0.998 / 0.975 по AUROC.
- Разрыв между смесью и одиночным CelebDF особенно заметен: 0.97 vs 0.93 при тестировании на CelebDF.

## Установка

### 1. Окружение

```bash
bash env.sh
```

### 2. Веса FAU

Скачать и поместить в:
```text
src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth
```

### 3. Веса rPPG

Веса из репозитория [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox):
```text
src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth
```

## Обучение

Обучение реализовано на **PyTorch Lightning**.

```bash
python src/train.py --config src/experiments/base_config.yml --data_dir <путь_к_датасету>
```

Основные параметры:
- Оптимизатор: AdamW (lr=1e-4 основной, lr=1e-5 энкодеры)
- Планировщик: CosineAnnealingLR
- Early stopping по `val_auc` (patience=15)
- Gradient accumulation: 2 батча
- Макс. эпох: 1000

## Примечания

- Это **исследовательский код**, а не production-пакет.
- Некоторые компоненты зависят от внешних весов моделей.
- Исходник диаграммы архитектуры: `docs/architecture.drawio`.

## Цитирование

Если вы используете этот репозиторий, ссылайтесь на страницу проекта или свяжитесь с автором.
