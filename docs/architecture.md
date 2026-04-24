# Architecture: DeepfakeDetector

## Overview

`DeepfakeDetector` (`src/models/rppg_p_fau.py`) is a multimodal deepfake detection model that fuses frame-level facial action unit (FAU) features with video-level physiological (rPPG) features using a Q-Former decoder architecture.

```
Input video [B, 3, T, 224, 224]
        │
        ├──────────────────────────────────────┐
        │  FAU branch                           │  rPPG branch
        ▼                                       ▼
[B·T, 3, 224, 224]                    [B, 3, T, 224, 224]
        │                                       │
   FAUEncoder                             RPPGEncoder
(Swin-T + MEFARG GNN)                   (PhysNet)
        │                                       │
   [B·T, 12, D_fau]                    [B, T, D_phys]
        │                                       │
  au_proj (Linear)                     phys_proj (Linear)
        │                                       │
   [B, T, 12, 512]                      [B, T, 512]
        │                                       │
Per-AU PositionalEncoding              PositionalEncoding
        │                                       │
  segment_embed(0)                      segment_embed(1)
        │                                       │
   [B, T·12, 512]                       [B, T, 512]
        └──────────────┬───────────────────────┘
                       │  concat
                       ▼
              [B, T·12 + T, 512]  ← KV memory
                       │
              TransformerDecoder
               (6 layers, 8 heads)
                  TGT: queries
              [B, 32, 512] ← learnable query_embed
                       │
              decoded queries
              [B, 32, 512]
                       │
              AttentionPooler
           (Linear → Tanh → Linear → Softmax)
                       │
              [B, 512]  +  attn_weights [B, 32]
                       │
              LayerNorm + Dropout
                       │
              classifier (Linear)
                       │
              logits [B, 2]
```

## Components

### FAU Encoder (`src/backbones/fau_encoder.py`)

Wraps **MEFARG** from ME-GraphAU. Processes one frame at a time (batched as `B·T` during training).

- **Backbone:** Swin Transformer Tiny (pretrained on ImageNet, fine-tuned on BP4D for AU detection)
- **GNN head:** Graph-based AU relation module that models AU co-occurrence
- **Output:** `f_v [B·T, N_AU, D_fau]` — per-AU node embeddings after GNN, plus AU logits `cl` and edge features `cl_edge` used as auxiliary outputs
- **`out_channels`:** feature dimension of `f_v` (used to set projection layer input size)

Pretrained checkpoint: `MEFARG_swin_tiny_BP4D_fold1.pth` (12 AUs, BP4D fold 1).

### rPPG Encoder (`src/backbones/rppg_encoder.py`)

Wraps **PhysNet** (`PhysNet_padding_Encoder_Decoder_MAX`) from rPPG-Toolbox.

- **Input:** `[B, 3, T, H, W]` — full video clip
- **Output:** `rPPG [B, T]` (pulse waveform) + `pool [B, T, D_phys]` (spatial features per frame)
- **`out_channels`:** `ConvBlock10.in_channels` — used for the projection layer
- The `pool` output is a pooled spatial feature that captures frame-level visual information correlated with the physiological signal

Pretrained checkpoint: `PURE_PhysNet_DiffNormalized.pth`.

### Positional Encoding (`src/backbones/pos.py`)

Standard sinusoidal positional encoding applied to sequences `[B, L, D]`.

**Per-AU temporal PE:** Before applying PE, FAU features are reshaped so that each AU is treated as an independent sequence of length T. PE is applied along the time axis for each AU independently, then the shape is restored. This preserves per-AU temporal dynamics without mixing AU identities.

### Segment Embeddings

A learned `nn.Embedding(2, embed_dim)` distinguishes the two modalities within the shared token sequence:
- Index `0` — FAU tokens
- Index `1` — rPPG tokens

### Q-Former Decoder

Standard `nn.TransformerDecoder` (PyTorch) with `norm_first=True` (pre-norm).

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Dropout | 0.3 |

32 learnable query embeddings (`nn.Parameter [1, 32, 512]`) act as the **target** sequence (`tgt`). The concatenated FAU + rPPG tokens act as **memory** (key-value). Each query learns to attend to whichever cross-modal combination is most discriminative.

### Attention Pooler (`src/pooler/attn_pooler.py`)

```
x [B, 32, 512]
    │
Linear(512 → 256) → Tanh → Linear(256 → 1)
    │
Softmax(dim=1) → attn_weights [B, 32]
    │
sum(x * attn_weights, dim=1) → [B, 512]
```

Returns both the weighted feature and the attention weights (useful for visualization).

### Classifier

`LayerNorm → Dropout(0.3) → Linear(512 → 2)`

Output: logits for `[FAKE, REAL]` (class 0 = fake/deepfake, class 1 = real).

### Multi-Task Heads (optional)

When `num_gender/ethnicity/emotion_classes > 0`, additional linear classifiers branch off the same pooled feature:

```
normed [B, 512]
  ├── gender_head    → [B, num_gender_classes]
  ├── ethnicity_head → [B, num_ethnicity_classes]
  └── emotion_head   → [B, num_emotion_classes]
```

Labels missing from a sample are set to `-1` and masked out in the loss.

## Lightning Module (`src/models/rppg_p_fau_lightning.py`)

`FauRPPGDeepFakeRecognizer` wraps `DeepfakeDetector` with:

### Loss

**Primary task:** class-weighted cross-entropy on main binary labels.

**Multi-task:** class-weighted cross-entropy per auxiliary head. Task weights are balanced via **uncertainty weighting** (Kendall et al. 2018):

```
total_loss = Σ_i [ exp(-log_var[i]) * L_i + 0.5 * log_var[i] ]
```

`log_var` is a 4-element learnable parameter `[main, gender, ethnicity, emotion]`. When no aux heads exist, it is registered as a fixed zero buffer (no scaling applied).

**Masked CE:** samples with label `-1` are excluded from the loss for that task.

### Optimizer

Two parameter groups with different learning rates:
- `other_params` (fusion, pooler, classifier): `lr` (default 1e-4)
- `trainable_enc` (FAU + rPPG encoder, only when `full_train=true`): `encoder_lr` (default 1e-5)

Scheduler: `CosineAnnealingLR(T_max=100, eta_min=1e-8)`.

### Metrics

Tracked for both train and val:
- Accuracy, F1 macro, Precision macro, Recall macro, AUROC
- Per aux head: accuracy

Checkpointing: best `val_auc` (max), early stopping patience=15.

## Data Pipeline (`src/data/`)

### VideoFolderDataset

- Expects `root/class_name/*.mp4` structure
- Randomly samples a contiguous `frames_per_video`-frame clip from each video
- Falls back to dummy frames on decoding failure

### MetaVideoDataset

- CSV-driven: reads `filename`, binary `target`, and optional `gender`, `ethnicity`, `emotion` columns
- String categories are auto-encoded to integers
- Missing columns filled with `-1` (masked in loss)
- `.samples = [(path, label), ...]` interface required by cluster split

### Processor + FaceDetector

`FaceDetector` uses **MTCNN** (facenet-pytorch) to detect the largest face per frame in a batched call. Detected boxes are propagated forward/backward to fill frames where detection fails. Falls back to center-crop when MTCNN is unavailable.

`Processor` chains face detection → video transform. Exposes `crop_frames()` separately for feature extraction (used in cluster split).

### VideoTransform

Consistent augmentations across all frames in a clip (same random params sampled once per clip):
- Horizontal flip (p=0.5)
- RandomResizedCrop (p=0.5, scale 0.85–1.0)
- Color jitter: brightness/contrast ±0.1, saturation ±0.05 (mild — preserves rPPG signal)
- Gaussian blur (p=0.1)

### Cluster Split (`src/data/split.py`)

Data is split into train / val / test using KMeans (k=3) on per-video **color histograms** of face crops:
1. Sample 4 frames uniformly from each video
2. Run face detection via `Processor`
3. Compute per-channel 32-bin histogram (96-dimensional feature)
4. KMeans k=3 on all videos
5. Assign: the two most distant cluster centroids → train and test; remaining cluster → val

`concat_cluster_split` applies this independently to each sub-dataset and merges indices with a global offset.

## Config Structure

```yaml
model_params:
  backbone_fau: swin_transformer_tiny
  num_frames: 32
  au_ckpt_path: ...
  phys_ckpt_path: ...
  num_classes: 2
  dropout: 0.3
  num_au_classes: 12
  embed_dim: 512
  num_queries: 32
  num_decoder_layers: 6
  nhead: 8
  full_train: true          # freeze or fine-tune encoders
  num_gender_classes: 0     # 0 = disabled
  num_ethnicity_classes: 0
  num_emotion_classes: 0

train_params:
  lr: 0.0001
  encoder_lr: 0.00001       # separate LR for encoders
  weight_decay: 0.05
  T_max: 100
  num_classes: 2

trainer_params:
  max_epochs: 1000
  accumulate_grad_batches: 2
  accelerator: auto
```

## Design choices

**Per-AU temporal PE** allows the model to track each facial action unit's temporal trajectory independently, rather than conflating spatial (AU identity) and temporal dimensions.

**Segment embeddings** give the Q-Former decoder a signal to distinguish modality origin within the shared memory, allowing modality-specific attention patterns.

**Q-Former over simple concatenation** lets the model learn which cross-modal co-occurrences are discriminative for deepfake detection, rather than committing to a fixed fusion rule.

**Uncertainty weighting** removes the need to manually tune auxiliary task loss weights — the model learns them adaptively, with the constraint that adding a task cannot decrease the main task's effective weight arbitrarily.

**Cluster-based split** avoids temporal/identity leakage that random splits can introduce when videos from the same subject or recording session end up in both train and test.
