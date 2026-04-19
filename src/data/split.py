from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import Subset



def _video_histogram(path: str, n_frames: int = 4, n_bins: int = 32) -> np.ndarray:
    """
    Compute mean per-channel color histogram over ``n_frames`` uniformly
    sampled frames. Returns a 1-D float32 vector of length ``3 * n_bins``.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return np.zeros(3 * n_bins, dtype=np.float32)

    sample_pos = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    hists: List[np.ndarray] = []

    for pos in sample_pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (64, 64))
        channels = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = np.concatenate([
            (cv2.calcHist([c], [0], None, [n_bins], [0, 256]).flatten()
             / (frame.size / 3 + 1e-7))
            for c in channels
        ])
        hists.append(h)

    cap.release()
    return np.mean(hists, axis=0).astype(np.float32) if hists else np.zeros(3 * n_bins, dtype=np.float32)


def _extract_features(
    samples: List[Tuple[str, int]],
    n_frames: int = 4,
) -> np.ndarray:
    """Extract color-histogram features for a list of (path, label) samples."""
    print(f"[ClusterSplit] Extracting features for {len(samples)} videos…")
    return np.stack([_video_histogram(path, n_frames=n_frames) for path, _ in samples])



def _assign_clusters(
    centers: np.ndarray,
    n_clusters: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Assign cluster IDs to train / val / test roles.

    Strategy: find the pair of centroids with maximum Euclidean distance
    → assign them to train and test (maximises domain gap). All remaining
    clusters go to val.

    For n_clusters=2: val is empty (test == val indices are combined by caller).
    """
    dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    i_train, i_test = np.unravel_index(np.argmax(dist), dist.shape)
    i_train, i_test = int(i_train), int(i_test)
    i_val = [c for c in range(n_clusters) if c != i_train and c != i_test]
    return [i_train], i_val, [i_test]


# ---------------------------------------------------------------------------
# Public API: single dataset
# ---------------------------------------------------------------------------

def cluster_split_indices(
    dataset,
    n_clusters: int = 3,
    seed: int = 42,
    n_feature_frames: int = 4,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return ``(train_indices, val_indices, test_indices)`` for *dataset*.

    The dataset must expose a ``.samples`` attribute (list of ``(path, label)``),
    as in :class:`VideoFolderDataset`.

    Args:
        dataset: dataset with ``.samples`` list.
        n_clusters: 2 or 3.  With 2 clusters val_indices == test_indices
            (the single non-train cluster split 50/50).
        seed: KMeans random seed.
        n_feature_frames: frames sampled per video for histogram computation.

    Returns:
        Three lists of integer indices (may be empty for val/test with k=2).
    """
    samples = dataset.samples
    features = _extract_features(samples, n_frames=n_feature_frames)

    print(f"[ClusterSplit] Running KMeans (k={n_clusters})…")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels: np.ndarray = km.fit_predict(features)

    train_cl, val_cl, test_cl = _assign_clusters(km.cluster_centers_, n_clusters)

    if n_clusters == 2 and not val_cl:
        # Split the single non-train cluster evenly into val and test
        non_train = [i for i, l in enumerate(labels) if l in test_cl]
        rng = np.random.default_rng(seed)
        rng.shuffle(non_train)
        mid = len(non_train) // 2
        val_indices = sorted(non_train[:mid])
        test_indices = sorted(non_train[mid:])
        train_indices = [i for i, l in enumerate(labels) if l in train_cl]
    else:
        train_indices = [i for i, l in enumerate(labels) if l in train_cl]
        val_indices   = [i for i, l in enumerate(labels) if l in val_cl]
        test_indices  = [i for i, l in enumerate(labels) if l in test_cl]

    print(
        f"[ClusterSplit] train_clusters={train_cl} → {len(train_indices)} samples | "
        f"val_clusters={val_cl} → {len(val_indices)} | "
        f"test_clusters={test_cl} → {len(test_indices)}"
    )
    return train_indices, val_indices, test_indices



def concat_cluster_split(
    datasets: Sequence,
    n_clusters: int = 3,
    seed: int = 42,
    n_feature_frames: int = 4,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Apply :func:`cluster_split_indices` to each sub-dataset and merge the
    resulting index lists, adjusting for ConcatDataset global offsets.

    Args:
        datasets: sequence of :class:`VideoFolderDataset` instances
            (the ``.datasets`` attribute of a ``ConcatDataset``).
        n_clusters, seed, n_feature_frames: forwarded to
            :func:`cluster_split_indices`.

    Returns:
        ``(train_global_indices, val_global_indices, test_global_indices)``
        ready to pass to ``torch.utils.data.Subset`` on the ``ConcatDataset``.
    """
    all_train, all_val, all_test = [], [], []
    offset = 0

    for ds in datasets:
        tr, va, te = cluster_split_indices(
            ds,
            n_clusters=n_clusters,
            seed=seed,
            n_feature_frames=n_feature_frames,
        )
        all_train.extend(i + offset for i in tr)
        all_val.extend(i + offset for i in va)
        all_test.extend(i + offset for i in te)
        offset += len(ds)

    return all_train, all_val, all_test
