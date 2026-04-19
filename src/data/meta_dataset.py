import os
import numpy as np
import cv2
import csv
from PIL import Image
from torch.utils.data import Dataset


def _auto_encode(values: list, sort: bool = True) -> tuple[list[int], dict]:
    """Map string/mixed values to ints. Returns (encoded_list, label_to_int_dict)."""
    unique = sorted(set(str(v) for v in values if v is not None and str(v) != "") ) if sort \
        else list(dict.fromkeys(str(v) for v in values if v is not None and str(v) != ""))
    mapping = {v: i for i, v in enumerate(unique)}
    encoded = [mapping.get(str(v), -1) for v in values]
    return encoded, mapping


class MetaVideoDataset(Dataset):
    """
    CSV-driven video dataset for multi-task learning.

    Expected CSV columns (defaults match train_meta_v5.csv):
        filename   — relative or absolute path to video
        target     — "fake" / "real"  (or any string/int; mapped via target_map)
        gender     — string category   (auto-encoded to int)
        ethnicity  — string category   (auto-encoded to int)
        emotion    — string category   (auto-encoded to int)

    Any aux column absent from the CSV is filled with -1 (masked in loss).
    Exposes .samples = [(path, label), ...] for cluster-based splitting.
    """

    # Default label map for the main binary target
    DEFAULT_TARGET_MAP = {"fake": 0, "real": 1}

    def __init__(
        self,
        csv_path: str,
        video_transform=None,
        frames_per_video: int = 32,
        root_dir: str = "",
        video_col: str = "filename",
        label_col: str = "target",
        gender_col: str | None = "gender",
        ethnicity_col: str | None = "ethnicity",
        emotion_col: str | None = "emotion",
        target_map: dict | None = None,
    ):
        self.video_transform = video_transform
        self.frames_per_video = frames_per_video
        self.root_dir = root_dir

        if target_map is None:
            target_map = self.DEFAULT_TARGET_MAP

        # ---- Read CSV ----
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for col in (video_col, label_col):
            if col not in rows[0]:
                raise ValueError(f"Column '{col}' not found in {csv_path}. Available: {list(rows[0].keys())}")

        # ---- Paths & binary labels ----
        raw_paths = [r[video_col] for r in rows]
        paths = [os.path.join(root_dir, p) if root_dir else p for p in raw_paths]

        raw_targets = [r[label_col] for r in rows]
        # Support both string ("fake"/"real") and numeric (0/1) targets
        try:
            labels = [int(target_map.get(str(v), v)) for v in raw_targets]
        except (ValueError, TypeError):
            labels = [int(v) for v in raw_targets]

        # .samples is required by cluster_split_indices
        self.samples = list(zip(paths, labels))

        # ---- Aux categorical labels (auto-encoded strings → ints) ----
        def _read_col(col):
            if col and col in rows[0]:
                return [r.get(col, "") for r in rows]
            return None

        gender_raw = _read_col(gender_col)
        ethnicity_raw = _read_col(ethnicity_col)
        emotion_raw = _read_col(emotion_col)

        if gender_raw is not None:
            self._gender, self.gender_map = _auto_encode(gender_raw)
        else:
            self._gender = [-1] * len(rows)
            self.gender_map = {}

        if ethnicity_raw is not None:
            self._ethnicity, self.ethnicity_map = _auto_encode(ethnicity_raw)
        else:
            self._ethnicity = [-1] * len(rows)
            self.ethnicity_map = {}

        if emotion_raw is not None:
            self._emotion, self.emotion_map = _auto_encode(emotion_raw)
        else:
            self._emotion = [-1] * len(rows)
            self.emotion_map = {}

        print(
            f"[MetaVideoDataset] {len(self.samples)} samples | "
            f"gender={self.gender_map} | "
            f"ethnicity={self.ethnicity_map} | "
            f"emotion={self.emotion_map}"
        )

    def __len__(self):
        return len(self.samples)

    def _get_dummy_video(self):
        return [Image.new("RGB", (224, 224)) for _ in range(self.frames_per_video)]

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return self._get_dummy_video()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return self._get_dummy_video()

        clip_len = self.frames_per_video
        start_frame = np.random.randint(0, max(total_frames - clip_len, 1)) if total_frames > clip_len else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()

        if not frames:
            return self._get_dummy_video()
        orig_len = len(frames)
        while len(frames) < clip_len:
            frames.append(frames[len(frames) % orig_len])
        return frames[:clip_len]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            frames = self._load_video(path)
        except Exception:
            frames = self._get_dummy_video()

        if self.video_transform is None:
            raise ValueError("video_transform must be provided")

        video_tensor = self.video_transform(frames)
        meta = {
            "label": int(label),
            "gender": int(self._gender[idx]),
            "ethnicity": int(self._ethnicity[idx]),
            "emotion": int(self._emotion[idx]),
        }
        return video_tensor, meta
