from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
from PIL import Image

from src.data.transforms import VideoTransform

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class FaceDetector:
    """
    Lightweight MTCNN face detector (facenet-pytorch).

    Args:
        margin: extra pixels added around the detected bounding box.
        min_face_size: minimum face size in pixels accepted by MTCNN.
        device: torch device string for MTCNN inference.
    """

    def __init__(
        self,
        margin: int = 20,
        min_face_size: int = 40,
        device: str = "cpu",
    ) -> None:
        self.margin = margin

        try:
            from facenet_pytorch import MTCNN  # type: ignore

            self._mtcnn = MTCNN(
                keep_all=False,
                select_largest=True,
                min_face_size=min_face_size,
                device=device,
                post_process=False,
            )
        except ImportError:
            warnings.warn(
                "facenet-pytorch is not installed; FaceDetector will use "
                "center-crop fallback for all frames. "
                "Install with: pip install facenet-pytorch"
            )
            self._mtcnn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_batch(self, frames: List[Image.Image]) -> List[Optional[Box]]:
        """
        Detect one face per frame in a single batched forward pass.

        Returns a list of (x1, y1, x2, y2) boxes (with margin applied),
        or None for frames where no face was found.
        """
        if self._mtcnn is None:
            return [None] * len(frames)

        try:
            boxes_batch, _ = self._mtcnn.detect(frames)
            # facenet-pytorch returns a list when input is a list
            if not isinstance(boxes_batch, (list, tuple)):
                boxes_batch = [boxes_batch]
        except Exception:
            boxes_batch = [None] * len(frames)

        results: List[Optional[Box]] = []
        for i, frame in enumerate(frames):
            boxes = boxes_batch[i] if i < len(boxes_batch) else None
            if boxes is None or len(boxes) == 0:
                results.append(None)
            else:
                x1, y1, x2, y2 = (int(v) for v in boxes[0])
                w, h = frame.size
                results.append((
                    max(0, x1 - self.margin),
                    max(0, y1 - self.margin),
                    min(w, x2 + self.margin),
                    min(h, y2 + self.margin),
                ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center_crop_box(frame: Image.Image, ratio: float = 0.8) -> Box:
        w, h = frame.size
        side = int(ratio * min(w, h))
        cx, cy = w // 2, h // 2
        return (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)


class Processor:
    """
    Video processing pipeline: face detection → consistent augmentation.

    Replaces VideoTransform as the ``video_transform`` argument of
    VideoFolderDataset, keeping the same call signature::

        tensor = processor(frames)  # frames: List[PIL.Image], tensor: [C,T,H,W]

    Detection is applied first so augmentations work on already-cropped faces.
    For frames where detection fails, the nearest successfully detected box is
    reused (temporal propagation); if no frame in the clip has a detected face
    the fallback is a center crop.

    Args:
        transform: VideoTransform in train or val mode.
        detector: FaceDetector instance. Pass None to skip face detection
            (equivalent to the old VideoTransform-only behaviour).
    """

    def __init__(
        self,
        transform: VideoTransform,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        self.transform = transform
        self.detector = detector

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Args:
            frames: T PIL Images (one clip).
        Returns:
            Tensor [C, T, H, W].
        """
        if self.detector is not None:
            frames = self._crop_faces(frames)
        return self.transform(frames)


    def _crop_faces(self, frames: List[Image.Image]) -> List[Image.Image]:
        boxes = self.detector.detect_batch(frames)

        last: Optional[Box] = None
        for i, box in enumerate(boxes):
            if box is not None:
                last = box
            elif last is not None:
                boxes[i] = last

        last = None
        for i in range(len(boxes) - 1, -1, -1):
            if boxes[i] is not None:
                last = boxes[i]
            elif last is not None:
                boxes[i] = last

        cropped: List[Image.Image] = []
        for frame, box in zip(frames, boxes):
            if box is None:
                box = FaceDetector._center_crop_box(frame)
            cropped.append(frame.crop(box))

        return cropped
