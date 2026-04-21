from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
from torchvision.transforms import functional as TF
from PIL import Image

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _frames_to_tensor(frames: List[Image.Image]) -> torch.Tensor:
    """Default tensorization: PIL list → ImageNet-normalized [C, T, H, W]."""
    tensors = [TF.normalize(TF.to_tensor(f), mean=IMAGENET_MEAN, std=IMAGENET_STD)
               for f in frames]
    return torch.stack(tensors).permute(1, 0, 2, 3)


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

    @staticmethod
    def _center_crop_box(frame: Image.Image, ratio: float = 0.8) -> Box:
        w, h = frame.size
        side = int(ratio * min(w, h))
        cx, cy = w // 2, h // 2
        return (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class Processor:
    """
    Video processing pipeline: optional face detection → any video-level transform.

    ``transform`` can be **any callable** that accepts ``List[PIL.Image]``::

        # VideoTransform (built-in)
        Processor(transform=VideoTransform(size=(224, 224), training=True),
                  detector=FaceDetector(margin=20))

        # torchvision per-frame (wrap in lambda)
        import torchvision.transforms as T
        t = T.Compose([T.Resize(224), T.ToTensor()])
        Processor(transform=lambda fs: torch.stack([t(f) for f in fs]).permute(1,0,2,3))

        # No transform — returns face-cropped PIL list
        Processor(detector=FaceDetector())

    Call interface (transformers-style)::

        tensor = processor(frames)                      # positional
        tensor = processor(videos=frames)               # keyword
        crops  = processor(videos=frames, return_tensors=None)  # PIL list

    ``return_tensors`` is only relevant when ``transform=None``:
        - ``"pt"``  → ImageNet-normalised ``Tensor [C, T, H, W]``
        - ``None``  → ``List[PIL.Image]`` (face-cropped)

    When ``transform`` is provided it controls the output format entirely
    (``return_tensors`` is ignored).

    ``crop_frames`` is exposed separately for feature-extraction tasks
    (e.g. cluster split) that need face crops without a full transform.
    """

    def __init__(
        self,
        transform: Optional[Callable[[List[Image.Image]], Any]] = None,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        self.transform = transform
        self.detector = detector


    def crop_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Face-crop only, no transform. Reusable for feature extraction."""
        if self.detector is not None:
            return self._crop_faces(frames)
        return frames

    def __call__(
        self,
        videos: Optional[List[Image.Image]] = None,
        *,
        return_tensors: Optional[str] = "pt",
    ) -> Any:
        """
        Process one video clip.

        Args:
            videos: ``List[PIL.Image]`` — T frames of one clip.
            return_tensors: output format when ``transform=None``.
                ``"pt"`` → ``Tensor [C, T, H, W]`` (ImageNet-normalised).
                ``None`` → ``List[PIL.Image]`` (face-cropped frames).

        Returns:
            Output of ``transform(cropped)``, or — when ``transform`` is None —
            a tensor or PIL list depending on ``return_tensors``.
        """
        if videos is None:
            raise ValueError("Provide frames via positional arg or videos=<list>")

        cropped = self.crop_frames(videos)

        if self.transform is not None:
            return self.transform(cropped)

        if return_tensors == "pt":
            return _frames_to_tensor(cropped)
        return cropped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _crop_faces(self, frames: List[Image.Image]) -> List[Image.Image]:
        boxes = self.detector.detect_batch(frames)

        # Forward propagation: fill None from the nearest earlier valid box
        last: Optional[Box] = None
        for i, box in enumerate(boxes):
            if box is not None:
                last = box
            elif last is not None:
                boxes[i] = last

        # Backward propagation: fill remaining None from the nearest later valid box
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
