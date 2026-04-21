import random
import torch
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoTransform:
    """
    Video-level augmentations with consistent random parameters across all frames.

    Design constraints:
      - rPPG (PhysNet): relies on temporal color dynamics (blood volume pulse).
        Color augmentations must be constant across frames to preserve temporal signal.
        Saturation and hue are kept very mild to avoid distorting color channels.
      - FAU (Swin/MEGraphAU): processes each frame independently for facial action units.
        Geometric augmentations (flip, crop) are safe as long as face stays visible.
    """

    def __init__(
        self,
        size=(224, 224),
        training=True,
        horizontal_flip_p=0.5,
        crop_scale=(0.85, 1.0),
        crop_ratio=(0.9, 1.1),
        crop_p=0.5,
        color_jitter_p=0.3,
        brightness_range=0.1,
        contrast_range=0.1,
        saturation_range=0.05,
        blur_p=0.1,
        blur_kernel=5,
        blur_sigma=(0.1, 2.0),
    ):
        self.size = size
        self.training = training
        self.horizontal_flip_p = horizontal_flip_p
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.crop_p = crop_p
        self.color_jitter_p = color_jitter_p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.blur_p = blur_p
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def __call__(self, frames: list) -> torch.Tensor:
        """
        Args:
            frames: list of PIL Images (T frames)
        Returns:
            Tensor [C, T, H, W]
        """
        if self.training:
            return self._train(frames)
        return self._val(frames)

    def _val(self, frames):
        tensors = []
        for f in frames:
            f = TF.resize(f, self.size)
            t = TF.to_tensor(f)
            t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tensors.append(t)
        return torch.stack(tensors).permute(1, 0, 2, 3)

    def _train(self, frames):
        # Normalize to consistent size before augmentation.
        # Face crops from MTCNN have variable pixel dimensions (bounding boxes change
        # as the face moves). RandomResizedCrop.get_params returns absolute pixel
        # coordinates based on frames[0]; applying them to frames with different
        # dimensions causes PIL to pad out-of-bounds regions with black, introducing
        # artificial temporal variations that corrupt the rPPG signal.
        pre_size = (int(self.size[0] * 1.15), int(self.size[1] * 1.15))
        frames = [TF.resize(f, pre_size) for f in frames]

        # --- sample ALL random params ONCE for the whole clip ---
        do_flip = random.random() < self.horizontal_flip_p

        do_crop = random.random() < self.crop_p
        if do_crop:
            crop_params = RandomResizedCrop.get_params(
                frames[0], scale=self.crop_scale, ratio=self.crop_ratio
            )

        do_color = random.random() < self.color_jitter_p
        if do_color:
            br = self.brightness_range
            cr = self.contrast_range
            sr = self.saturation_range
            brightness_factor = random.uniform(1 - br, 1 + br)
            contrast_factor = random.uniform(1 - cr, 1 + cr)
            saturation_factor = random.uniform(1 - sr, 1 + sr)

        do_blur = random.random() < self.blur_p
        if do_blur:
            sigma = random.uniform(*self.blur_sigma)

        # --- apply consistently to every frame ---
        tensors = []
        for f in frames:
            if do_flip:
                f = TF.hflip(f)
            if do_crop:
                i, j, h, w = crop_params
                f = TF.resized_crop(f, i, j, h, w, self.size)
            else:
                f = TF.resize(f, self.size)
            if do_color:
                f = TF.adjust_brightness(f, brightness_factor)
                f = TF.adjust_contrast(f, contrast_factor)
                f = TF.adjust_saturation(f, saturation_factor)
            if do_blur:
                f = TF.gaussian_blur(f, kernel_size=self.blur_kernel, sigma=sigma)

            t = TF.to_tensor(f)
            t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            tensors.append(t)

        return torch.stack(tensors).permute(1, 0, 2, 3)
