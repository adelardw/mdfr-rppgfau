"""Deepfake-trained backbones.

rPPG branch
-----------
DeepFakesON-Phys (BiDAlab, 2021) — fine-tuned for face-forgery detection on
Celeb-DF v2 / DFDC starting from an MTTS-CAN heart-rate estimator. Architecture
is the (single-task) Convolutional Attention Network with temporal-shift
modules already implemented in rPPGToolbox (TS-CAN). Weights are released as
Keras `.h5`; we ship a converter to PyTorch state_dict.

FAU branch
----------
OpenGraphAU (lingjivoo / CVI-SZU) — ME-GraphAU trained on a 2M-image hybrid
corpus covering 41 facial action units. No public AU-encoder *specifically*
fine-tuned on deepfake exists (AUNet/FauForensics did not release weights), so
this is the strongest off-the-shelf AU detector. Used **frozen** as a feature
extractor: the downstream Q-Former learns deepfake-discriminative weighting of
its 41-AU outputs and 27 per-AU node embeddings.
"""

from .fau_df import FAUEncoderDF
from .rppg_df import RPPGEncoderDF

__all__ = ["FAUEncoderDF", "RPPGEncoderDF"]
