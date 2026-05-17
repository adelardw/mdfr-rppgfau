"""Convert DeepFakesON-Phys Keras `.h5` weights → PyTorch `state_dict`.

The architecture (extracted from the .h5 `model_config`) has 12 conv/dense
layers plus 2 BatchNormalization layers, named by BiDAlab as Layer1..Layer9,
Layer{1,2,4,5}_2, Conv1x1_{1,2}, batch_normalization_{1,2}. We map them
directly to the PyTorch attributes of
`src.backbones_df.rppg_df._DeepFakesOnPhysCAN`.

Run
---
    python scripts/convert_deepfakeson_phys.py \\
        --src checkpoints/df_phys/DeepFakesON-Phys_CelebDF_V2.h5 \\
        --dst checkpoints/df_phys/DeepFakesON-Phys_CelebDF_V2.pth

    # inspect raw layer/weight inventory:
    python scripts/convert_deepfakeson_phys.py --src <file.h5> --inspect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# Keras layer name → (pytorch attribute, kind)
# kind: "conv"  → kernel (H,W,Cin,Cout) → (Cout,Cin,H,W),  bias
#       "dense" → kernel (in, out)        → (out, in),       bias
#       "bn"    → (gamma, beta, moving_mean, moving_variance)
LAYER_MAP: dict[str, tuple[str, str]] = {
    # Motion (att-generator) branch
    "Layer1_2":              ("Layer1_2",    "conv"),
    "Layer2_2":              ("Layer2_2",    "conv"),
    "Conv1x1_1":             ("Conv1x1_1",   "conv"),
    "batch_normalization_1": ("bn1",         "bn"),
    "Layer4_2":              ("Layer4_2",    "conv"),
    "Layer5_2":              ("Layer5_2",    "conv"),
    "Conv1x1_2":             ("Conv1x1_2",   "conv"),
    "batch_normalization_2": ("bn2",         "bn"),
    # Appearance (att-consumer) branch
    "Layer1":                ("Layer1",      "conv"),
    "Layer2":                ("Layer2",      "conv"),
    "Layer4":                ("Layer4",      "conv"),
    "Layer5":                ("Layer5",      "conv"),
    # Head
    "Layer8":                ("Layer8",      "dense"),
    "Layer9":                ("Layer9",      "dense"),
}


def _open_h5(path: Path):
    try:
        import h5py  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Need `h5py` to read Keras .h5 weights. "
            "Install with: pip install h5py"
        ) from e
    import h5py
    return h5py.File(path, "r")


def _layer_grp(h5, name: str):
    """Walk into `model_weights/<name>/<name>/...` and yield the weight grp."""
    root = h5["model_weights"] if "model_weights" in h5 else h5
    if name not in root:
        return None
    grp = root[name]
    inner = grp[name] if name in grp else grp
    return inner


def inspect(path: Path) -> None:
    with _open_h5(path) as h5:
        root = h5["model_weights"] if "model_weights" in h5 else h5
        names = list(root.attrs.get("layer_names", list(root.keys())))
        print(f"Found {len(names)} top-level entries in {path}:\n")
        for n in names:
            if isinstance(n, bytes):
                n = n.decode()
            inner = _layer_grp(h5, n)
            if inner is None or len(inner) == 0:
                print(f"  {n:30s}  (no weights)")
                continue
            shapes = {k: tuple(inner[k].shape) for k in inner.keys()}
            print(f"  {n:30s}  {shapes}")


def convert(src: Path, dst: Path) -> None:
    state = {}
    matched = 0
    missing_keys: list[str] = []

    with _open_h5(src) as h5:
        for keras_name, (py_name, kind) in LAYER_MAP.items():
            inner = _layer_grp(h5, keras_name)
            if inner is None:
                missing_keys.append(keras_name)
                continue

            if kind in ("conv", "dense"):
                kk = next(k for k in inner.keys() if k.startswith("kernel"))
                kernel = np.asarray(inner[kk])
                bk = next((k for k in inner.keys() if k.startswith("bias")), None)
                bias = np.asarray(inner[bk]) if bk is not None else None
                if kind == "conv":
                    # Keras (H, W, Cin, Cout) → PyTorch (Cout, Cin, H, W)
                    w = torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))).float()
                else:
                    # Keras (in, out) → PyTorch (out, in)
                    w = torch.from_numpy(kernel.T).float()
                state[f"{py_name}.weight"] = w
                if bias is not None:
                    state[f"{py_name}.bias"] = torch.from_numpy(bias).float()

            elif kind == "bn":
                # Keras stores gamma, beta, moving_mean, moving_variance.
                # PyTorch BatchNorm: weight=gamma, bias=beta,
                #                    running_mean=moving_mean,
                #                    running_var =moving_variance.
                def _get(prefix):
                    k = next((k for k in inner.keys() if k.startswith(prefix)), None)
                    return torch.from_numpy(np.asarray(inner[k])).float() if k else None

                gamma  = _get("gamma")
                beta   = _get("beta")
                m_mean = _get("moving_mean")
                m_var  = _get("moving_variance")
                if gamma  is not None: state[f"{py_name}.weight"]       = gamma
                if beta   is not None: state[f"{py_name}.bias"]         = beta
                if m_mean is not None: state[f"{py_name}.running_mean"] = m_mean
                if m_var  is not None: state[f"{py_name}.running_var"]  = m_var
                # PyTorch also wants num_batches_tracked
                state[f"{py_name}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

            else:
                raise SystemExit(f"Unknown kind {kind} for {keras_name}")

            matched += 1
            print(f"  ✓ {keras_name:30s} → {py_name}  ({kind})")

    if missing_keys:
        print(f"\n⚠ Keras layers not found in .h5: {missing_keys}")
        print("  Run --inspect to see what's actually inside the file.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)
    print(f"\nSaved {len(state)} tensors ({matched}/{len(LAYER_MAP)} layers) → {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="path to DeepFakesON-Phys_*.h5")
    ap.add_argument("--dst", type=Path,
                    help="output .pth path (default: alongside .h5 with .pth ext)")
    ap.add_argument("--inspect", action="store_true",
                    help="just print layer names and weight shapes")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"❌ Source file not found: {args.src}", file=sys.stderr)
        return 1

    if args.inspect:
        inspect(args.src)
        return 0

    dst = args.dst or args.src.with_suffix(".pth")
    convert(args.src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
