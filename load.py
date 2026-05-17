import typer
import gdown
import os
import zipfile
import subprocess
import sys

app = typer.Typer()

# База ссылок
WEIGHTS_DB = {
    "backbone": {
        "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "swin-tiny": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        "swin-small": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
        "swin-base": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth",
    },
    "fau-bp4d": {
        "resnet50": "1EiQd6q7x1bEO6JBLi3s2y5348EuVdP3L",
        "resnet101": "1Ti0auMA5o94toJfszuHoMlSlWUumm9L8",
        "swin-tiny": "1BT4n7_5Wr6bGxHWVf3WrT7uBT0Zg9B5c",
        "swin-small": "1EiQd6q7x1bEO6JBLi3s2y5348EuVdP3L",
        "swin-base": "1gYVHRjIj6ounxtTdXMje4WNHFMfCTVF9",
    },
    "fau-disfa": {
        "resnet50": "1V-imbmhg-OgcP2d9SETT5iswNtCA0f8_",
        "swin-base": "1T44KPDaUhi4J_C-fWa6RxXNkY3yoDwIi",
    },
    # ── Deepfake-trained backbones (used by src/backbones_df) ──────────────
    # rPPG: DeepFakesON-Phys (BiDAlab) — single-frame CAN fine-tuned for
    # face-forgery detection, released as Keras .h5 (~1.7 MB each).
    "rppg-df": {
        "celebdf-v2":   "https://raw.githubusercontent.com/BiDAlab/DeepFakesON-Phys/main/pretrained%20models/DeepFakesON-Phys_CelebDF_V2.h5",
        "dfdc-preview": "https://raw.githubusercontent.com/BiDAlab/DeepFakesON-Phys/main/pretrained%20models/DeepFakesON-Phys_DFDC_Preview.h5",
    },
    # FAU: OpenGraphAU (lingjivoo / CVI-SZU) — 41-AU ME-GraphAU pretrained on
    # a 2M-image hybrid corpus. Swin-Base second-stage checkpoint is not
    # publicly released ("-" in README); only ResNet-50 stage 1 / stage 2.
    "fau-df": {
        "resnet50-stage1": "11xh9r2e4qCpWEtQ-ptJGWut_TQ0_AmSp",
        "resnet50-stage2": "1UMnpbj_YKlqHF1m0DHV0KYD3qmcOmeXp",
    },
}

# Per-category target directory and download mode (direct URL vs gdrive).
TARGET_DIR = {
    "backbone": ("src/backbones/MEGraphAU/checkpoints", False),
    "fau-bp4d": ("src/backbones/MEGraphAU/checkpoints", True),
    "fau-disfa": ("src/backbones/MEGraphAU/checkpoints", True),
    "rppg-df":  ("checkpoints/df_phys",   False),  # direct URL (GitHub raw)
    "fau-df":   ("checkpoints/opengraphau", True), # Google Drive
}


def download_file(url_or_id, output_dir, is_gdrive=False):
    os.makedirs(output_dir, exist_ok=True)
    if not is_gdrive:
        # wget -nc не перекачивает файл, если он уже есть
        subprocess.run(["wget", "-nc", url_or_id, "-P", output_dir])
    else:
        # gdown скачивает с Google Drive
        out_path = gdown.download(f'https://drive.google.com/uc?id={url_or_id}',
                                  output=output_dir + "/", quiet=False, fuzzy=True)
        return out_path
    return None


@app.command()
def download(category: str, model: str):
    if category not in WEIGHTS_DB:
        typer.secho(f"❌ Неизвестная категория: {category}", fg="red")
        typer.echo(f"   Доступные: {', '.join(WEIGHTS_DB.keys())}")
        raise typer.Exit(1)
    if model not in WEIGHTS_DB[category]:
        typer.secho(f"❌ Модель '{model}' не найдена в {category}", fg="red")
        typer.echo(f"   Доступные: {', '.join(WEIGHTS_DB[category].keys())}")
        raise typer.Exit(1)

    output_dir, is_gdrive = TARGET_DIR[category]
    os.makedirs(output_dir, exist_ok=True)

    # Backward-compat: fau-bp4d/fau-disfa auto-pull base backbone first.
    if category in ("fau-bp4d", "fau-disfa") and model in WEIGHTS_DB["backbone"]:
        typer.secho(f"📦 Авто-загрузка базового backbone для {model}...", fg="blue")
        download_file(WEIGHTS_DB["backbone"][model],
                      TARGET_DIR["backbone"][0], is_gdrive=False)

    target = WEIGHTS_DB[category][model]

    if category == "backbone":
        download_file(target, output_dir, is_gdrive=False)

    elif category == "rppg-df":
        typer.secho(f"🫀 DeepFakesON-Phys ({model}) — Keras .h5", fg="yellow")
        download_file(target, output_dir, is_gdrive=False)
        h5_path = os.path.join(output_dir, os.path.basename(target).replace("%20", " "))
        if os.path.exists(h5_path):
            typer.secho("🔁 Конвертирую .h5 → .pth (h5py)...", fg="blue")
            pth_path = os.path.splitext(h5_path)[0] + ".pth"
            ret = subprocess.run([
                sys.executable, "scripts/convert_deepfakeson_phys.py",
                "--src", h5_path, "--dst", pth_path,
            ])
            if ret.returncode == 0:
                typer.secho(f"✅ Сохранено: {pth_path}", fg="green")
            else:
                typer.secho("⚠ Конвертация не удалась — см. вывод выше.", fg="red")

    elif category == "fau-df":
        typer.secho(f"🎭 OpenGraphAU ({model}) — Google Drive", fg="yellow")
        download_file(target, output_dir, is_gdrive=True)

    else:
        # fau-bp4d / fau-disfa (legacy path)
        typer.secho(f"🔥 Скачиваю FAU Model ({category}): {model}...", fg="yellow")
        out_path = download_file(target, output_dir, is_gdrive=True)
        if out_path and out_path.endswith('.zip'):
            with zipfile.ZipFile(out_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(out_path)
            typer.secho(f"✅ Фолды для {model} распакованы.", fg="green")


if __name__ == "__main__":
    app()
