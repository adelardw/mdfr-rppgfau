import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import torch
import torch.nn as nn
import numpy as np
import typer
import gc
import time
from PIL import Image
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from omegaconf import OmegaConf
from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer
from src.data.processor import FaceDetector, Processor
from src.data.transforms import VideoTransform

app = typer.Typer(pretty_exceptions_show_locals=False)

# =========================================================================
# 1. ТРАНСФОРМЕРЫ РАЗМЕРНОСТЕЙ
# =========================================================================

def swin_reshape_transform(tensor):
    """Для FAU (Swin): (B, Tokens, Dim) -> (B, Dim, H, W)"""
    if isinstance(tensor, tuple): tensor = tensor[0]
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    N, Num_Tokens, Dim = tensor.shape
    H = W = int(np.sqrt(Num_Tokens))
    if H * W != Num_Tokens:
        tensor = tensor[:, :H*W, :]
    result = tensor.transpose(1, 2).reshape(N, Dim, H, W)
    return result

def rppg_reshape_transform(tensor):
    """
    Для PhysNet:
    Вход: (Batch, Channels, Time, H, W) -> например (1, 64, 128, 8, 8)
    Выход: (Batch*Time, Channels, H, W) -> например (128, 64, 8, 8)
    """
    if isinstance(tensor, tuple): tensor = tensor[0]
    
    # PhysNet выдает 5D тензор
    if len(tensor.shape) == 5:
        B, C, T, H, W = tensor.shape
        # Перемещаем время к батчу: (B, T, C, H, W)
        result = tensor.permute(0, 2, 1, 3, 4)
        # Объединяем: (B*T, C, H, W)
        result = result.reshape(B*T, C, H, W)
        return result
        
    return tensor

# =========================================================================
# 2. ОБЕРТКА
# =========================================================================

class GradCAMModelWrapper(nn.Module):
    """Принимает [T, 3, H, W] → упаковывает в [1, 3, T, H, W] для модели."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x.unsqueeze(0).permute(0, 2, 1, 3, 4), return_info=False)

# =========================================================================
# 3. ВИЗУАЛИЗАТОР
# =========================================================================

class Visualizer:
    def draw_graph_strip(self, data, title, color, width, height):
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        if len(data) > 1:
            d_arr = np.array(data)
            mn, mx = d_arr.min(), d_arr.max()
            norm_data = (d_arr - mn) / (mx - mn + 1e-6)
        else:
            norm_data = data
        ax.plot(norm_data, color=color, linewidth=2)
        ax.set_facecolor('black')
        ax.axis('off')
        ax.set_xlim(0, len(data) if len(data) > 0 else 1)
        ax.set_ylim(-0.1, 1.1)
        ax.text(0.01, 0.8, title, transform=ax.transAxes, color='white', fontsize=12, weight='bold')
        fig.patch.set_facecolor('black')
        plt.tight_layout(pad=0)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())
        img = img.astype(np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        return img

# =========================================================================
# 4. MAIN
# =========================================================================

@app.command()
def process(
    video_path: str = typer.Option(..., "-i", help="Input video"),
    output_path: str = typer.Option("viz_rppg.avi", "-o", help="Output video"),
    ckpt_path: str = typer.Option(..., "-c", help="Model checkpoint"),
    config_path: str = typer.Option("config.yaml", "-cfg", help="Config file"),
    device: str = typer.Option("cuda", help="Device (cuda / cpu / mps)"),
    use_rppg: bool = typer.Option(False, "--use-rppg", help="Visualize PhysNet (rPPG) attention"),
    target_stage: int = typer.Option(3, help="For FAU only: Swin stage (1-4)")
):
    torch.set_grad_enabled(True)

    # --- Load ---
    print(f"Loading checkpoint: {ckpt_path}")
    file_config = OmegaConf.load(config_path)
    defaults = {
        'backbone_fau': 'swin_transformer_tiny',
        'num_frames': 128,
        'num_classes': 2,
        'dropout': 0.1,
        'num_au_classes': 12,
        'lora_cfg': None,
        'num_gender_classes': 0,
        'num_ethnicity_classes': 0,
        'num_emotion_classes': 0,
    }
    model_params = {**defaults, **OmegaConf.to_container(file_config.model_params, resolve=True)}

    lit_model = FauRPPGDeepFakeRecognizer.load_from_checkpoint(ckpt_path, model_params=model_params, map_location=device)
    for param in lit_model.parameters():
        param.requires_grad = True
    lit_model.eval()
    lit_model.to(device)
    
    cam_model = GradCAMModelWrapper(lit_model.model)
    
    # --- Layer Selection ---
    if use_rppg:
        print(">>> MODE: Visualizing PhysNet (rPPG)")
        # В PhysNet upsample2 - это последний слой перед пулингом.
        # Он возвращает размерность времени T (в отличие от других слоев), 
        # поэтому идеально подходит для покадровой визуализации.
        try:
            target_layer = lit_model.model.phys_encoder.upsample2
            print("Targeting: phys_encoder.upsample2")
        except AttributeError:
            print("Warning: 'upsample2' not found, using last child.")
            target_layer = list(lit_model.model.phys_encoder.children())[-1]
            
        reshape_func = rppg_reshape_transform
    else:
        print(">>> MODE: Visualizing FAU (Swin)")
        try:
            layer_idx = target_stage - 1 
            target_layer = lit_model.model.au_encoder.backbone.layers[layer_idx]
            if hasattr(target_layer, 'blocks'):
                target_layer = target_layer.blocks[-1]
                if hasattr(target_layer, 'norm1'):
                    target_layer = target_layer.norm1
        except:
            target_layer = list(lit_model.model.au_encoder.backbone.children())[-1]
        reshape_func = swin_reshape_transform

    cam = LayerCAM(model=cam_model, target_layers=[target_layer], reshape_transform=reshape_func)

    # --- Processing ---
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video: {total_frames} frames @ {orig_fps:.1f} FPS")

    processor = Processor(
        transform=VideoTransform(size=(224, 224), training=False),
        detector=FaceDetector(margin=20, device="cpu"),
    )

    viz = Visualizer()
    BATCH_SIZE = 128

    # autocast только если устройство поддерживает
    _autocast_device = device if device in ("cuda", "cpu") else "cpu"

    frames_buffer = []   # List[PIL.Image]
    raw_buffer    = []
    rppg_buffer   = []
    writer        = None
    chunks_done   = 0
    t_start       = time.time()

    def process_chunk(frames_pil, raw_frames, rppg_hist, vid_writer, n_real_frames):
        """
        frames_pil может быть длиннее n_real_frames (padding хвоста).
        CAM и rPPG считаются на полном буфере, рендерятся только n_real_frames кадров.
        """
        video_tensor = processor(frames_pil).to(device)   # [3, T, H, W]
        input_tensor = video_tensor.permute(1, 0, 2, 3)   # [T, 3, H, W]
        input_tensor = input_tensor.requires_grad_(True)

        # Inference (без градиентов)
        with torch.no_grad():
            video_input = input_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
            info_out = lit_model.model(video_input, return_info=True)

            rppg_raw = info_out["rPPG"][0].cpu().numpy()
            # Обрезаем/паддим rPPG до длины чанка
            T = len(frames_pil)
            if len(rppg_raw) >= T:
                rppg_sig = rppg_raw[:T]
            else:
                pad = np.full(T - len(rppg_raw), rppg_raw[-1] if len(rppg_raw) else 0.0)
                rppg_sig = np.concatenate([rppg_raw, pad])

            probs = torch.softmax(info_out["logits"], dim=1)
            score_real = probs[0, 1].item()
            is_real = score_real > 0.5
            label_text = f"REAL: {score_real:.2f}" if is_real else f"FAKE: {probs[0, 0].item():.2f}"
            color_text = (0, 255, 0) if is_real else (0, 0, 255)
            mode_text = "MODE: rPPG (PhysNet)" if use_rppg else "MODE: FAU (Swin)"

        # CAM
        targets = [ClassifierOutputTarget(1 if is_real else 0)]
        with torch.amp.autocast(_autocast_device):
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        del input_tensor, video_input
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Render — только n_real_frames кадров (паддинг не пишем)
        frame_h, frame_w = raw_frames[0].shape[:2]
        if vid_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vid_writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (frame_w, frame_h + 150))

        for i in range(n_real_frames):
            orig = cv2.resize(raw_frames[i], (frame_w, frame_h))
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if i < len(grayscale_cam):
                mask = grayscale_cam[i]
                if np.max(mask) > 0:
                    mask = mask / np.max(mask)
                mask = np.power(mask, 0.7)
            else:
                mask = np.zeros((frame_h, frame_w), dtype=np.float32)

            mask_resized = cv2.resize(mask, (frame_w, frame_h))
            vis = show_cam_on_image(orig_rgb, mask_resized, use_rgb=True)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            rppg_hist.append(float(rppg_sig[i]))
            if len(rppg_hist) > 100:
                rppg_hist.pop(0)
            graph_img = viz.draw_graph_strip(rppg_hist, "rPPG Pulse", "#00ff00", frame_w, 150)

            cv2.rectangle(vis, (0, 0), (300, 80), (0, 0, 0), -1)
            cv2.putText(vis, label_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_text, 2)
            cv2.putText(vis, mode_text,  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vid_writer.write(np.vstack([vis, graph_img]))

        return rppg_hist, vid_writer

    print("Starting...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        raw_buffer.append(frame)

        if len(frames_buffer) == BATCH_SIZE:
            rppg_buffer, writer = process_chunk(frames_buffer, raw_buffer, rppg_buffer, writer, BATCH_SIZE)
            frames_done = (chunks_done + 1) * BATCH_SIZE
            chunks_done += 1
            elapsed = time.time() - t_start
            fps_proc = frames_done / elapsed if elapsed > 0 else 0
            pct = f"{100 * frames_done / total_frames:.1f}%" if total_frames > 0 else "?"
            print(f"  chunk {chunks_done} | {frames_done} frames | {pct} | {fps_proc:.1f} fps")
            frames_buffer, raw_buffer = [], []

    if frames_buffer:
        n_real = len(frames_buffer)
        print(f"Tail chunk: {n_real} real frames (padding to {BATCH_SIZE})")
        needed = BATCH_SIZE - n_real
        frames_padded = frames_buffer + [frames_buffer[-1]] * needed
        rppg_buffer, writer = process_chunk(frames_padded, raw_buffer, rppg_buffer, writer, n_real)

    cap.release()
    if writer:
        writer.release()
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s! Saved to {output_path}")

if __name__ == "__main__":
    app()