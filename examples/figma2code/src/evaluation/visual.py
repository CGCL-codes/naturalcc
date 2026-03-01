"""
Visual similarity metrics for Figma2Code evaluation.

A set of image similarity assessment tools, uniformly taking **NumPy H×W×C (RGB)** as input; accepts PIL.Image when needed.

Main features:
1. **Pixel-wise metrics**: MSE / MAE / PSNR / SSIM / LPIPS — require consistent input dimensions.
   - Recommendation: First, remove white borders, then scale proportionally to the reference image size + padding for alignment.
2. **Representation metrics**: CLIP / ViT-MAE / DINOv2 — models' own processors handle normalization, no need for consistent input dimensions.
3. **Model caching**: Avoids repeated loading, improving performance.
4. **Annotations and sources**: Each algorithm includes its paper source and a description of its meaning.
5. **Assertions**: Assertions are added in functions that require the same input dimensions.

Dependencies: numpy, pillow, torch, torchvision, scikit-image, lpips, transformers, clip
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

try:
    import clip # CLIP: Radford et al., 2021
except ImportError:
    clip = None

try:
    from transformers import AutoImageProcessor, ViTMAEModel, Dinov2Model # MAE: He et al., 2022; DINOv2: Oquab et al., 2023
except ImportError:
    AutoImageProcessor = None
    ViTMAEModel = None
    Dinov2Model = None

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========== Utility Functions ==========

def crop_nonwhite_border_pil(img: Image.Image, white_thresh: int = 250) -> Image.Image:
    """
    Remove white borders from an image.
    
    Args:
        img: PIL Image to crop
        white_thresh: Pixel values >= this are considered white
    
    Returns:
        Cropped PIL Image (original if all white)
    """
    arr = np.array(img.convert("RGB"))
    mask = np.all(arr >= white_thresh, axis=2)
    coords = np.argwhere(~mask)
    if coords.size == 0:
        return img
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def resize_to_ref(
    img: Image.Image,
    ref_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Resize image proportionally and pad to reference size.
    - Maintains aspect ratio, the remaining area is filled with pad_color.
    - Used for alignment before pixel-wise metrics.
    
    Args:
        img: PIL Image to resize
        ref_size: Target size (width, height)
        pad_color: RGB color for padding
    
    Returns:
        Resized and padded PIL Image
    """
    ref_w, ref_h = ref_size
    w, h = img.size
    scale = min(ref_w / w, ref_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (ref_w, ref_h), pad_color)
    offset = ((ref_w - new_w) // 2, (ref_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def ensure_hwc_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HxWx3 RGB format."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Requires HxWx3 RGB, but got {arr.shape}")
    return arr


def numpy_to_lpips_tensor(arr: np.ndarray, size: int) -> torch.Tensor:
    """Convert numpy array (HWC) to LPIPS input: [-1,1] NCHW tensor."""
    arr = ensure_hwc_rgb(arr)
    pil = Image.fromarray(arr.astype(np.uint8))
    tfm = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    return tfm(pil).unsqueeze(0)


def normalize_float_0_255(arr: np.ndarray) -> np.ndarray:
    """Normalize array to float32 with range [0, 255]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32)
    a = arr.astype(np.float32)
    maxv = float(np.max(a)) if a.size else 1.0
    if maxv <= 1.0 + 1e-6:
        a *= 255.0
    return np.clip(a, 0.0, 255.0)


@torch.no_grad()
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return torch.sum(a_n * b_n, dim=-1).mean().item()


# ========== Model Cache ==========
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_VIT_MAE_MODEL = None
_VIT_MAE_PROC = None
_DINOV2_MODEL = None
_DINOV2_PROC = None
_LPIPS_MODELS = {}


# ========== Metric Functions ==========

def clip_sim(im1: Image.Image, im2: Image.Image, device: str = _DEVICE) -> float:
    """
    CLIP image cosine similarity (Radford et al., 2021).
    
    Uses ViT-B/32 model. Higher values indicate more similar images.
    
    Args:
        im1, im2: PIL Images to compare
        device: Computation device
    
    Returns:
        Cosine similarity score [0, 1]
    """
    if clip is None:
        raise ImportError("clip package required for clip_sim")
    
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device, jit=False)
        _CLIP_MODEL.eval()
    batch = torch.stack([_CLIP_PREPROCESS(im) for im in (im1, im2)]).to(device)
    feats = _CLIP_MODEL.encode_image(batch)
    return float(F.cosine_similarity(feats[0], feats[1], dim=0).item())


def psnr(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """
    PSNR: Peak Signal-to-Noise Ratio.
    
    Higher values indicate more similar images. Requires identical input shapes.
    
    Args:
        img1_np, img2_np: HWC numpy arrays
    
    Returns:
        PSNR value in dB
    """
    assert img1_np.shape == img2_np.shape, "PSNR requires identical dimensions"
    return float(peak_signal_noise_ratio(img1_np, img2_np, data_range=255))


@torch.no_grad()
def compute_lpips(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    size: int = 256,
    device: str = _DEVICE,
    net: str = "vgg"
) -> float:
    """
    LPIPS: Learned Perceptual Image Patch Similarity (Zhang et al., 2018).
    
    Lower values indicate more similar images.
    
    Args:
        img1_np, img2_np: HWC numpy arrays
        size: Resize dimension for LPIPS
        device: Computation device
        net: Network architecture ("vgg" or "alex")
    
    Returns:
        LPIPS distance [0, ~1]
    """
    if lpips_lib is None:
        raise ImportError("lpips package required for compute_lpips")
    
    global _LPIPS_MODELS
    if net not in _LPIPS_MODELS:
        _LPIPS_MODELS[net] = lpips_lib.LPIPS(net=net).to(device)
    loss_fn = _LPIPS_MODELS[net]
    t1 = numpy_to_lpips_tensor(img1_np, size=size)
    t2 = numpy_to_lpips_tensor(img2_np, size=size)
    d = loss_fn(t1.to(device), t2.to(device))
    return float(d.squeeze().item())


@torch.no_grad()
def vit_mae(img1_np: np.ndarray, img2_np: np.ndarray, device: str = _DEVICE) -> float:
    """
    ViT-MAE: Cosine similarity using MAE features (He et al., 2022).
    
    Higher values indicate more similar images.
    
    Args:
        img1_np, img2_np: HWC numpy arrays
        device: Computation device
    
    Returns:
        Cosine similarity [0, 1]
    """
    if ViTMAEModel is None:
        raise ImportError("transformers package required for vit_mae")
    
    global _VIT_MAE_MODEL, _VIT_MAE_PROC
    if _VIT_MAE_MODEL is None:
        _VIT_MAE_MODEL = ViTMAEModel.from_pretrained("facebook/vit-mae-base").to(device).eval()
        _VIT_MAE_PROC = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    i1 = normalize_float_0_255(ensure_hwc_rgb(img1_np)).astype(np.uint8)
    i2 = normalize_float_0_255(ensure_hwc_rgb(img2_np)).astype(np.uint8)
    inputs1 = _VIT_MAE_PROC(images=i1, return_tensors="pt").to(device)
    inputs2 = _VIT_MAE_PROC(images=i2, return_tensors="pt").to(device)
    f1 = _VIT_MAE_MODEL(**inputs1).last_hidden_state[:, 0, :]
    f2 = _VIT_MAE_MODEL(**inputs2).last_hidden_state[:, 0, :]
    return cosine_sim(f1, f2)


@torch.no_grad()
def dinov2(img1_np: np.ndarray, img2_np: np.ndarray, device: str = _DEVICE) -> float:
    """
    DINOv2: Cosine similarity using DINOv2 features (Oquab et al., 2023).
    
    Higher values indicate more similar images.
    
    Args:
        img1_np, img2_np: HWC numpy arrays
        device: Computation device
    
    Returns:
        Cosine similarity [0, 1]
    """
    if Dinov2Model is None:
        raise ImportError("transformers package required for dinov2")
    
    global _DINOV2_MODEL, _DINOV2_PROC
    if _DINOV2_MODEL is None:
        _DINOV2_MODEL = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
        _DINOV2_PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    i1 = normalize_float_0_255(ensure_hwc_rgb(img1_np)).astype(np.uint8)
    i2 = normalize_float_0_255(ensure_hwc_rgb(img2_np)).astype(np.uint8)
    inputs1 = _DINOV2_PROC(images=i1, return_tensors="pt").to(device)
    inputs2 = _DINOV2_PROC(images=i2, return_tensors="pt").to(device)
    f1 = _DINOV2_MODEL(**inputs1).last_hidden_state[:, 0, :]
    f2 = _DINOV2_MODEL(**inputs2).last_hidden_state[:, 0, :]
    return cosine_sim(f1, f2)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SSIM: Structural Similarity (Wang et al., 2004).
    
    Higher values indicate more similar images. Requires identical dimensions.
    
    Args:
        img1, img2: HWC numpy arrays
    
    Returns:
        SSIM score [0, 1]
    """
    assert img1.shape == img2.shape, "SSIM requires identical dimensions"
    return float(ssim_metric(img1, img2, data_range=255, channel_axis=-1))


def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    MAE: Mean Absolute Error (pixel-wise).
    
    Lower values indicate more similar images.
    
    Args:
        img1, img2: HWC numpy arrays
    
    Returns:
        MAE score [0, 1]
    """
    assert img1.shape == img2.shape, "MAE requires identical dimensions"
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)) / 255.0))


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    MSE: Mean Squared Error (pixel-wise).
    
    Lower values indicate more similar images.
    
    Args:
        img1, img2: HWC numpy arrays
    
    Returns:
        MSE score [0, 1]
    """
    assert img1.shape == img2.shape, "MSE requires identical dimensions"
    diff = (img1.astype(np.float32) - img2.astype(np.float32)) / 255.0
    return float(np.mean(diff ** 2))
