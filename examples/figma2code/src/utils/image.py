"""
Image processing utilities for Figma2Code.

Provides functions for image URL generation, compression, and base64 encoding.
"""

import io
import base64
from pathlib import Path
from typing import Tuple, Optional, Union

from PIL import Image, ImageOps

from ..configs.settings import get_settings
from .console_logger import logger


def get_root_url(subdir_name: str, get_small: bool = False) -> str:
    """
    Get the URL for a reference image hosted on the image server.
    
    Args:
        subdir_name: Sample subdirectory name (used as filename without extension)
        get_small: If True, use compressed image version (for Claude API < 5MB requirement)
    
    Returns:
        Full URL to the hosted image
    
    Note:
        Only images from the test set are available on the host.
        Default version provides files < 10MB (GPT-5 requirement).
        Small version provides files with base64 < 5MB (Claude 4.1 requirement).
    """
    settings = get_settings()
    
    if get_small:
        prefix = settings.image_host_prefix_small
    else:
        prefix = settings.image_host_prefix
    
    return f"{prefix}{subdir_name}.png"


def encode_image_to_url(
    image: Image.Image,
    prefer: str = "PNG",               # "PNG" | "WEBP" | "JPEG"
    max_width: Optional[int] = 1600,   # Limit max width to avoid unnecessary token overhead
    png_compress_level: int = 6,       # 0-9, higher means smaller but slower
    jpeg_quality: int = 90,            # 85-92 is a good balance between quality and size
    webp_quality: int = 90,            # 0-100, 100 is best quality
    webp_lossless: bool = False,       # True for lossless WebP
    background: Tuple[int, int, int] = (255, 255, 255), # Background for images with alpha
    size_limit_bytes: Optional[int] = None, # Fall back to smaller format if size is exceeded
    fallback_format: str = "WEBP",     # Fallback format if size limit exceeded ("WEBP" or "JPEG")
) -> str:
    """
    Encode a PIL Image to a base64 string.

    Steps:
    1) EXIF orientation correction
    2) Resize to max_width (maintain ratio)
    3) Convert to preferred format with compression/quality
    4) Fallback to a different format if size limit is set and exceeded

    Args:
        image: PIL Image to encode
        prefer: Target format ("PNG", "WEBP", "JPEG")
        max_width: Maximum allowed width for the image
        png_compress_level: PNG compression level (0-9)
        jpeg_quality: JPEG quality (0-100)
        webp_quality: WebP lossy quality (0-100)
        webp_lossless: Use WebP lossless mode
        background: RGB color for alpha compositing
        size_limit_bytes: Limit for output size; fallback if exceeded
        fallback_format: Fallback format ("WEBP" or "JPEG")

    Returns:
        Base64 encoded string of the image
    """
    # 1) EXIF orientation correction
    img = ImageOps.exif_transpose(image)

    # 2) Resize if image is wider than max_width
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)

    def _encode(image: Image.Image, fmt: str) -> Tuple[bytes, str]:
        fmt = fmt.upper()
        buf = io.BytesIO()
        if fmt == "PNG":
            # Preserve transparency if present
            image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")
            image.save(buf, format="PNG", optimize=True, compress_level=png_compress_level)
            return buf.getvalue(), "image/png"
        elif fmt in ("JPG", "JPEG"):
            # JPEG does not support transparency; flatten onto background if needed
            if "A" in image.getbands():
                bg = Image.new("RGB", image.size, background)
                bg.paste(image, mask=image.split()[-1])
                image = bg
            else:
                image = image.convert("RGB")
            image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            return buf.getvalue(), "image/jpeg"
        elif fmt == "WEBP":
            # Use WebP lossy or lossless
            image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")
            if webp_lossless:
                image.save(buf, format="WEBP", lossless=True, method=6)
            else:
                image.save(buf, format="WEBP", quality=webp_quality, method=6)
            return buf.getvalue(), "image/webp"
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # 3) Encode using preferred format
    data, mime = _encode(img, prefer)

    # 4) Fallback to alternate format if size exceeds limit
    if size_limit_bytes and len(data) > size_limit_bytes and prefer.upper() != fallback_format.upper():
        data, mime = _encode(img, fallback_format)

    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    return data_url


def load_image(path_or_img: Union[Path, str, Image.Image]) -> Optional[Image.Image]:
    """
    Load an image from path or return existing Image.
    
    Args:
        path_or_img: Path to image file or PIL Image
    
    Returns:
        PIL Image or None if loading failed
    """
    if isinstance(path_or_img, Image.Image):
        return path_or_img
    
    path = Path(path_or_img)
    if path.exists():
        try:
            return Image.open(path).convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    return None


def save_png(img: Image.Image, output_path: Union[str, Path]) -> None:
    """
    Save a PIL Image as an optimized PNG file.
    
    Args:
        img: PIL Image to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG", optimize=True, compress_level=9)
    logger.debug(f"PNG saved to: {output_path}")


def base64_size_mb(img_bytes: bytes) -> float:
    """
    Calculate the size of base64-encoded bytes in megabytes.
    
    Args:
        img_bytes: Raw image bytes
    
    Returns:
        Size in MB after base64 encoding
    """
    return len(base64.b64encode(img_bytes)) / (1024 * 1024)


def compress_png_to_target(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_base64_mb: float = 5.0,
    min_dimension: int = 512,
    preserve_transparency: bool = True,
    ratio: float = 0.9,
    max_iters: int = 50
) -> Tuple[bool, str, float]:
    """
    Compress a PNG image to fit within a target base64 size.
    
    Iteratively scales down the image until the base64-encoded size
    is below the target threshold.
    
    Args:
        input_path: Input PNG file path
        output_path: Output PNG file path
        target_base64_mb: Target size in MB for base64-encoded output
        min_dimension: Minimum allowed dimension (width or height)
        preserve_transparency: Whether to preserve alpha channel
        ratio: Scale factor per iteration (e.g., 0.9 = 90% of previous size)
        max_iters: Maximum number of iterations
    
    Returns:
        Tuple of (success, message, final_size_mb)
    """
    try:
        img = Image.open(input_path)
        original_width, original_height = img.size
        logger.debug(f"Original size: {original_width}x{original_height}")

        # Handle transparency
        if preserve_transparency:
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode == "LA":
                img = img.convert("RGBA")
        else:
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                else:
                    img = img.convert("RGB")

        # Check if original fits
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True, compress_level=9)
        size_mb = base64_size_mb(buf.getvalue())
        
        if size_mb <= target_base64_mb:
            save_png(img, output_path)
            return True, f"Original size {original_width}x{original_height}", size_mb

        # Iteratively scale down
        cur_img = img
        cur_w, cur_h = original_width, original_height
        
        for i in range(max_iters):
            new_w = max(int(cur_w * ratio), min_dimension)
            new_h = max(int(cur_h * ratio), min_dimension)
            
            if new_w == cur_w or new_h == cur_h:
                break  # Can't scale further

            cur_img = cur_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            buf = io.BytesIO()
            cur_img.save(buf, format="PNG", optimize=True, compress_level=9)
            size_mb = base64_size_mb(buf.getvalue())

            logger.debug(f"Iteration {i+1}: {new_w}x{new_h}, {size_mb:.2f}MB")

            if size_mb <= target_base64_mb:
                save_png(cur_img, output_path)
                return True, f"Compressed to {new_w}x{new_h}", size_mb

            cur_w, cur_h = new_w, new_h

        # Save final version even if still over target
        save_png(cur_img, output_path)
        return False, f"Compressed to {cur_w}x{cur_h}, still over target", size_mb

    except Exception as e:
        logger.exception(f"Compression error: {e}")
        return False, f"Error: {str(e)}", 0.0