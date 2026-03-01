"""
Main evaluator module for Figma2Code evaluation.

Provides unified interface for evaluating generated HTML against reference designs.
Combines visual similarity metrics and code quality metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image
import pandas as pd
import torch
import tqdm

from . import visual
from . import quality
from ..utils.html_screenshot import html2shot

from ..utils.console_logger import logger


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    device: str = "auto"
    lpips_size: int = 256
    lpips_net: str = "vgg"
    bands: Tuple[int, int, int] = (640, 1024, 1440)
    dmax: int = 10
    cmax: int = 6
    save_rendered_image: bool = False


def get_device(device: str = "auto") -> str:
    """Resolve device string to actual device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def render_html_to_image(
    html_path: Union[str, Path],
    width: int,
    height: int,
    use_viewport: bool = True,
    save_rendered_image: bool = False
) -> Image.Image:
    """
    Render HTML file to image using Playwright.
    
    Args:
        html_path: Path to HTML file
        width: Viewport width
        height: Viewport height
        use_viewport: Whether to use viewport settings
        save_rendered_image: Whether to save the rendered image
    
    Returns:
        PIL Image in RGB mode
    """
    html_path = Path(html_path)
    output_file = html_path.with_suffix(".png") if save_rendered_image else None
    
    img = html2shot(
        html_file_path=html_path,
        output_file=output_file,
        use_viewport=use_viewport,
        viewport={'width': width, 'height': height}
    )
    if not isinstance(img, Image.Image):
        raise TypeError("html2shot must return a PIL.Image.Image")
    return img.convert("RGB")


def compute_visual_metrics(
    ref_img: Image.Image,
    pred_img: Image.Image,
    device: str = "auto",
    lpips_size: int = 256,
    lpips_net: str = "vgg"
) -> Dict[str, Any]:
    """
    Compute visual similarity metrics between reference and predicted images.
    
    Args:
        ref_img: Reference PIL Image
        pred_img: Predicted/generated PIL Image
        device: Computation device
        lpips_size: Size for LPIPS computation
        lpips_net: Network for LPIPS ("vgg" or "alex")
    
    Returns:
        Dictionary of metric names to values
    """
    device = get_device(device)
    
    # Align predicted image to reference size
    pred_img_aligned = visual.resize_to_ref(pred_img, ref_img.size)
    
    ref_np = np.array(ref_img.convert("RGB"))
    pred_np = np.array(pred_img.convert("RGB"))
    pred_np_aligned = np.array(pred_img_aligned.convert("RGB"))
    
    metrics = {}
    
    # Pixel-wise metrics (require alignment)
    metrics["PSNR"] = visual.psnr(ref_np, pred_np_aligned)
    metrics["SSIM"] = visual.ssim(ref_np, pred_np_aligned)
    metrics["LPIPS"] = visual.compute_lpips(ref_np, pred_np_aligned, size=lpips_size, device=device, net=lpips_net)
    metrics["MAE_px"] = visual.mae(ref_np, pred_np_aligned)
    metrics["MSE_px"] = visual.mse(ref_np, pred_np_aligned)
    
    # Representation metrics (no alignment needed)
    try:
        metrics["CLIP"] = visual.clip_sim(ref_img, pred_img, device=device)
    except Exception as e:
        logger.warning(f"CLIP computation failed: {e}")
        metrics["CLIP"] = np.nan
    
    try:
        metrics["MAE_embed"] = visual.vit_mae(ref_np, pred_np, device=device)
    except Exception as e:
        logger.warning(f"ViT-MAE computation failed: {e}")
        metrics["MAE_embed"] = np.nan
    
    try:
        metrics["DINOv2"] = visual.dinov2(ref_np, pred_np, device=device)
    except Exception as e:
        logger.warning(f"DINOv2 computation failed: {e}")
        metrics["DINOv2"] = np.nan
    
    return metrics


def flatten_quality_metrics(qres: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten quality analysis results with prefixed keys.
    
    Args:
        qres: Quality analysis results from quality.analyze()
    
    Returns:
        Flattened dictionary with RESP_ and MAINT_ prefixed keys
    """
    flat: Dict[str, Any] = {}
    if not isinstance(qres, dict):
        return flat
    
    resp = qres.get("responsiveness")
    if isinstance(resp, dict):
        for k, v in resp.items():
            flat[f"RESP_{k}"] = v
    
    maint = qres.get("maintainability")
    if isinstance(maint, dict):
        for k, v in maint.items():
            flat[f"MAINT_{k}"] = v
    
    # Legacy fallback: copy numeric/bool keys with heuristic prefix
    if not flat:
        for k, v in qres.items():
            if isinstance(v, (int, float, bool)):
                name = k.lower()
                if any(x in name for x in ["unit", "breakpoint", "flex", "grid", "absolute", "fixed", "viewport", "responsive", "container"]):
                    flat[f"RESP_{k}"] = v
                elif any(x in name for x in ["semantic", "dom", "inline", "duplicate", "class", "selector", "arbitrary"]):
                    flat[f"MAINT_{k}"] = v
    
    return flat


def compute_html_quality(
    html_path: Union[str, Path],
    assets_dir: Optional[Path] = None,
    bands: Tuple[int, int, int] = (640, 1024, 1440),
    dmax: int = 10,
    cmax: int = 6
) -> Dict[str, Any]:
    """
    Compute HTML code quality metrics.
    
    Args:
        html_path: Path to HTML file
        assets_dir: Optional assets directory
        bands: Responsive breakpoint values
        dmax: Max DOM depth for normalization
        cmax: Max selector complexity for normalization
    
    Returns:
        Flattened quality metrics dictionary
    """
    res = quality.analyze(
        Path(html_path),
        assets_dir=assets_dir,
        bands=bands,
        dmax=dmax,
        cmax=cmax
    )
    if not isinstance(res, dict):
        return {}
    return flatten_quality_metrics(res)


def iter_html_files(folder: Path) -> List[Path]:
    """List all HTML files in a folder, sorted."""
    return sorted([p for p in folder.glob("*.html") if p.is_file()])


def load_reference_image(
    sample_dir: Path,
    crop_whitespace: bool = True
) -> Tuple[Optional[Image.Image], int, int]:
    """
    Load reference image from sample directory.
    
    Args:
        sample_dir: Sample directory containing root.png and metadata
        crop_whitespace: Whether to crop white borders
    
    Returns:
        Tuple of (image, width, height) or (None, 0, 0) if not found
    """
    ref_path = sample_dir / "root.png"
    if not ref_path.exists():
        return None, 0, 0
    
    ref_img = Image.open(ref_path).convert("RGB")
    
    # Load metadata for dimensions
    json_path = sample_dir / "processed_metadata.json"
    width = height = 0
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                meta_data = json.load(f)
            bbox = meta_data.get('document', {}).get('absoluteBoundingBox', {})
            width = int(bbox.get('width', 0))
            height = int(bbox.get('height', 0))
            if width > 0 and height > 0:
                ref_img = ref_img.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {json_path}: {e}")
    
    if crop_whitespace:
        ref_img = visual.crop_nonwhite_border_pil(ref_img)
    
    return ref_img, width, height


def evaluate_single(
    html_path: Path,
    ref_img: Image.Image,
    width: int,
    height: int,
    config: Optional[EvaluationConfig] = None
) -> Dict[str, Any]:
    """
    Evaluate a single HTML file against a reference image.
    
    Args:
        html_path: Path to HTML file
        ref_img: Reference PIL Image
        width: Viewport width
        height: Viewport height
        config: Evaluation configuration
    
    Returns:
        Dictionary of all metrics
    """
    config = config or EvaluationConfig()
    device = get_device(config.device)
    
    # Render HTML to image
    pred_img = render_html_to_image(html_path, width, height, save_rendered_image=config.save_rendered_image)
    pred_img = visual.crop_nonwhite_border_pil(pred_img)
    
    # Compute visual metrics
    vis_metrics = compute_visual_metrics(
        ref_img, pred_img,
        device=device,
        lpips_size=config.lpips_size,
        lpips_net=config.lpips_net
    )
    
    # Compute quality metrics
    q_metrics = compute_html_quality(
        str(html_path),
        bands=config.bands,
        dmax=config.dmax,
        cmax=config.cmax
    )
    
    result = {}
    result.update(vis_metrics)
    result.update(q_metrics)
    return result


def evaluate_folder(
    root: Path,
    config: Optional[EvaluationConfig] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Evaluate all HTML files in a folder structure.
    
    Expects folder structure:
        root/
            sample1/
                root.png
                processed_metadata.json
                output1.html
                output2.html
            sample2/
                ...
    
    Args:
        root: Root directory containing sample folders
        config: Evaluation configuration
        show_progress: Whether to show progress bar
    
    Returns:
        DataFrame with evaluation results
    """
    config = config or EvaluationConfig()
    rows = []
    
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if show_progress:
        subdirs = tqdm.tqdm(subdirs, desc="Evaluating samples")
    
    for sub in subdirs:
        ref_img, width, height = load_reference_image(sub)
        if ref_img is None:
            continue
        
        for html in iter_html_files(sub):
            if show_progress:
                tqdm.tqdm.write(f"Evaluating {html} ...")
            
            try:
                metrics = evaluate_single(html, ref_img, width, height, config)
                row = {"folder": sub.name, "result_name": html.stem}
                row.update(metrics)
                rows.append(row)
            except Exception as e:
                logger.error(f"Failed to evaluate {html}: {e}")
                continue
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Order columns
    key_cols = ["folder", "result_name"]
    vis_cols = ["PSNR", "SSIM", "LPIPS", "MAE_px", "MSE_px", "CLIP", "MAE_embed", "DINOv2"]
    resp_cols = sorted([c for c in df.columns if c.startswith("RESP_")])
    maint_cols = sorted([c for c in df.columns if c.startswith("MAINT_")])
    ordered_cols = key_cols + [c for c in vis_cols if c in df.columns] + resp_cols + maint_cols
    
    return df[ordered_cols]


def save_results(
    df: pd.DataFrame,
    output_path: Path,
    format: str = "csv"
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        df: DataFrame with results
        output_path: Output file path
        format: Output format ("csv" or "json")
    """
    if df.empty:
        logger.warning("No results to save")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Results saved to {output_path}")


def load_existing_results(output_path: Path) -> Optional[pd.DataFrame]:
    """
    Load existing evaluation results from file.
    
    Args:
        output_path: Path to the results file
    
    Returns:
        DataFrame with existing results, or None if file doesn't exist
    """
    if not output_path.exists():
        return None
    
    try:
        df = pd.read_csv(output_path, encoding="utf-8-sig")
        logger.info(f"Loaded existing results from {output_path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"Failed to load existing results from {output_path}: {e}")
        return None


def get_evaluated_keys(
    existing_df: Optional[pd.DataFrame],
    sample_key_col: str = "sample_key",
    result_name_col: str = "result_name",
    only_successful: bool = True
) -> set:
    """
    Get set of (sample_key, result_name) tuples that have been evaluated.
    
    Args:
        existing_df: DataFrame with existing results
        sample_key_col: Column name for sample key
        result_name_col: Column name for result name
        only_successful: If True, only return keys for successful evaluations
                        (where failure_reason is empty and generation_failed is False).
                        This allows re-evaluation of previously failed samples.
    
    Returns:
        Set of (sample_key, result_name) tuples
    """
    if existing_df is None or existing_df.empty:
        return set()
    
    if sample_key_col not in existing_df.columns or result_name_col not in existing_df.columns:
        return set()
    
    df_filtered = existing_df
    
    if only_successful:
        # Filter to only successful evaluations
        # Check generation_failed column
        if "generation_failed" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["generation_failed"] == False]
    
    return set(zip(df_filtered[sample_key_col], df_filtered[result_name_col]))


def is_already_evaluated(
    sample_key: str,
    result_name: str,
    evaluated_keys: set
) -> bool:
    """
    Check if a (sample_key, result_name) pair has already been evaluated.
    
    Args:
        sample_key: Sample identifier
        result_name: Result/method name
        evaluated_keys: Set of already evaluated (sample_key, result_name) tuples
    
    Returns:
        True if already evaluated, False otherwise
    """
    return (sample_key, result_name) in evaluated_keys


def merge_results(
    existing_df: Optional[pd.DataFrame],
    new_rows: List[Dict[str, Any]],
    sample_key_col: str = "sample_key",
    result_name_col: str = "result_name"
) -> pd.DataFrame:
    """
    Merge new evaluation results with existing results.
    
    New results will replace existing ones for the same (sample_key, result_name) pair.
    
    Args:
        existing_df: DataFrame with existing results
        new_rows: List of new result dictionaries
        sample_key_col: Column name for sample key
        result_name_col: Column name for result name
    
    Returns:
        Merged DataFrame
    """
    if not new_rows:
        return existing_df if existing_df is not None else pd.DataFrame()
    
    new_df = pd.DataFrame(new_rows)
    
    if existing_df is None or existing_df.empty:
        return new_df
    
    # Create composite key for deduplication
    new_keys = set(zip(new_df[sample_key_col], new_df[result_name_col]))
    
    # Filter out existing rows that will be replaced
    mask = ~existing_df.apply(
        lambda row: (row[sample_key_col], row[result_name_col]) in new_keys,
        axis=1
    )
    filtered_existing = existing_df[mask]
    
    # Concatenate and return
    merged = pd.concat([filtered_existing, new_df], ignore_index=True)
    
    # Sort by sample_key and result_name for consistency
    merged = merged.sort_values([sample_key_col, result_name_col]).reset_index(drop=True)
    
    return merged
