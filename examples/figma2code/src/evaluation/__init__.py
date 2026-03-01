"""
Evaluation metrics for Figma2Code.

This module provides comprehensive evaluation tools:
- Visual similarity metrics (PSNR, SSIM, LPIPS, CLIP, DINOv2, etc.)
- Code quality metrics (responsiveness, maintainability)
- LLM-based responsiveness evaluation
- Unified evaluation orchestrator

Visual Metrics:
- PSNR, SSIM, LPIPS (pixel-wise, require alignment)
- CLIP, ViT-MAE, DINOv2 (representation-based, no alignment needed)
- MSE, MAE (basic pixel errors)

Quality Metrics:
- Responsiveness: relative units, breakpoints, flex/grid usage, viewport meta
- Maintainability: semantic tags, DOM depth, inline styles, selector complexity

LLM-based Evaluation:
- ResponsivenessEvaluator: Multi-viewport rendering + LLM assessment
- Scores HTML pages on responsive design quality (1-5 scale)
"""

from .visual import (
    psnr,
    ssim,
    compute_lpips,
    clip_sim,
    vit_mae,
    dinov2,
    mae,
    mse,
    crop_nonwhite_border_pil,
    resize_to_ref,
)
from .quality import analyze as analyze_quality
from .evaluator import (
    EvaluationConfig,
    render_html_to_image,
    compute_visual_metrics,
    compute_html_quality,
    load_reference_image,
    evaluate_single,
    evaluate_folder,
    save_results,
)
from .responsiveness_llm import (
    evaluate_folder as evaluate_responsiveness_folder,
    run_evaluation as run_responsiveness_evaluation,
)

__all__ = [
    # Visual utilities
    "crop_nonwhite_border_pil",
    "resize_to_ref",
    # Visual metrics
    "psnr",
    "ssim",
    "compute_lpips",
    "clip_sim",
    "vit_mae",
    "dinov2",
    "mae",
    "mse",
    # Quality metrics
    "analyze_quality",
    # Evaluator
    "EvaluationConfig",
    "render_html_to_image",
    "compute_visual_metrics",
    "compute_html_quality",
    "load_reference_image",
    "evaluate_single",
    "evaluate_folder",
    "save_results",
    # LLM-based responsiveness evaluation
    "evaluate_responsiveness_folder",
    "run_responsiveness_evaluation",
]
