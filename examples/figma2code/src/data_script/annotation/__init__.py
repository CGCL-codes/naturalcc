"""
Data annotation utilities for Figma2Code.

Provides tools for:
- Annotate images with LLM
- Split dataset into multiple splits
- Filter and annotate images with Gradio
- Annotate and generate whitelist
- Process whitelist CSV and analyze content distribution
- Process complexity reports and compute complexity
"""

from .llm_images_annotator import annotate_single_image, annotate_images
from .dataset_split_package import run_split_package
from .gradio_filter_and_annotation import main as run_gradio_filter_and_annotation
from .annotate_and_generate_whitelist import run_annoate_and_generate_whitelist
from .content_category_mapper import process_whitelist_csv, analyze_content_distribution
from .complexity_compute import process_reports as process_complexity_reports

__all__ = [
    "annotate_single_image",
    "annotate_images",
    "run_split_package",
    "run_gradio_filter_and_annotation",
    "run_annoate_and_generate_whitelist",
    "process_whitelist_csv",
    "analyze_content_distribution",
    "process_complexity_reports",
]
