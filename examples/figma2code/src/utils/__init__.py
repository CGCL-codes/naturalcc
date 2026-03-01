"""
Common utilities for Figma2Code.

- logging: Unified logging setup
- html_screenshot: HTML to screenshot conversion
- image: Image processing utilities
- parsing: Output parsing utilities
- experiment: Experiment output management
"""

from .console_logger import console, create_progress, setup_logging, logger
from .files import load_json, save_json
from .figma_utils import FigmaSession, safe_filename, find_imageref_in_json, find_component_instances, compare_images, get_node_statics, infer_image_ext
from .html_screenshot import html2shot, batch_html_to_screenshots
from .image import get_root_url, compress_png_to_target, load_image
from .parsing import parse_output, remove_empty, extract_html

__all__ = [
    # Logging
    "console",
    "create_progress",
    "setup_logging",
    "logger",
    # Files
    "load_json",
    "save_json",
    # figma utils
    "FigmaSession",
    "safe_filename",
    "find_imageref_in_json",
    "find_component_instances",
    "compare_images",
    "get_node_statics",
    "infer_image_ext",
    # HTML screenshot
    "html2shot",
    "batch_html_to_screenshots",
    # Image utilities
    "get_root_url",
    "compress_png_to_target",
    "load_image",
    # Parsing
    "parse_output",
    "remove_empty",
    "extract_html",
]
