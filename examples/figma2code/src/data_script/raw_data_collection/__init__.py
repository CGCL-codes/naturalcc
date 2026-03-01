"""
Raw data collection module for Figma2Code.
Provides tools for:
- Crawling Figma templates (filekeys) from multiple community links
- Splitting file keys into multiple files
- Downloading Figma pages and rendering images for candidate selection
"""
from .filekey_crawl import get_figma_templates_from_multiple_urls as filekey_crawl
from .filekeysplit import filekeysplit
from .figma_page_filter import main as figma_page_filter

__all__ = [
    "filekey_crawl",
    "filekeysplit",
    "figma_page_filter",
]