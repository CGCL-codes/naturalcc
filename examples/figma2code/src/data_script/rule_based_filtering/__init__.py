"""
Data filtering utilities for Figma2Code.

Provides tools for:
- JSON-based page filtering (aspect ratio, children count, image coverage)
- CLIP similarity-based deduplication
- Score-based filtering (LLM filter.json, plot distributions)
"""

from .json_filter import FilterConfig, run_json_filter
from .similarity_filter import SimilarityConfig, run_similarity_filter
from .score_filter import ScoreFilterConfig, run_score_filter

__all__ = [
    "FilterConfig",
    "run_json_filter",
    "SimilarityConfig",
    "run_similarity_filter",
    "ScoreFilterConfig",
    "run_score_filter",
]
