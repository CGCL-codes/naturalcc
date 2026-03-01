"""
Human evaluation tools for Figma2Code.

Provides Gradio-based interfaces for human evaluation of:
- Generated code quality (HTML structure, class readability)
- Visual similarity (image comparisons)

Also includes utilities for:
- Results analysis pipeline
"""

from .eval_script.code_evaluator import CodeEvaluator, create_code_evaluation_interface, launch_code_evaluator
from .eval_script.image_evaluator import ImageEvaluator, create_image_evaluation_interface, launch_image_evaluator
from .results_analysis.pipeline import AnalysisConfig, run_all_analyses

__all__ = [
    # Code evaluation
    "CodeEvaluator",
    "create_code_evaluation_interface",
    "launch_code_evaluator",
    # Image evaluation
    "ImageEvaluator",
    "create_image_evaluation_interface",
    "launch_image_evaluator",
    # Analysis
    "AnalysisConfig",
    "run_all_analyses",
]
