"""
Experiment scripts for Figma2Code.

This module provides experiment runners for:
- exp1: Model comparison (testing different LLM models)
- exp2: Method comparison (testing different generation methods)
- exp3: Ablation study (testing importance of different input components)

Each experiment script handles both generation and evaluation.
"""

from .base import ExperimentBase, ExperimentConfig
from .exp1_model_comparison import Exp1ModelComparison
from .exp2_method_comparison import Exp2MethodComparison
from .exp3_ablation import Exp3Ablation

# Experiment registry
EXPERIMENTS = {
    "exp1": Exp1ModelComparison,
    "exp2": Exp2MethodComparison,
    "exp3": Exp3Ablation,
}


def get_experiment(name: str) -> type:
    """Get experiment class by name."""
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return EXPERIMENTS[name]


def list_experiments() -> list:
    """List available experiment names."""
    return list(EXPERIMENTS.keys())


__all__ = [
    "ExperimentBase",
    "ExperimentConfig",
    "Exp1ModelComparison",
    "Exp2MethodComparison",
    "Exp3Ablation",
    "EXPERIMENTS",
    "get_experiment",
    "list_experiments",
]
