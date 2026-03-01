"""
Code generation methods for Figma2Code.

This module provides various methods for generating HTML/CSS code from Figma designs:
- Design2Code baseline methods
- Direct generation (Figma JSON + optional image)
- Agent-based generation with critic-refiner pipeline
- Ablation processing utilities
"""

from .base import BaseGeneration, GenerationError, InputValidationError, OutputParsingError
from .design2code import (
    ImageDirectGeneration,
    TextAugmentedGeneration,
    direct_prompting,
    text_augmented_prompting,
)
from .direct import (
    DirectGeneration,
    direct_generation,
    INPUT_FIGMA,
    INPUT_FIGMA_IMAGE,
    INPUT_FIGMA_URL,
)
from .agent import AgentPipeline, agent_generation

from .ablation import (
    ablate_geometry,
    ablate_style,
    ablate_image_refs,
    ablate_structure,
    ablate_text,
    ABLATION_FUNCTIONS,
    get_ablation_function,
    list_ablation_types,
)

__all__ = [
    # Base
    "BaseGeneration",
    "GenerationError",
    "InputValidationError",
    "OutputParsingError",
    # Design2Code methods
    "ImageDirectGeneration",
    "TextAugmentedGeneration",
    "direct_prompting",
    "text_augmented_prompting",
    # Direct generation
    "DirectGeneration",
    "direct_generation",
    "INPUT_FIGMA",
    "INPUT_FIGMA_IMAGE",
    "INPUT_FIGMA_URL",
    # Agent-based generation
    "AgentPipeline",
    "agent_generation",
    # Ablation
    "ablate_geometry",
    "ablate_style",
    "ablate_image_refs",
    "ablate_structure",
    "ablate_text",
    "ABLATION_FUNCTIONS",
    "get_ablation_function",
    "list_ablation_types",
]
