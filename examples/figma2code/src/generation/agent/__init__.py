"""
Agent-based code generation pipeline.

This module implements a critic-refiner agent pipeline for iterative code improvement:
- FigmaToIR: Converts Figma JSON to intermediate representation
- IRToTailwind: Converts IR to Tailwind HTML
- Critic: Analyzes generated HTML against reference design
- Refiner: Improves HTML based on critique feedback
- Pipeline: Orchestrates the generation and refinement process
"""

from .figma_to_ir import FigmatoIR
from .ir_to_tailwind import IRtoTailwind
from .critic import run_critic, METRIC_DEFINITIONS
from .refiner import run_refiner
from .pipeline import AgentPipeline, agent_generation

__all__ = [
    "FigmatoIR",
    "IRtoTailwind",
    "run_critic",
    "METRIC_DEFINITIONS",
    "run_refiner",
    "AgentPipeline",
    "agent_generation",
]
