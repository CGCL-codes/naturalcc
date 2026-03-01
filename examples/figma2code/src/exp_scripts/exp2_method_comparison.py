"""
Experiment 2: Method Comparison

Compares different generation methods using the same model.
Tests the effectiveness of various input types and generation approaches.

Methods tested:
- image_direct: Image-only direct generation (Design2Code baseline)
- image_text_augmented: Image + extracted text (Design2Code baseline)
- figma_direct: Figma JSON only (metadata-based)
- figma_image_direct: Figma JSON + image
- figma_image_agent: Agent-based with critic-refiner loop

When add_guidelines is True, only figma_direct and figma_image_direct methods are tested.

Output filename format: {save_prefix}{method}__{model}.html
Example: figma_direct__ernie4_5_vl_424b_a47b.html

Note: exp1 and exp2 share the same filename format, so if exp1 generates 
figma_image_direct__gpt4o.html, exp2 will skip regenerating it.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .base import ExperimentBase, ExperimentConfig
from ..configs import MODELS
from ..llm import create_llm, OpenRouterLLM
from ..generation import (
    direct_generation,
    direct_prompting,
    text_augmented_prompting,
    agent_generation,
    INPUT_FIGMA,
    INPUT_FIGMA_URL,
    INPUT_FIGMA_IMAGE,
)
from ..utils.console_logger import logger


# Method definitions
METHODS = {
    "image_direct": {
        "description": "Image-only direct generation (Design2Code baseline)",
        "input_type": "image",
    },
    "image_text_augmented": {
        "description": "Image + extracted text (Design2Code baseline)",
        "input_type": "image_text",
    },
    "figma_direct": {
        "description": "Figma JSON only (metadata-based)",
        "input_type": "figma",
    },
    "figma_image_direct": {
        "description": "Figma JSON + page image",
        "input_type": "figma_image",
    },
    "figma_image_agent": {
        "description": "Agent-based with critic-refiner loop",
        "input_type": "agent",
    },
}

# Default methods to test
DEFAULT_METHODS = [
    "image_direct",
    "image_text_augmented",
    "figma_direct",
    "figma_image_direct",
    "figma_image_agent",  # Agent method is slower, run separately if needed
]

# Default model for exp2
DEFAULT_MODEL = "ernie4_5_vl_424b_a47b"


@dataclass
class Exp2Config(ExperimentConfig):
    """Configuration for exp2 method comparison."""
    model: str = DEFAULT_MODEL
    methods: List[str] = field(default_factory=lambda: DEFAULT_METHODS.copy())
    agent_max_steps: int = 5
    use_url: bool = True # only used when input contains image
    add_guidelines: bool = False # only for figma_direct and figma_image_direct methods


class Exp2MethodComparison(ExperimentBase):
    """
    Experiment 2: Method Comparison.
    
    Tests different generation methods with the same model to compare
    the effectiveness of various input types and generation approaches.
    """
    
    def __init__(
        self,
        config: Optional[Exp2Config] = None,
        model: Optional[str] = None,
        methods: Optional[List[str]] = None
    ):
        """
        Initialize exp2 runner.
        
        Args:
            config: Experiment configuration
            model: Model name to use (overrides config.model)
            methods: List of method names to test (overrides config.methods)
        """
        if config is None:
            config = Exp2Config()
        super().__init__(config)
        
        # Override model and methods if provided
        if model is not None:
            self.config.model = model
        if methods is not None:
            self.config.methods = methods
        if self.config.add_guidelines:
            self.config.methods = [method for method in self.config.methods if method in ["figma_direct", "figma_image_direct"]]
            if not self.config.save_prefix:
                self.config.save_prefix = "guide_"
        
        # Validate model
        if self.config.model not in MODELS:
            available = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model: {self.config.model}. Available: {available}")
        
        # Validate methods
        for method in self.config.methods:
            if method not in METHODS:
                available = ", ".join(METHODS.keys())
                raise ValueError(f"Unknown method: {method}. Available: {available}")
        
        # LLM instances
        self._backbone: Optional[OpenRouterLLM] = None
        self._backbone_critic: Optional[OpenRouterLLM] = None
    
    @property
    def experiment_name(self) -> str:
        return "exp2"
    
    @property
    def description(self) -> str:
        return f"Method comparison using model: {self.config.model}"
    
    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Return run configurations for all methods."""
        return [{"method": method, "model": self.config.model} for method in self.config.methods]
    
    def get_output_filename(self, run_config: Dict[str, Any]) -> str:
        """
        Get output filename for a method.
        
        Format: {save_prefix}{method}__{model}.html
        Example: figma_direct__ernie4_5_vl_424b_a47b.html
        """
        method = run_config["method"]
        model = run_config["model"]
        return f"{self.config.save_prefix}{method}__{model}.html"
    
    def _get_backbone(self) -> OpenRouterLLM:
        """Get or create main LLM backbone."""
        if self._backbone is None:
            logger.info(f"Creating backbone LLM: {self.config.model}")
            self._backbone = create_llm(
                self.config.model,
                max_tokens=None,
                img_max_width=1024,
                best_provider=True
            )
        return self._backbone
    
    def _get_backbone_critic(self) -> OpenRouterLLM:
        """Get or create critic LLM backbone (for agent method)."""
        if self._backbone_critic is None:
            logger.info(f"Creating critic backbone LLM: {self.config.model}")
            self._backbone_critic = create_llm(
                self.config.model,
                max_tokens=None,
                img_max_width=1024,
                best_provider=True,
                temperature=0.5
            )
        return self._backbone_critic
    
    def generate_single(
        self,
        sample_dir: Path,
        run_config: Dict[str, Any]
    ) -> str:
        """Generate HTML for a single sample using specified method."""
        method = run_config["method"]
        backbone = self._get_backbone()
        
        if method == "image_direct":
            return direct_prompting(sample_dir, backbone, use_url=self.config.use_url)
        
        elif method == "image_text_augmented":
            return text_augmented_prompting(sample_dir, backbone, use_url=self.config.use_url)
        
        elif method == "figma_direct":
            return direct_generation(
                sample_dir, backbone,
                input_type=INPUT_FIGMA,
                include_guidelines=self.config.add_guidelines
            )
        
        elif method == "figma_image_direct":
            use_small_img = self.config.model.startswith("claude")
            return direct_generation(
                sample_dir, backbone,
                input_type=INPUT_FIGMA_IMAGE if self.config.use_url else INPUT_FIGMA_URL,
                use_small_img=use_small_img,
                include_guidelines=self.config.add_guidelines
            )
        
        elif method == "figma_image_agent":
            backbone_critic = self._get_backbone_critic()
            return agent_generation(
                sample_dir,
                backbone_critic=backbone_critic,
                backbone_refiner=backbone,
                max_steps=self.config.agent_max_steps
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_run_info(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract method info for result rows."""
        return {
            "method": run_config["method"],
            "model": self.config.model
        }
    
    def run_single_method(self, method_name: str) -> Dict[str, Any]:
        """
        Run experiment for a single method.
        
        Args:
            method_name: Name of the method to run
        
        Returns:
            Results dictionary
        """
        if method_name not in METHODS:
            raise ValueError(f"Unknown method: {method_name}")
        
        return self.run(run_configs=[{"method": method_name}])


def run_exp2(
    model: Optional[str] = None,
    methods: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
    samples: Optional[List[str]] = None,
    replace: bool = False,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    agent_max_steps: int = 5,
    use_url: bool = True,
    save_prefix: str = "",
    add_guidelines: bool = False,
    replace_evaluation: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run exp2.
    
    Args:
        model: Model name to use (default: DEFAULT_MODEL)
        methods: List of method names to test (default: DEFAULT_METHODS)
        data_dir: Data directory path
        samples: Specific sample keys to process
        replace: Replace existing HTML files
        skip_generation: Skip generation phase
        skip_evaluation: Skip evaluation phase
        agent_max_steps: Max steps for agent method
        use_url: Use Figma URL input (default: True)
        save_prefix: Extra prefix for the saved file name
        add_guidelines: Add guidelines to the generated HTML
        replace_evaluation: If True, re-evaluate all samples; if False, skip already evaluated
    
    Returns:
        Results dictionary
    """
    config = Exp2Config(
        samples=samples,
        replace=replace,
        skip_generation=skip_generation,
        skip_evaluation=skip_evaluation,
        agent_max_steps=agent_max_steps,
        use_url=use_url,
        save_prefix=save_prefix,
        add_guidelines=add_guidelines,
        replace_evaluation=replace_evaluation
    )
    if data_dir:
        config.data_dir = data_dir
    
    experiment = Exp2MethodComparison(config, model=model, methods=methods)
    return experiment.run()