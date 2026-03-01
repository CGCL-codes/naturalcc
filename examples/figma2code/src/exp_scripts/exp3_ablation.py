"""
Experiment 3: Ablation Study

Studies the importance of different input components by systematically
removing specific information from Figma JSON metadata.

Ablation types:
- geometry: Remove layout information (x, y, width, height, transforms)
- style: Remove visual styles (colors, fonts, effects)
- image_refs: Remove image references
- structure: Flatten node hierarchy
- text: Remove text content

Methods:
- figma_direct: Figma JSON only with ablation
- figma_image_direct: Figma JSON + image with ablation

Output filename format: {save_prefix}{method}__ablate_{ablation}__{model}.html
Example: figma_direct__ablate_geometry__ernie4_5_vl_424b_a47b.html
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .base import ExperimentBase, ExperimentConfig
from ..configs import MODELS
from ..llm import create_llm, OpenRouterLLM
from ..generation import (
    direct_generation,
    INPUT_FIGMA,
    INPUT_FIGMA_URL,
    INPUT_FIGMA_IMAGE,
    ABLATION_FUNCTIONS,
    get_ablation_function,
    list_ablation_types,
)
from ..utils.console_logger import logger


# Methods that support ablation
ABLATION_METHODS = {
    "figma_direct": {
        "description": "Figma JSON only with ablation",
        "input_type": "figma",
    },
    "figma_image_direct": {
        "description": "Figma JSON + image with ablation",
        "input_type": "figma_image",
    },
}

# Default ablation types
DEFAULT_ABLATION_TYPES = list_ablation_types()

# Default model for exp3
DEFAULT_MODEL = "ernie4_5_vl_424b_a47b"

# Default method for exp3
DEFAULT_METHODS = list(ABLATION_METHODS.keys())


@dataclass
class Exp3Config(ExperimentConfig):
    """Configuration for exp3 ablation study."""
    model: str = DEFAULT_MODEL
    methods: List[str] = field(default_factory= lambda: DEFAULT_METHODS.copy())
    ablation_types: List[str] = field(default_factory=lambda: DEFAULT_ABLATION_TYPES.copy())
    use_url: bool = True


class Exp3Ablation(ExperimentBase):
    """
    Experiment 3: Ablation Study.
    
    Studies the importance of different input components by systematically
    removing specific information from Figma JSON metadata.
    """
    
    def __init__(
        self,
        config: Optional[Exp3Config] = None,
        model: Optional[str] = None,
        methods: Optional[List[str]] = None,
        ablation_types: Optional[List[str]] = None
    ):
        """
        Initialize exp3 runner.
        
        Args:
            config: Experiment configuration
            model: Model name to use (overrides config.model)
            methods: Method names to use (overrides config.methods)
            ablation_types: List of ablation types (overrides config.ablation_types)
        """
        if config is None:
            config = Exp3Config()
        super().__init__(config)
        
        # Override parameters if provided
        if model is not None:
            self.config.model = model
        if methods is not None:
            self.config.methods = methods
        if ablation_types is not None:
            self.config.ablation_types = ablation_types
        
        # Validate model
        if self.config.model not in MODELS:
            available = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model: {self.config.model}. Available: {available}")
        
                # Validate methods
        for method in self.config.methods:
            if method not in ABLATION_METHODS:
                available = ", ".join(ABLATION_METHODS.keys())
                raise ValueError(f"Unknown method: {method}. Available: {available}")
        
        # Validate ablation types
        for ablation in self.config.ablation_types:
            if ablation not in ABLATION_FUNCTIONS:
                available = ", ".join(ABLATION_FUNCTIONS.keys())
                raise ValueError(f"Unknown ablation type: {ablation}. Available: {available}")
        
        # LLM instance
        self._backbone: Optional[OpenRouterLLM] = None
    
    @property
    def experiment_name(self) -> str:
        return "exp3"
    
    @property
    def description(self) -> str:
        return f"Ablation study using {self.config.method} with model: {self.config.model}"
    
    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Return run configurations for all ablation types."""
        return [
            {"method": method, "ablation": ablation, "model": self.config.model} 
            for method in self.config.methods 
            for ablation in self.config.ablation_types
        ]
    
    def get_output_filename(self, run_config: Dict[str, Any]) -> str:
        """
        Get output filename for an ablation type.
        
        Format: {save_prefix}{method}__ablate_{ablation}__{model}.html
        Example: figma_direct__ablate_geometry__ernie4_5_vl_424b_a47b.html
        """
        method = run_config["method"]
        ablation = run_config["ablation"]
        model = run_config["model"]
        return f"{self.config.save_prefix}{method}__ablate_{ablation}__{model}.html"
    
    def _get_backbone(self) -> OpenRouterLLM:
        """Get or create LLM backbone."""
        if self._backbone is None:
            logger.info(f"Creating backbone LLM: {self.config.model}")
            self._backbone = create_llm(
                self.config.model,
                max_tokens=None,
                img_max_width=1024,
                best_provider=True
            )
        return self._backbone
    
    def generate_single(
        self,
        sample_dir: Path,
        run_config: Dict[str, Any]
    ) -> str:
        """Generate HTML for a single sample with ablation."""
        ablation_type = run_config["ablation"]
        backbone = self._get_backbone()
        ablation_func = get_ablation_function(ablation_type)
        
        method_info = ABLATION_METHODS[run_config["method"]]
        input_type = INPUT_FIGMA if method_info["input_type"] == "figma" else INPUT_FIGMA_URL if self.config.use_url else INPUT_FIGMA_IMAGE
        
        use_small_img = self.config.model.startswith("claude")
        
        return direct_generation(
            sample_dir,
            backbone,
            input_type=input_type,
            use_small_img=use_small_img,
            ablation_func=ablation_func
        )
    
    def _extract_run_info(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ablation info for result rows."""
        return {
            "method": run_config["method"],
            "model": self.config.model,
            "ablation": run_config["ablation"]
        }
    
    def run_single_ablation(self, method: str, ablation_type: str) -> Dict[str, Any]:
        """
        Run experiment for a single ablation type.
        
        Args:
            method: Method to use
            ablation_type: Type of ablation to run
        
        Returns:
            Results dictionary
        """
        if ablation_type not in ABLATION_FUNCTIONS:
            available = ", ".join(ABLATION_FUNCTIONS.keys())
            raise ValueError(f"Unknown ablation type: {ablation_type}. Available: {available}")
        
        if method not in ABLATION_METHODS:
            available = ", ".join(ABLATION_METHODS.keys())
            raise ValueError(f"Unknown method: {method}. Available: {available}")
        return self.run(run_configs=[{"method": method, "ablation": ablation_type}])


def run_exp3(
    model: Optional[str] = None,
    methods: Optional[List[str]] = None,
    ablation_types: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
    samples: Optional[List[str]] = None,
    replace: bool = False,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    use_url: bool = True,
    save_prefix: str = "",
    replace_evaluation: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run exp3.
    
    Args:
        model: Model name to use (default: DEFAULT_MODEL)
        methods: Method names to use (default: DEFAULT_METHODS)
        ablation_types: List of ablation types (default: DEFAULT_ABLATION_TYPES)
        data_dir: Data directory path
        samples: Specific sample keys to process
        replace: Replace existing HTML files
        skip_generation: Skip generation phase
        skip_evaluation: Skip evaluation phase
        use_url: Use Figma URL input (default: True)
        save_prefix: Extra prefix for the saved file name
        replace_evaluation: If True, re-evaluate all samples; if False, skip already evaluated
    
    Returns:
        Results dictionary
    """
    config = Exp3Config(
        samples=samples,
        replace=replace,
        skip_generation=skip_generation,
        skip_evaluation=skip_evaluation,
        use_url=use_url,
        save_prefix=save_prefix,
        replace_evaluation=replace_evaluation
    )
    if data_dir:
        config.data_dir = data_dir
    
    experiment = Exp3Ablation(
        config,
        model=model,
        methods=methods,
        ablation_types=ablation_types
    )
    return experiment.run()