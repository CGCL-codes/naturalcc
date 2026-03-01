"""
Experiment 1: Model Comparison

Compares different LLM models using the same generation method (figma_image_direct).
Tests the capability of various models to generate HTML from Figma designs.

Models tested:
- GPT-4o, GPT-5
- Claude Opus 4.1
- Gemini 2.5 Pro
- Qwen 2.5 VL
- ERNIE 4.5 VL
- Grok 4
- Llama 4 (Maverick, Scout)
- Nova Pro V1

Output filename format: {save_prefix}{method}__{model}.html
Example: figma_image_direct__gpt4o.html
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .base import ExperimentBase, ExperimentConfig
from ..configs import MODELS
from ..llm import create_llm, OpenRouterLLM
from ..generation import direct_generation, INPUT_FIGMA_URL, INPUT_FIGMA_IMAGE
from ..utils.console_logger import logger

# Default models to test in exp1
DEFAULT_MODELS = [
    "llama4_maverick",
    "llama4_scout",
    "qwen2_5_vl",
    "ernie4_5_vl_424b_a47b",
    "nova_pro_v1",
    "gemini2_5_pro",
    "grok4",
    "claude_opus_4_1",
    "gpt4o",
    "gpt5",
]

# Fixed method for exp1 (model comparison uses figma_image_direct)
EXP1_METHOD = "figma_image_direct"


@dataclass
class Exp1Config(ExperimentConfig):
    """Configuration for exp1 model comparison."""
    models: List[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    method: str = EXP1_METHOD  # Fixed method for exp1
    use_url: bool = True

class Exp1ModelComparison(ExperimentBase):
    """
    Experiment 1: Model Comparison.
    
    Tests different LLM models with the same generation method (figma+image input)
    to compare their capability in generating HTML from Figma designs.
    """
    
    def __init__(
        self,
        config: Optional[Exp1Config] = None,
        models: Optional[List[str]] = None
    ):
        """
        Initialize exp1 runner.
        
        Args:
            config: Experiment configuration
            models: List of model names to test (overrides config.models)
        """
        if config is None:
            config = Exp1Config()
        super().__init__(config)
        
        # Override models if provided
        if models is not None:
            self.config.models = models
        
        if config.use_url:
            self.input_type = INPUT_FIGMA_URL
        else:
            self.input_type = INPUT_FIGMA_IMAGE
        
        # Validate models
        for model in self.config.models:
            if model not in MODELS:
                available = ", ".join(MODELS.keys())
                raise ValueError(f"Unknown model: {model}. Available: {available}")
        
        # LLM cache
        self._llm_cache: Dict[str, OpenRouterLLM] = {}
    
    @property
    def experiment_name(self) -> str:
        return "exp1"
    
    @property
    def description(self) -> str:
        return "Model comparison using figma+image input"
    
    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Return run configurations for all models."""
        return [{"model": model, "method": self.config.method} for model in self.config.models]
    
    def get_output_filename(self, run_config: Dict[str, Any]) -> str:
        """
        Get output filename for a model.
        
        Format: {save_prefix}{method}__{model}.html
        Example: figma_image_direct__gpt4o.html
        """
        model = run_config["model"]
        method = run_config["method"]
        return f"{self.config.save_prefix}{method}__{model}.html"
    
    def _get_llm(self, model_name: str) -> OpenRouterLLM:
        """Get or create LLM instance (cached)."""
        if model_name not in self._llm_cache:
            logger.info(f"Creating LLM for model: {model_name}")
            self._llm_cache[model_name] = create_llm(
                model_name,
                max_tokens=None,
                img_max_width=1024,
                best_provider=True
            )
        return self._llm_cache[model_name]
    
    def generate_single(
        self,
        sample_dir: Path,
        run_config: Dict[str, Any]
    ) -> str:
        """Generate HTML for a single sample using specified model."""
        model_name = run_config["model"]
        backbone = self._get_llm(model_name)
        
        # Claude requires smaller images
        use_small_img = model_name.startswith("claude")
        
        return direct_generation(
            sample_dir,
            backbone,
            input_type=self.input_type,
            use_small_img=use_small_img
        )
    
    def _extract_run_info(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model and method info for result rows."""
        return {
            "model": run_config["model"],
            "method": run_config["method"]
        }
    
    def run_single_model(self, model_name: str) -> Dict[str, Any]:
        """
        Run experiment for a single model.
        
        Args:
            model_name: Name of the model to run
        
        Returns:
            Results dictionary
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.run(run_configs=[{"model": model_name}])


def run_exp1(
    models: Optional[List[str]] = None,
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
    Convenience function to run exp1.
    
    Args:
        models: List of model names to test (default: DEFAULT_MODELS)
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
    config = Exp1Config(
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
    
    experiment = Exp1ModelComparison(config, models=models)
    return experiment.run()
