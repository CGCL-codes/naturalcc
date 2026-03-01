"""
LLM model configurations for Figma2Code.

This module defines the available models and their configurations.
"""

from typing import Dict, Any

# Default random seed for reproducibility
DEFAULT_SEED = 2026

# Model configurations
# Each model has: model (API identifier), base_url, and optional parameters
MODELS: Dict[str, Dict[str, Any]] = {
    # OpenAI models
    "gpt4o": {
        "model": "openai/gpt-4o",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gpt5": {
        "model": "openai/gpt-5",
        "base_url": "https://openrouter.ai/api/v1"
    },
    
    # Anthropic models
    "claude_opus_4_1": {
        "model": "anthropic/claude-opus-4.1",
        "base_url": "https://openrouter.ai/api/v1"
    },
    
    # Google models
    "gemini2_5_pro": {
        "model": "google/gemini-2.5-pro",
        "base_url": "https://openrouter.ai/api/v1"
    },
    # "gemini2_5_pro": {
    #     "model": "gemini-2.5-pro",
    #     "base_url": "https://api.apiduke.com/v1",
    # },
    
    # Qwen models
    "qwen2_5_vl": {
        "model": "qwen/qwen2.5-vl-72b-instruct",
        "base_url": "https://openrouter.ai/api/v1"
    },
    
    # Baidu ERNIE models
    "ernie4_5_vl_424b_a47b": {
        "model": "baidu/ernie-4.5-vl-424b-a47b",
        "base_url": "https://openrouter.ai/api/v1",
    },
    
    # xAI Grok models
    "grok4": {
        "model": "x-ai/grok-4",
        "base_url": "https://openrouter.ai/api/v1"
    },
    
    # Meta Llama models
    "llama4_maverick": {
        "model": "meta-llama/llama-4-maverick",
        "base_url": "https://openrouter.ai/api/v1"
    },
    "llama4_scout": {
        "model": "meta-llama/llama-4-scout",
        "base_url": "https://openrouter.ai/api/v1"
    },

    # Other models
    "nova_pro_v1": {
        "model": "amazon/nova-pro-v1",
        "base_url": "https://openrouter.ai/api/v1"
    },
    # "glm4_5v": {
    #     "model": "z-ai/glm-4.5v",
    #     "base_url": "https://openrouter.ai/api/v1"
    # },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (key in MODELS dict)
    
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model_name is not found
    """
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    return MODELS[model_name]