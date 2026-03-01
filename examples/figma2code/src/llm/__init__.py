"""
LLM provider interfaces for Figma2Code.

This module provides unified interfaces for interacting with various LLM providers:
- OpenRouter (GPT, Claude, Gemini, etc.)
- DeepSeek (local vLLM)
"""

from .base import BaseLLM
from .openrouter import OpenRouterLLM, create_llm, gpt

__all__ = ["BaseLLM", "OpenRouterLLM", "create_llm", "gpt"]
