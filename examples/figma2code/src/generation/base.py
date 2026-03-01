"""
Base class for code generation methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ..llm.base import BaseLLM


class BaseGeneration(ABC):
    """
    Abstract base class for code generation methods.
    
    All generation methods should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def generate(
        self,
        data_dir: Path,
        backbone: BaseLLM,
        **kwargs
    ) -> str:
        """
        Generate HTML code from input data.
        
        Args:
            data_dir: Directory containing input data (root.png, processed_metadata.json, assets/)
            backbone: LLM backend for generation
            **kwargs: Additional method-specific parameters
        
        Returns:
            Generated HTML string
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method identifier for this generation approach."""
        pass


class GenerationError(Exception):
    """Base exception for generation-related errors."""
    pass


class InputValidationError(GenerationError):
    """Raised when input data is invalid or missing required files."""
    pass


class OutputParsingError(GenerationError):
    """Raised when model output cannot be parsed."""
    pass
