"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from PIL import Image

class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM implementations should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def __call__(
        self, 
        prompt: str, 
        texts_imgs: Optional[List[Union[str, Image.Image]]] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: System prompt / instructions
            texts_imgs: List of text strings and/or PIL Images for user content.
                       Strings starting with http:// or https:// are treated as image URLs.
        
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        **kwargs
    ) -> str:
        """
        Send a chat completion request with raw messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text response
        """
        pass

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or unexpected."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass
