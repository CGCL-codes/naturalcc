"""
OpenRouter LLM provider implementation.

Provides access to various LLM models through the OpenRouter API.
"""

import json
import time
from typing import List, Union, Optional, Dict, Any

import httpx
import requests
import tiktoken
from openai import OpenAI
from PIL import Image

from .base import BaseLLM, LLMConnectionError, LLMResponseError
from ..configs.settings import get_settings
from ..configs.models import DEFAULT_SEED, get_model_config
from ..utils.image import encode_image_to_url
from ..utils.console_logger import logger

def count_message_tokens(messages: List[dict], model: str = "openai/gpt-4o") -> tuple:
    """
    Estimate the token count for input messages.
    
    Separates text tokens from image tokens (base64 encoded images).
    
    Args:
        messages: Message list in OpenAI format
        model: Model name (for tokenizer selection)
    
    Returns:
        Tuple of (text_tokens, image_tokens)
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    text_segments = []
    image_segments = []
    
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            text_segments.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_segments.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        image_segments.append(url)
    
    text_tokens = len(enc.encode("\n".join(text_segments))) if text_segments else 0
    image_tokens = len(enc.encode("\n".join(image_segments))) if image_segments else 0
    
    return text_tokens, image_tokens


def format_messages(
    prompt: str, 
    texts_imgs: List[Union[str, Image.Image]], 
    img_max_width: int = 1024
) -> List[dict]:
    """
    Format prompt and user content into OpenAI message format.
    
    Args:
        prompt: System prompt text
        texts_imgs: List of text strings and/or PIL Images.
                   Strings starting with http:// or https:// are treated as image URLs.
        img_max_width: Maximum width for image resizing
    
    Returns:
        Formatted message list
    """
    user_content = []
    
    for c in texts_imgs:
        if isinstance(c, Image.Image):
            # PIL.Image -> base64
            data_url = encode_image_to_url(c, max_width=img_max_width)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })
        elif isinstance(c, str):
            if c.startswith("http://") or c.startswith("https://"):
                # URL string -> image URL
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": c}
                })
            else:
                # Other strings -> text
                user_content.append({
                    "type": "text",
                    "text": c
                })
        else:
            raise ValueError(f"Unsupported content type: {type(c)}")
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def get_best_provider(model_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Query OpenRouter for model info and return the provider with max context length.
    
    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o")
        api_key: OpenRouter API key
    
    Returns:
        Provider info dict or None if not found
    """
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to query model info: {e}")
        return None
    
    models = resp.json().get("data", [])
    
    for m in models:
        if m["id"] == model_id:
            providers = []
            
            if "top_provider" in m and m["top_provider"]:
                providers.append(("top_provider", m["top_provider"]))
            
            if "providers" in m:
                for name, info in m["providers"].items():
                    providers.append((name, info))
            
            if not providers:
                return None
            
            # Find provider with maximum context length
            best = max(providers, key=lambda x: x[1].get("context_length", 0))
            name, info = best
            
            return {
                "model": model_id,
                "provider": name,
                "context_length": info.get("context_length"),
                "max_completion_tokens": info.get("max_completion_tokens")
            }
    
    return None


def gpt(
    messages: List[dict],
    temperature: float = 0,
    seed: int = 0,
    request_timeout: int = 120,
    max_tokens: Optional[int] = None,
    model: str = "openai/gpt-4o-2024-11-20",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 2,
    provider: Optional[str] = None,
    stream: bool = True,
) -> str:
    """
    Send a chat completion request to OpenRouter (OpenAI-compatible API).

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        seed: Random seed
        request_timeout: Request timeout in seconds
        max_tokens: Max tokens in response
        model: Model id
        base_url: API base URL
        api_key: API key (defaults to settings)
        max_retries: Retry count on connection errors
        retry_delay: Delay between retries in seconds
        provider: Optional OpenRouter provider hint
        stream: Use streaming response

    Returns:
        Generated text response
    """
    key = api_key or get_settings().openrouter_api_key
    client = OpenAI(base_url=base_url, api_key=key)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            extra_headers = {}
            if provider:
                extra_headers["X-OpenRouter-Provider"] = provider
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=1,
                seed=seed,
                timeout=request_timeout,
                **({"extra_headers": extra_headers} if extra_headers else {}),
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                stream=stream,
            )
            if not stream:
                return resp.choices[0].message.content or ""
            collected = []
            for chunk in resp:
                if chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            return "".join(collected)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse JSON from response: %s", e)
            raise LLMResponseError(f"Invalid JSON response: {e}") from e
        except (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ) as e:
            last_err = e
            if attempt < max_retries:
                logger.error(
                    "Connection error, retrying %s/%s in %ss: %s",
                    attempt,
                    max_retries,
                    retry_delay,
                    e,
                )
                time.sleep(retry_delay)
                continue
            raise LLMConnectionError(
                f"Connection failed after {max_retries} retries: {e}"
            ) from e
        except Exception as e:
            raise LLMResponseError(f"LLM request failed: {e}") from e
    raise LLMConnectionError(
        f"Request failed after {max_retries} retries: {last_err}"
    )


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM provider implementation.
    
    Supports various models through the OpenRouter API, including OpenAI, Anthropic,
    Google, and other providers.
    """
    
    def __init__(
        self,
        model: str = "openai/gpt-4o-2024-11-20",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0,
        seed: int = DEFAULT_SEED,
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 2,
        img_max_width: int = 1024,
        best_provider: bool = False,
        stream: bool = True
    ):
        """
        Initialize OpenRouter LLM provider.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            base_url: API base URL
            api_key: API key (defaults to OPENROUTER_API_KEY from env)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0 = deterministic)
            seed: Random seed for reproducibility
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for connection errors
            retry_delay: Delay between retries in seconds
            img_max_width: Maximum image width (larger images are resized)
            best_provider: If True, query for best provider by context length
            stream: If True, use streaming responses
        """
        self.model = model
        self.base_url = base_url
        self._api_key = api_key or get_settings().openrouter_api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.img_max_width = img_max_width
        self.stream = stream
        
        self.provider = None
        if best_provider and self._api_key:
            model_info = get_best_provider(self.model, self._api_key)
            if model_info:
                logger.info(f"Model info: {model_info}")
                self.provider = model_info.get("provider")
    
    def __call__(
        self, 
        prompt: str, 
        texts_imgs: Optional[List[Union[str, Image.Image]]] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: System prompt / instructions
            texts_imgs: List of text strings and/or PIL Images for user content
        
        Returns:
            Generated text response
        """
        if texts_imgs is None:
            texts_imgs = []
        
        messages = format_messages(prompt, texts_imgs, self.img_max_width)
        
        text_tokens, image_tokens = count_message_tokens(messages, self.model)
        logger.debug(f"Message tokens - text: {text_tokens}, image: {image_tokens}")

        # provider_context_length = self.model_info.get("context_length", None) if self.model_info else None
        # if not self.max_tokens and provider_context_length:
        #     tokens_remain = provider_context_length - text_tokens - image_tokens
        #     self.max_tokens = None if tokens_remain <= 0 else tokens_remain
        #     logger.debug(f"[openrouter] Set max_tokens to {self.max_tokens}")   
            
        # test
        #self.max_tokens = None
        
        return self.chat(messages)
    
    def chat(self, messages: List[dict], **kwargs) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (override instance defaults)
        
        Returns:
            Generated text response
        """
        return gpt(
            messages,
            temperature=kwargs.get("temperature", self.temperature),
            seed=kwargs.get("seed", self.seed),
            request_timeout=self.request_timeout,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            model=self.model,
            base_url=self.base_url,
            api_key=self._api_key,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            provider=self.provider,
            stream=kwargs.get("stream", self.stream),
        )


def create_llm(model_name: str, **kwargs) -> OpenRouterLLM:
    """
    Factory function to create an LLM instance by model name.
    
    Args:
        model_name: Model name from MODELS config (e.g., "gpt4o", "claude_opus_4")
        **kwargs: Additional parameters to pass to OpenRouterLLM
    
    Returns:
        Configured OpenRouterLLM instance
    
    Raises:
        ValueError: If model_name is not found in MODELS
    """
    config = get_model_config(model_name)
    return OpenRouterLLM(
        model=config["model"],
        base_url=config["base_url"],
        **kwargs
    )
