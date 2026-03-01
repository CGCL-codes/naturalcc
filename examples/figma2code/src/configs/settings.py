"""
Environment settings for Figma2Code.

Settings are loaded from environment variables and .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # API Keys
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    
    # Proxy settings (optional)
    http_proxy: Optional[str] = field(default_factory=lambda: os.getenv("HTTP_PROXY"))
    https_proxy: Optional[str] = field(default_factory=lambda: os.getenv("HTTPS_PROXY"))
    
    # Figma credentials (for data collection)
    figma_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FIGMA_API_KEY"))
    figma_email: Optional[str] = field(default_factory=lambda: os.getenv("FIGMA_EMAIL"))
    figma_password: Optional[str] = field(default_factory=lambda: os.getenv("FIGMA_PASSWORD"))
    
    # Image hosting settings
    image_host_prefix: str = field(
        default_factory=lambda: os.getenv(
            "IMAGE_HOST_PREFIX", 
            "https://raw.githubusercontent.com/gystar/img_hub/main/figma/"
        )
    )
    image_host_prefix_small: str = field(
        default_factory=lambda: os.getenv(
            "IMAGE_HOST_PREFIX_SMALL",
            "https://raw.githubusercontent.com/gystar/img_hub/main/figma_small/"
        )
    )
    
    def __post_init__(self):
        """Apply proxy settings to environment if configured."""
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
        if self.https_proxy:
            os.environ["https_proxy"] = self.https_proxy
    

# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience instance (created on first import)
settings = get_settings()
