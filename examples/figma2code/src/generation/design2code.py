"""
Design2Code baseline generation methods.

Implements image-based code generation methods from the Design2Code paper:
- Direct prompting: Image only
- Text-augmented prompting: Image + extracted text
- Visual revision prompting: Iterative refinement

Reference: https://github.com/NoviScl/Design2Code
"""

import re
import json
from pathlib import Path
from typing import Union, List, Optional

from PIL import Image

from .base import BaseGeneration
from ..llm.base import BaseLLM
from ..utils.image import get_root_url
from ..utils.console_logger import logger


def _extract_texts_from_figma(figma_json_path: Union[str, Path]) -> List[str]:
    """
    Extract all text content from Figma JSON.
    
    Args:
        figma_json_path: Path to processed_metadata.json
    
    Returns:
        List of text strings found in TEXT nodes
    """
    texts = []
    
    with open(figma_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT" and "characters" in node:
                texts.append(node["characters"])
            for child in node.get("children", []):
                walk(child)
    
    walk(data.get("document", data))
    return texts


def _extract_text_from_html(html_content: str) -> List[str]:
    """
    Extract visible text content from HTML.
    
    Args:
        html_content: HTML string
    
    Returns:
        List of text strings
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return [chunk for chunk in chunks if chunk]
    except ImportError:
        logger.warning("BeautifulSoup not installed, using regex fallback")
        # Simple regex fallback
        text = re.sub(r'<[^>]+>', ' ', html_content)
        return [t.strip() for t in text.split() if t.strip()]


def _parse_html_output(raw: str) -> str:
    """
    Extract HTML content from model output.
    
    Args:
        raw: Raw model output
    
    Returns:
        Cleaned HTML string
    """
    # First, remove the leading ```html or ``` (possibly with spaces)
    cleaned = re.sub(r"^```(?:html)?\s*", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # Then, remove the trailing ``` (possibly with spaces, newlines)
    cleaned = re.sub(r"\s*```(?:\s*)$", "", cleaned, flags=re.MULTILINE)
    
    # Extract content between <!DOCTYPE html> and </html>
    match = re.search(r"(<!DOCTYPE\s+html.*?</html>)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: if DOCTYPE is not found, return the content after removing code block symbols
    return cleaned.strip()

class ImageGeneration(BaseGeneration):
    """
    Image-based code generation.

    Takes a screenshot of the design and generates HTML code.
    Uses a generic placeholder for images.
    """

    def __init__(self, use_url: bool = True):
        """
        Args:
            use_url: If True, use hosted image URL instead of local file
        """
        self.use_url = use_url


class ImageDirectGeneration(ImageGeneration):
    """
    Direct image-based code generation.
    """

    @property
    def method_name(self) -> str:
        return "image_direct"
    
    def generate(
        self,
        data_dir: Path,
        backbone: BaseLLM,
        **kwargs
    ) -> str:
        """
        Generate HTML from design screenshot.
        
        Args:
            data_dir: Directory containing root.png
            backbone: LLM backend
        
        Returns:
            Generated HTML string
        """
        data_dir = Path(data_dir)
        
        prompt = (
            "You are an expert web developer who specializes in HTML and CSS.\n"
            "A user will provide you with a screenshot of a webpage.\n"
            "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
            "Include all CSS code in the HTML file itself.\n"
            "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
            "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
            "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
            "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
            "Respond with the content of the HTML+CSS file:\n"
        )
        
        if self.use_url:
            key = data_dir.name
            input_image = get_root_url(key)
        else:
            input_image = Image.open(data_dir / "root.png").convert("RGB")
        
        raw = backbone(prompt, [input_image])
        return _parse_html_output(raw)


class TextAugmentedGeneration(ImageGeneration):
    """
    Text-augmented image-based code generation.
    
    Takes a screenshot plus extracted text content to improve text accuracy.
    """
    
    @property
    def method_name(self) -> str:
        return "image_text_augmented"
    
    def generate(
        self,
        data_dir: Path,
        backbone: BaseLLM,
        figma_json_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        """
        Generate HTML from design screenshot with text hints.
        
        Args:
            data_dir: Directory containing root.png and processed_metadata.json
            backbone: LLM backend
        
        Returns:
            Generated HTML string
        """
        data_dir = Path(data_dir) 
        if figma_json_path is None:
            figma_json_path = data_dir / "processed_metadata.json"
        
        # Extract text from Figma JSON
        texts = _extract_texts_from_figma(figma_json_path)
        texts_str = "\n".join(texts)
        
        prompt = (
            "You are an expert web developer who specializes in HTML and CSS.\n"
            "A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.\n"
            f"The text elements are:\n{texts_str}\n"
            "You should generate the correct layout structure for the webpage, and put the texts in the correct places so that the resultant webpage will look the same as the given one.\n"
            "You need to return a single html file that uses HTML and CSS to reproduce the given website.\n"
            "Include all CSS code in the HTML file itself.\n"
            "If it involves any images, use \"rick.jpg\" as the placeholder.\n"
            "Some images on the webpage are replaced with a blue rectangle as the placeholder, use \"rick.jpg\" for those as well.\n"
            "Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.\n"
            "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
            "Respond with the content of the HTML+CSS file (directly start with the code, do not add any additional explanation):\n"
        )
        
        if self.use_url:
            key = data_dir.name
            input_image = get_root_url(key)
        else:
            input_image = Image.open(data_dir / "root.png").convert("RGB")
        
        raw = backbone(prompt, [input_image])
        return _parse_html_output(raw)

# Note: Not used in the paper
class VisualRevisionGeneration(ImageGeneration):
    """
    Visual revision-based code generation.
    
    Iteratively refines generated HTML by comparing screenshots.
    """
    
    @property
    def method_name(self) -> str:
        return "image_visual_revision"
    
    def generate(
        self,
        data_dir: Path,
        backbone: BaseLLM,
        initial_html: Optional[str] = None,
        initial_screenshot: Optional[Image.Image] = None,
        **kwargs
    ) -> str:
        """
        Generate/refine HTML through visual comparison.
        
        Args:
            data_dir: Directory containing root.png
            backbone: LLM backend
            initial_html: Initial HTML to refine (if None, generates from scratch)
            initial_screenshot: Screenshot of initial HTML
        
        Returns:
            Refined HTML string
        """
        data_dir = Path(data_dir)
        
        if initial_html is None:
            # First generate initial version
            image_gen = ImageDirectGeneration(self.use_url)
            initial_html = image_gen.generate(data_dir, backbone)
        
        if initial_screenshot is None:
            logger.warning("No initial screenshot provided, skipping revision")
            return initial_html
        
        # Load reference image
        if self.use_url:
            key = data_dir.name
            ref_image = get_root_url(key)
        else:
            ref_image = Image.open(data_dir / "root.png").convert("RGB")
        
        # Extract text from current HTML
        texts = _extract_text_from_html(initial_html)
        texts_str = "\n".join(texts)
        
        prompt = (
            "You are an expert web developer who specializes in HTML and CSS.\n"
            "I have an HTML file for implementing a webpage but it has some missing or wrong elements that are different from the original webpage. The current implementation I have is:\n"
            f"{initial_html}\n\n"
            "I will provide the reference webpage that I want to build as well as the rendered webpage of the current implementation.\n"
            "I also provide you all the texts that I want to include in the webpage here:\n"
            f"{texts_str}\n\n"
            "Please compare the two webpages and refer to the provided text elements to be included, and revise the original HTML implementation to make it look exactly like the reference webpage. "
            "Make sure the code is syntactically correct and can render into a well-formed webpage. You can use \"rick.jpg\" as the placeholder image file.\n"
            "Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.\n"
            "Respond directly with the content of the new revised and improved HTML file without any extra explanations:\n"
        )
        
        raw = backbone(
            prompt, 
            ["The first image is reference webpage, and the second image is current webpage", 
             ref_image, 
             initial_screenshot]
        )
        
        return _parse_html_output(raw)


def direct_prompting(
    image_path: Union[str, Path],
    backbone: BaseLLM,
    use_url: bool = True
) -> str:
    """
    Convenience function for image-based direct generation.
    
    Args:
        image_path: Path to design image or data directory
        backbone: LLM backend
        use_url: If True, use hosted image URL instead of local file
    
    Returns:
        Generated HTML string
    """
    image_path = Path(image_path)
    
    # If path is to image file, get parent directory
    if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        data_dir = image_path.parent
    else:
        data_dir = image_path
    
    generator = ImageDirectGeneration(use_url)
    return generator.generate(data_dir, backbone)


def text_augmented_prompting(
    image_path: Union[str, Path],
    backbone: BaseLLM,
    use_url: bool = True,
    figma_json_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Convenience function for text-augmented generation.
    
    Args:
        image_path: Path to design image or data directory
        backbone: LLM backend
        use_url: If True, use hosted image URL instead of local file
        figma_json_path: Path to Figma JSON (can be None if in data_dir)
    
    Returns:
        Generated HTML string
    """
    image_path = Path(image_path)
    
    if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        data_dir = image_path.parent
    else:
        data_dir = image_path
    
    generator = TextAugmentedGeneration(use_url)
    return generator.generate(data_dir, backbone, figma_json_path)
