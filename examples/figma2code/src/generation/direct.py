"""
Direct generation methods for Figma-to-Code.

Implements direct code generation using Figma JSON with optional image reference.
"""

import json
from pathlib import Path
from typing import Union, Optional, Callable, List

from .base import BaseGeneration, OutputParsingError, InputValidationError
from ..llm.base import BaseLLM
from ..utils.parsing import parse_output, remove_empty
from ..utils.image import get_root_url, load_image
from ..utils.console_logger import logger


# Input types
INPUT_FIGMA = "figma"
INPUT_FIGMA_IMAGE = "figma_image"
INPUT_FIGMA_URL = "figma_url"


def build_system_prompt(
    input_type: str,
    include_guidelines: bool = False
) -> str:
    """
    Build the system prompt for direct generation.
    
    Args:
        input_type: One of "figma", "figma_image", "figma_url"
        include_guidelines: Whether to include generation guidelines (for exp2 rebuttal)
    
    Returns:
        JSON-formatted system prompt string
    """
    base_prompt = {
        "task": "",
        "inputs_contract": {
            "contains": ["figma_json (JSON)"],
            "preprocessing_notes": [
                "The provided Figma JSON is preprocessed.",
                "Originally, Figma JSON used imageRef as opaque IDs (e.g. '347ba7a7c57adabed33deffd9e936c9b285f611e').",
                "We have downloaded these images and replaced imageRef with the actual local relative path (e.g. 'assets/foo.png', 'assets/bar.svg').",
                "Some vectors were merged and exported as local SVG files. A Rectangle node was created to hold each exported SVG, and the local file path is written into its imageRef.",
                "Unnecessary properties have been removed to simplify the JSON."
            ]
        },
        "output": {
            "format": "html_only (valid HTML document with no extra text before or after).",
            "example": "<!DOCTYPE html>\n<html lang=\"en\">\n<head> ... </head>\n<body> ... </body>\n</html>",
            "requirements": [
                "Return ONLY the complete HTML document.",
                "Do not wrap the output in JSON or any other structure.",
                "The HTML must use Tailwind CSS classes and preserve visual fidelity of the design."
            ]
        },
        "constraints": [
            "IMAGE ASSET BINDING — MANDATORY: For every visible node whose fills include an IMAGE with a non-empty, case-sensitive 'imageRef' value (already a local relative path), you MUST render that exact asset path in the HTML. This is a hard requirement.",
            "ALLOWED RENDERING FOR imageRef: Prefer <img src=\"{imageRef}\"> for placed/raster/SVG assets. Use CSS background-image only when the design explicitly uses the image as a background fill. When using Tailwind arbitrary values, escape properly: bg-[url('assets/foo.png')].",
            "DO NOT ALTER PATHS (SEMANTICS)",
            "NO HALLUCINATED ASSETS",
            "No external URLs except Tailwind CDN.",
        ],
    }
    
    # Set task description based on input type
    if input_type == INPUT_FIGMA:
        base_prompt["task"] = (
            "Generate a complete HTML document using Tailwind CSS classes from the provided Figma JSON. "
            "Adjust the generation strategy according to the preprocessing_notes in inputs_contract. "
            "Implementation should ensure that the code not only reproduces the design draft visually, but also enhances responsiveness and maintainability. "
            "Figma JSON is the primary source for layout, spacing, sizing, colors, typography, borders, radii, "
            "shadows/effects and asset bindings."
        )
    elif input_type in [INPUT_FIGMA_IMAGE, INPUT_FIGMA_URL]:
        base_prompt["task"] = (
            "Generate a complete HTML document using Tailwind CSS from the provided preprocessed Figma JSON and page screenshot. "
            "Adjust the generation strategy according to the preprocessing_notes in inputs_contract. "
            "The output must preserve the visual fidelity of the design while improving responsiveness and maintainability. "
            "The Figma JSON is the authoritative source for layout, spacing, sizing, colors, typography, borders, radii, "
            "shadows/effects, and asset bindings. "
            "The screenshot is only a secondary reference to resolve ambiguities or fill in missing details, and must never override explicit JSON values."
        )
        base_prompt["inputs_contract"]["contains"].append("page_screenshot (image)")
    
    # Add generation guidelines for better code quality
    if include_guidelines:
        base_prompt["generation_guidelines"] = [
            "RESPONSIVE LAYOUTS: Use Tailwind responsive prefixes (sm:, md:, lg:, xl:) so layouts adapt to different screen sizes. Prefer flexible layouts over fixed-width designs.",
            "AVOID DIRECT FIGMA COORDINATES: Do not rigidly copy x/y/width/height from Figma. Use natural flex/grid flow unless a part of the design clearly requires fixed positioning.",
            "USE FLEX/GRID FIRST: Favor flex or grid for structure, with gap and spacing utilities. Absolute positioning is allowed only when the design explicitly depends on it.",
            "REDUCE UNNECESSARY WRAPPERS: Simplify the DOM when possible. Keep structure clean but do not remove elements that are semantically or visually important.",
            "REUSE UI PATTERNS: When similar UI fragments repeat, prefer component extraction. Avoid excessive duplication of markup.",
            "TAILWIND SCALES PREFERRED: Use Tailwind's spacing, color, font, and radius scales when reasonable. Arbitrary values are allowed when needed for closer visual match.",
            "CLEAN CLASS LISTS: Keep Tailwind classes organized and avoid redundant utilities, but do not over-optimize at the cost of clarity.",
            "BALANCED FIDELITY: Match the design visually, but when pixel precision conflicts with readability or maintainability, choose the simpler and clearer implementation."
        ]
    
    return json.dumps(base_prompt, ensure_ascii=False)


def build_user_contents(
    data_dir: Path,
    input_type: str,
    use_small_img: bool = False,
    ablation_func: Optional[Callable] = None
) -> List:
    """
    Build user message contents for the LLM.
    
    Args:
        data_dir: Directory containing sample data
        input_type: One of "figma", "figma_image", "figma_url"
        use_small_img: Whether to use compressed images (for Claude API when input_type is INPUT_FIGMA_URL)
        ablation_func: Optional function to apply ablation to Figma JSON
    
    Returns:
        List of text strings and/or PIL Images for user message
    """
    if input_type not in [INPUT_FIGMA, INPUT_FIGMA_IMAGE, INPUT_FIGMA_URL]:
        raise ValueError(f"Invalid input_type: {input_type}")
    
    data_dir = Path(data_dir)
    page_image = data_dir / "root.png"
    figma_json_path = data_dir / "processed_metadata.json"
    
    if not figma_json_path.exists():
        raise InputValidationError(f"Figma JSON not found: {figma_json_path}")
    
    texts_imgs = []
    
    # Overview
    overview = {
        "input_overview": {
            "description": "The following parts are provided in this exact order.",
            "order": [
                {"part_1": "figma_json", "type": "json", "purpose": "Structured design data from Figma"},
            ],
            # "notes": [
            #     "All asset paths are relative to the current HTML file location.",
            #     "Paths may include URL-encoding for unsafe characters (e.g., spaces -> %20, colon -> %3A). Do not alter them.",
            #     "Ignore fields like: description / desc / comment / comments."
            # ]
        }
    }
    
    if input_type in [INPUT_FIGMA_IMAGE, INPUT_FIGMA_URL]:
        overview["input_overview"]["order"].append(
            {"part_2": "page_screenshot", "type": "image", "purpose": "Visual reference for the layout and content"}
        )
    
    texts_imgs.append(json.dumps(overview, ensure_ascii=False))
    
    # Load and process Figma JSON
    with open(figma_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data = remove_empty(data)
    
    # Apply ablation if provided
    if ablation_func:
        data = ablation_func(data)
        # with open(figma_json_path.with_name(f"{ablation_func.__name__}.json"), "w", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Compact JSON string (remove extra spaces and newlines to save tokens)
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    
    texts_imgs.append("## PART 1: FIGMA JSON (entire file, plain JSON)")
    texts_imgs.append(json_str)
    
    # Add page screenshot if requested
    if input_type == INPUT_FIGMA_IMAGE:
        img = load_image(page_image)
        if img:
            texts_imgs.append("## PART 2: PAGE SCREENSHOT")
            texts_imgs.append(img)
        else:
            logger.warning(f"Page image not found: {page_image}")
    elif input_type == INPUT_FIGMA_URL:
        key = data_dir.name
        texts_imgs.append("## PART 2: PAGE SCREENSHOT")
        texts_imgs.append(get_root_url(key, use_small_img))
    
    return texts_imgs


class DirectGeneration(BaseGeneration):
    """
    Direct code generation from Figma JSON with optional image reference.
    
    Supports three input modes:
    - figma: Figma JSON only
    - figma_image: Figma JSON + local image
    - figma_url: Figma JSON + hosted image URL
    """
    
    def __init__(
        self,
        input_type: str = INPUT_FIGMA_URL,
        use_small_img: bool = False,
        include_guidelines: bool = False
    ):
        """
        Initialize direct generation.
        
        Args:
            input_type: One of "figma", "figma_image", "figma_url"
            use_small_img: Use compressed images for Claude compatibility
            include_guidelines: Include generation guidelines in prompt (for exp2 rebuttal)
        """
        if input_type not in [INPUT_FIGMA, INPUT_FIGMA_IMAGE, INPUT_FIGMA_URL]:
            raise ValueError(f"Invalid input_type: {input_type}")
        
        self.input_type = input_type
        self.use_small_img = use_small_img
        self.include_guidelines = include_guidelines
    
    @property
    def method_name(self) -> str:
        return f"figma_direct" if self.input_type == INPUT_FIGMA else f"figma_image_direct"
    
    def generate(
        self,
        data_dir: Union[str, Path],
        backbone: BaseLLM,
        ablation_func: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Generate HTML code from Figma data.
        
        Args:
            data_dir: Directory containing input data
            backbone: LLM backend for generation
            ablation_func: Optional ablation function to apply to Figma JSON
            **kwargs: Additional parameters (ignored)
        
        Returns:
            Generated HTML string
        
        Raises:
            OutputParsingError: If model output cannot be parsed
            InputValidationError: If input data is invalid
        """
        data_dir = Path(data_dir)
        
        system_prompt = build_system_prompt(self.input_type, self.include_guidelines)
        user_contents = build_user_contents(
            data_dir, 
            self.input_type, 
            self.use_small_img,
            ablation_func
        )
        
        raw = backbone(prompt=system_prompt, texts_imgs=user_contents)
        
        try:
            result = parse_output(raw)
        except Exception as e:
            raise OutputParsingError(f"Failed to parse model output: {e}") from e
        
        if isinstance(result, dict) and "html" in result:
            return result["html"]
        
        raise OutputParsingError(f"Unexpected output format: {type(result)}")


def direct_generation(
    data_dir: Union[str, Path],
    backbone: BaseLLM,
    input_type: str = INPUT_FIGMA_URL,
    use_small_img: bool = False,
    ablation_func: Optional[Callable] = None,
    include_guidelines: bool = False
) -> str:
    """
    Convenience function for direct generation.
    
    Args:
        data_dir: Directory containing input data
        backbone: LLM backend for generation
        input_type: One of "figma", "figma_image", "figma_url"
        use_small_img: Use compressed images for Claude compatibility
        ablation_func: Optional ablation function
        include_guidelines: Include generation guidelines in prompt (for exp2 rebuttal)
    
    Returns:
        Generated HTML string
    """
    generator = DirectGeneration(
        input_type=input_type,
        use_small_img=use_small_img,
        include_guidelines=include_guidelines
    )
    return generator.generate(data_dir, backbone, ablation_func=ablation_func)
