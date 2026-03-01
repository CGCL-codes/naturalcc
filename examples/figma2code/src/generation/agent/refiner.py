"""
Refiner agent for HTML code improvement.

The refiner takes HTML code, a reference design, and critique feedback
to produce an improved version of the HTML.
"""

import json
from typing import Dict, Any

from ...utils.image import get_root_url
from ...utils.console_logger import logger
from ...llm.base import BaseLLM


def build_refiner_prompt(
    html_string: str,
    critique: Dict[str, Any],
    ref_image_key: str
) -> tuple:
    """
    Build the prompt for the refiner agent.
    
    Args:
        html_string: The original HTML string
        critique: The critique from the critic agent
        ref_image_key: Key for the reference design image
    
    Returns:
        Tuple of (system_prompt, user_content list)
    """
    system_prompt = """
You are an expert web developer specializing in fixing and refining HTML with Tailwind CSS.
Your task is to take an existing HTML file, a design screenshot, and a list of critiques, and then produce a new, improved version of the HTML code that addresses all the points in the critique.

**Instructions:**
1.  **Preserve Visual Fidelity**: This is the most important rule. The final output must be a pixel-perfect match to the design screenshot on a standard desktop view.
2.  **Implement Structural Improvements**: Carefully implement every suggestion from the critique to improve responsiveness and maintainability.
3.  **Adhere to Tailwind CSS**: Your entire output must be a single block of HTML code that uses Tailwind CSS classes.
4.  **Replace, Don't Just Add**: When a critique points out an issue (e.g., absolute positioning), you must replace the problematic code with a better implementation (e.g., flexbox) that achieves the same visual result.
5.  **Output Only Code**: Do not include any explanations, apologies, or any text outside of the final, complete HTML code.

**Priorities for Refinement:**
- **Visuals**: For visual critiques, make **conservative, minimal changes** to address the general feedback. The goal is to fix subtle inaccuracies without accidentally breaking the layout. For example, if the critique says "the button color is slightly off," make a small adjustment to the color class rather than rewriting the entire button.
- **Responsiveness**: Aggressively replace absolute positioning and fixed sizes with responsive layouts, ensuring the desktop view remains unchanged.
- **Maintainability**: Convert `div` and `span` tags to semantic HTML elements and replace arbitrary values, all while maintaining the original appearance.
"""
    
    critique_str = json.dumps(critique, indent=2)
    
    user_content = []
    user_content.append(f"""
Please refine the following HTML code based on the provided image and the critique below.

**Original HTML:**
```html
{html_string}
```

**Critique:**
```json
{critique_str}
```

Your task is to rewrite the HTML code to fix all the issues mentioned in the critique, following all the instructions and priorities. Remember to only output the full, corrected HTML code.
"""
    )
    user_content.append("Here is the design screenshot for reference:")
    user_content.append(get_root_url(ref_image_key))
    
    return system_prompt, user_content


def parse_refiner_output(text: str) -> str:
    """
    Parse the refiner's output to extract HTML.
    
    Args:
        text: Raw model output
    
    Returns:
        Extracted HTML string
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove markdown code fences
    if text.startswith("```html"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Find HTML boundaries
    start_index = text.find('<')
    end_index = text.rfind('>')
    
    logger.debug(
        f"parse_refiner_output: raw length={len(text)}, "
        f"matched length={(end_index - start_index + 1) if start_index != -1 and end_index != -1 and start_index < end_index else 0}"
    )
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index:end_index + 1]
    
    logger.warning("Could not find valid HTML tags in refiner output")
    return ""


def run_refiner(
    backbone: BaseLLM,
    html_string: str,
    ref_image_key: str,
    critique: Dict[str, Any]
) -> str:
    """
    Run the refiner agent to improve HTML code.
    
    Args:
        backbone: LLM backend
        html_string: The original HTML code
        ref_image_key: Key for the reference design image
        critique: The critique from the critic agent
    
    Returns:
        The refined HTML string
    """
    system_prompt, user_content = build_refiner_prompt(
        html_string, critique, ref_image_key
    )
    
    refined_html = backbone(system_prompt, user_content)
    
    return parse_refiner_output(refined_html)
