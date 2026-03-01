"""
Critic agent for code quality evaluation.

The critic analyzes generated HTML against a reference design
and provides structured feedback for improvement.
"""

import json
from typing import List, Dict, Any, Optional

from PIL import Image

from ...utils.image import get_root_url
from ...llm.base import BaseLLM
from ...utils.console_logger import logger


# Metric definitions for the critic
METRIC_DEFINITIONS = {
    'visual': """1.  **`visual`**:
    *   **Issue**: Identify minor visual discrepancies.
    *   **Suggestion**: Provide a **general, high-level description** of the problem and its location. Do not give specific code advice.
    *   *Example*: "The color of the main title seems slightly too dark compared to the screenshot."
""",
    'STS': """2.  **`semantic_tags` (STS)**:
    *   **Issue**: A generic `<div>` or `<span>` is used where a more semantic tag (`<nav>`, `<header>`, etc.) would be appropriate and visually identical.
    *   **Suggestion**: Recommend replacing the generic tag with a specific semantic tag. Only suggest this for tags that render as block-level elements by default, like `<nav>`, `<header>`, `<footer>`, `<main>`, `<section>`.
""",
    'BC': """3.  **`breakpoint_coverage` (BC)**:
    *   **Issue**: The layout is not responsive.
    *   **Suggestion**: Suggest **adding** responsive prefixes (e.g., `md:`, `lg:`) to existing classes to improve adaptability, ensuring the desktop view remains unchanged.
""",
    'ISR': """4.  **`inline_styles` (ISR)**:
    *   **Issue**: An element uses an inline `style="..."` attribute.
    *   **Suggestion**: Recommend converting the inline styles into equivalent Tailwind CSS utility classes.
""",
    'CCR': """5.  **`class_reuse` (CCR)**:
    *   **Issue**: The same long list of utility classes is repeated on multiple elements.
    *   **Suggestion**: Recommend extracting the repeated classes into a single, reusable custom class using `@apply`.
""",
    'RUR': """6.  **`redundant_unit_replacement` (RUR)**:
    *   **Issue**: An element has a redundant utility class (e.g., `w-full` on a `div`).
    *   **Suggestion**: Recommend removing the redundant class.
""",
    'FU': """7.  **`flexbox_usage` (FU)**:
    *   **Issue**: A layout could be simplified or made more robust using Flexbox properties.
    *   **Suggestion**: Recommend adding `flex` and related properties for simple alignment cases where it is clearly beneficial and safe.
""",
    'AVUR': """8.  **`absolute_value_unit_replacement` (AVUR)**:
    *   **Issue**: An arbitrary value (e.g., `w-[16px]`) is used where a standard Tailwind spacing unit would be an **exact** equivalent.
    *   **Suggestion**: Recommend replacing the arbitrary value with its corresponding Tailwind theme value (e.g., `w-4`).
"""
}


def build_critic_prompt(
    html_string: str,
    design_img_key: str,
    pred_img: Image.Image,
    metrics: Optional[List[str]] = None
) -> tuple:
    """
    Build the prompt for the critic agent.
    
    Args:
        html_string: The HTML string to be critiqued
        design_img_key: The key for the reference design image
        pred_img: Screenshot of the rendered HTML
        metrics: List of metric keys to include (if None, all metrics are used)
    
    Returns:
        Tuple of (system_prompt, user_content list)
    """
    if not metrics:
        metrics_to_include = list(METRIC_DEFINITIONS.keys())
    else:
        metrics_to_include = metrics
    
    critique_rules = "\n".join(
        METRIC_DEFINITIONS[key] 
        for key in metrics_to_include 
        if key in METRIC_DEFINITIONS
    )
    
    system_prompt = f"""
You are an expert web developer and a meticulous quality assurance engineer, with a specialization in Tailwind CSS.
Your task is to critique a given HTML code by comparing its rendered screenshot against a reference design screenshot. Your goal is to identify the most significant discrepancies first.

Focus on high-impact visual differences. Please prioritize your critiques in the following order:
1.  **Background Color Mismatches**: An element's background color is clearly different from the design.
2.  **Layout & Alignment Issues**: Problems with Flexbox or Grid alignment (e.g., `justify-` or `items-` properties) that cause incorrect positioning of elements within a container.
3.  **Major Sizing Discrepancies**: An element's width or height is visibly wrong compared to the design.
4.  **Text-related issues**: font size, color, weight is wrong.
5.  **Spacing issues**: padding, margin, or gap is wrong.

You should provide a maximum of **three** of the most important and high-confidence suggestions. It is better to provide fewer, high-quality critiques or even an empty list than to provide uncertain ones. If you have no high-confidence suggestions, return a JSON object with an empty list for the "critique" key.

Provide your critique in a structured JSON format. The JSON must have a single key "critique", which is a list of objects.

You are allowed to critique the following categories, but **only when you are highly confident the change will NOT alter the visual layout**:

{critique_rules}

**Strict Prohibitions:**
*   **DO NOT** suggest replacing `position: absolute`.
*   **DO NOT** suggest converting `px` units to `rem` in a general sense; only use the AVUR rule for exact matches.
*   **DO NOT** critique any other category not listed above.

**Example of a valid response:**
```json
{{
  "critique": [
    {{
      "type": "inline_styles",
      "element": "The paragraph with text 'Hello'",
      "issue": "The element uses an inline style `font-size: 14px;`.",
      "suggestion": "Replace the inline style with the Tailwind class `text-sm`."
    }},
    {{
      "type": "absolute_value_unit_replacement",
      "element": "The avatar image `<img>`",
      "issue": "The image uses `w-[48px]` and `h-[48px]` which corresponds to a standard Tailwind size.",
      "suggestion": "Replace `w-[48px]` with `w-12` and `h-[48px]` with `h-12`."
    }}
  ]
}}
```
"""
    
    user_content = []
    user_content.append(f"""
Here is the HTML code to critique, which uses Tailwind CSS. Please critique it against the provided screenshot, following all the guidelines.
```html
{html_string}
```
"""
    )
    user_content.append("Here is the design screenshot for reference:")
    user_content.append(get_root_url(design_img_key))
    
    user_content.append("Here is the rendered screenshot of the provided HTML:")
    user_content.append(pred_img)
    
    return system_prompt, user_content


def parse_critic_output(text: str) -> str:
    """
    Parse the critic's output to extract JSON.
    
    Args:
        text: Raw model output
    
    Returns:
        Extracted JSON string
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove markdown code fences
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Find JSON boundaries
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index:end_index + 1]
    
    logger.warning("Could not find valid JSON in critic output")
    return ""


def run_critic(
    backbone: BaseLLM,
    html_string: str,
    image_key: str,
    pred_image: Image.Image,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run the critic agent to get feedback on HTML code.
    
    Args:
        backbone: LLM backend
        html_string: The HTML code to critique
        image_key: Key for the reference design image
        pred_image: Screenshot of the rendered HTML
        metrics: List of metric keys to include
    
    Returns:
        Dictionary containing the critique
    """
    system_prompt, user_content = build_critic_prompt(
        html_string, image_key, pred_image, metrics
    )
    
    raw_output = backbone(system_prompt, user_content)
    critique_json_string = parse_critic_output(raw_output)
    
    try:
        critique = json.loads(critique_json_string)
        return critique
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON response from critic model",
            "raw_response": critique_json_string
        }
