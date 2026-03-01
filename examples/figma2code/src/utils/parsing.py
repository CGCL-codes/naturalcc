"""
Output parsing utilities for Figma2Code.

Provides functions for parsing model outputs (JSON, HTML) and path handling.
"""

import re
import json
from typing import Any, Dict, Optional

def parse_output(raw: str) -> Dict[str, Any]:
    """
    Parse model output and extract JSON or HTML content.
    
    Handles the following formats:
    - Direct JSON
    - JSON wrapped in ```json ... ```
    - Extra text around JSON { ... }
    - HTML (```html ... ``` or raw HTML starting with <!DOCTYPE or <html)
    
    Args:
        raw: Raw model output string
    
    Returns:
        Dictionary with parsed content:
        - For HTML: {"type": "html", "html": "<html>..."}
        - For JSON: The parsed JSON object
    
    Raises:
        ValueError: If unable to parse content
    """
    text = raw.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|html)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    
    # Check for HTML content
    text_lower = text.lower()
    if text_lower.startswith("<!doctype html") or text_lower.startswith("<html"):
        # Extract complete <html>...</html>
        match = re.search(r"(?is)<html.*?</html>", text)
        if match:
            return {"type": "html", "html": match.group(0).strip()}
        return {"type": "html", "html": text}

    # Try to extract and parse JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            raise ValueError(
                f"Found JSON-like block but failed to parse:\n{candidate[:200]}..."
            )

    raise ValueError(f"Unable to parse JSON or HTML from output:\n{text[:200]}...")


def extract_html(raw: str) -> Optional[str]:
    """
    Extract HTML content from model output.
    
    Args:
        raw: Raw model output string
    
    Returns:
        HTML string or None if not found
    """
    try:
        result = parse_output(raw)
        if isinstance(result, dict):
            if result.get("type") == "html":
                return result.get("html")
            elif "html" in result:
                return result["html"]
        return None
    except ValueError:
        return None


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON content from model output.
    
    Args:
        raw: Raw model output string
    
    Returns:
        Parsed JSON dict or None if not found/parseable
    """
    try:
        result = parse_output(raw)
        if isinstance(result, dict) and result.get("type") != "html":
            return result
        return None
    except ValueError:
        return None


def remove_empty(obj: Any) -> Any:
    """
    Recursively remove empty dicts and lists from a JSON-like structure.
    
    Args:
        obj: JSON-like object (dict, list, or primitive)
    
    Returns:
        Cleaned object with empty containers removed
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            cv = remove_empty(v)
            # Skip empty dicts/lists
            if not (isinstance(cv, (dict, list)) and len(cv) == 0):
                cleaned[k] = cv
        return cleaned
    elif isinstance(obj, list):
        cleaned_list = []
        for v in obj:
            cv = remove_empty(v)
            if not (isinstance(cv, (dict, list)) and len(cv) == 0):
                cleaned_list.append(cv)
        return cleaned_list
    else:
        return obj
