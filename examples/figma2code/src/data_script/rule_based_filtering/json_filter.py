"""
JSON-based page filtering for Figma designs.

Filters pages based on metadata criteria like aspect ratio,
children count, and image coverage.

Output directory structure:
```
output/page_filter/json_filtered/
├── filekey1/
│   └── summary.json
├── filekey2/
│   └── summary.json
└── ...

Each summary.json contains:
  - file_key, retained_pages_count, removed_pages_count
  - retained_node_ids, removed_node_ids_with_reason
```
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from ...utils.console_logger import setup_logging, logger

@dataclass
class FilterConfig:
    """Configuration for JSON-based filtering."""
    aspect_ratio_threshold: float = 5.0  # Max aspect ratio (e.g., 5:1)
    min_children_threshold: int = 2  # Minimum child nodes
    image_area_threshold: float = 0.8  # Max image area ratio (80%)


class JsonFilter:
    """
    Filters Figma pages based on JSON metadata.
    
    Applies rules like aspect ratio limits, minimum children count,
    and maximum single-image coverage.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize JSON filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
    
    def filter_page(
        self,
        page_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Filter a single page.
        
        Args:
            page_data: Page node data
        
        Returns:
            Tuple of (should_retain, rejection_reason)
        """
        if page_data.get("type") != "FRAME":
            return False, "Not a FRAME node"
        
        bbox = page_data.get("absoluteBoundingBox", {})
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        children = page_data.get("children", [])
        
        # Rule 1: Valid dimensions
        if width <= 0 or height <= 0:
            return False, "Invalid dimensions"
        
        # Rule 2: Aspect ratio
        min_dim = min(width, height)
        max_dim = max(width, height)
        if min_dim == 0:
            return False, "Zero dimension"
        
        aspect_ratio = max_dim / min_dim
        if aspect_ratio > self.config.aspect_ratio_threshold:
            return False, f"Aspect ratio ({aspect_ratio:.1f}) exceeds threshold"
        
        # Rule 3: Minimum children
        if len(children) < self.config.min_children_threshold:
            return False, f"Too few children ({len(children)})"
        
        # Rule 4: Single image coverage
        page_area = width * height
        reason = self._check_image_coverage(children, page_area)
        if reason:
            return False, reason
        
        return True, None
    
    def _check_image_coverage(
        self,
        children: List[Dict[str, Any]],
        page_area: float
    ) -> Optional[str]:
        """
        Check if any single image covers too much of the page.
        
        Args:
            children: Child nodes
            page_area: Total page area
        
        Returns:
            Rejection reason if threshold exceeded, None otherwise
        """
        nodes_to_check = list(children)
        
        while nodes_to_check:
            node = nodes_to_check.pop(0)
            
            # Add children to check list
            if "children" in node:
                nodes_to_check.extend(node["children"])
            
            # Check for image fills
            fills = node.get("fills", [])
            if any(fill.get("type") == "IMAGE" for fill in fills):
                node_area = self._get_node_area(node)
                ratio = node_area / page_area
                
                if ratio > self.config.image_area_threshold:
                    return f"Single image covers {ratio*100:.1f}% of page"
        
        return None
    
    def _get_node_area(self, node: Dict[str, Any]) -> float:
        """Get area from node bounding box."""
        bbox = node.get("absoluteBoundingBox", {})
        width = bbox.get("width", 0)
        height = bbox.get("height", 0)
        return width * height
    
    def filter_filekey(
        self,
        filekey_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Filter all pages in a filekey directory.
        
        Args:
            filekey_path: Path to filekey directory
            output_dir: Optional output directory for summary
        
        Returns:
            Summary dictionary
        """
        filekey_name = filekey_path.name
        json_file = filekey_path / f"{filekey_name}.json"
        
        if not json_file.exists():
            logger.warning(f"JSON file not found: {json_file}")
            return {"error": "JSON file not found"}
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        retained = []
        removed = []
        
        canvases = data.get("document", {}).get("children", [])
        for canvas in canvases:
            if canvas.get("type") != "CANVAS":
                continue
            
            pages = canvas.get("children", [])
            for page in pages:            
                page_id = page.get("id")
                should_retain, reason = self.filter_page(page)
                
                if should_retain:
                    retained.append({"node_id": page_id})
                else:
                    removed.append({"node_id": page_id, "reason": reason})
        
        summary = {
            "file_key": filekey_name,
            "retained_pages_count": len(retained),
            "removed_pages_count": len(removed),
            "retained_node_ids": sorted([p["node_id"] for p in retained]),
            "removed_node_ids_with_reason": sorted(removed, key=lambda x: x["node_id"])
        }
        
        # Save summary if output directory provided
        if output_dir:
            output_path = output_dir / filekey_name
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary


def run_json_filter(
    input_dir: str,
    output_dir: str,
    config: Optional[FilterConfig] = None,
) -> Dict[str, Any]:
    """
    Filter all pages in a directory structure.
    
    Args:
        input_dir: Input directory containing filekey subdirectories
        output_dir: Output directory for summaries
        config: Filter configuration
    
    Returns:
        Overall statistics dictionary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    filter_instance = JsonFilter(config)
    
    total_retained = 0
    total_removed = 0
    filekey_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for filekey_dir in tqdm(filekey_dirs, desc="Filtering filekeys"):
        summary = filter_instance.filter_filekey(filekey_dir, output_path)
        total_retained += summary.get("retained_pages_count", 0)
        total_removed += summary.get("removed_pages_count", 0)
    
    return {
        "total_filekeys": len(filekey_dirs),
        "total_retained": total_retained,
        "total_removed": total_removed
    }

if __name__ == "__main__":
    from ...configs.paths import enter_project_root, OUTPUT_DIR
    enter_project_root()
    setup_logging(logger, log_name="json_filter")
    result = run_json_filter(
        input_dir=OUTPUT_DIR / "page_filter" / "candidate",
        output_dir=OUTPUT_DIR / "page_filter" / "json_filtered",
    )
    logger.info(result)