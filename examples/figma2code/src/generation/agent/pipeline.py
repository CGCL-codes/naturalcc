"""
Agent-based code generation pipeline.

Implements a critic-refiner pipeline for iterative HTML improvement,
combining rule-based initial generation with LLM-based refinement.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .critic import run_critic
from .refiner import run_refiner
from .ir_to_tailwind import IRtoTailwind
from .figma_to_ir import FigmatoIR
from ...llm.base import BaseLLM
from ...evaluation.visual import crop_nonwhite_border_pil, resize_to_ref, dinov2, mae
from ...utils.html_screenshot import html2shot
from ...utils.console_logger import logger


def calc_visual_score(ref_img: Image.Image, pred_html_path: Path) -> Dict[str, float]:
    """
    Calculate visual similarity score between reference image and rendered HTML.
    
    Args:
        ref_img: Reference design image
        pred_html_path: Path to the HTML file to render
    
    Returns:
        Dictionary with 'ves' (visual embedding similarity), 'mae', and 'mix' scores
    """
    pred_img = html2shot(
        str(pred_html_path),
        str(pred_html_path.with_suffix(".png")),
        use_viewport=True,
        viewport={'width': ref_img.width, 'height': ref_img.height}
    )
    
    # Align prediction to reference size
    pred_img_aligned = resize_to_ref(pred_img, ref_img.size)
    
    # Convert to numpy arrays
    ref_np = np.array(ref_img.convert("RGB"))
    pred_np = np.array(pred_img.convert("RGB"))
    pred_np_aligned = np.array(pred_img_aligned.convert("RGB"))
    
    # Calculate metrics
    ves = dinov2(ref_np, pred_np, device="cuda")
    mae_px = mae(ref_np, pred_np_aligned)
    
    return {
        "ves": ves,
        "mae": mae_px,
        "mix": (ves - mae_px) / 2.0
    }


def process_ref_image(image_path: Path, data_dir: Path) -> Image.Image:
    """
    Process reference image by resizing to match Figma metadata dimensions.
    
    Args:
        image_path: Path to reference image
        data_dir: Directory containing processed_metadata.json
    
    Returns:
        Processed PIL Image
    """
    ref_img = Image.open(image_path).convert("RGB")
    
    # Resize to match JSON metadata dimensions
    json_path = data_dir / "processed_metadata.json"
    with open(json_path, 'r') as f:
        meta_data = json.load(f)
    bbox = meta_data['document']['absoluteBoundingBox']
    width = int(bbox.get('width', 0))
    height = int(bbox.get('height', 0))
    
    if width > 0 and height > 0:
        ref_img = ref_img.resize((width, height), Image.Resampling.LANCZOS)
    
    # Crop white borders
    ref_img = crop_nonwhite_border_pil(ref_img)
    
    return ref_img


class AgentPipeline:
    """
    Agent-based code generation pipeline.
    
    Combines rule-based IR conversion with LLM-based critic-refiner loop
    for iterative HTML improvement.
    """
    
    def __init__(
        self,
        backbone_critic: BaseLLM,
        backbone_refiner: BaseLLM,
        max_steps: int = 5,
        score_threshold: float = 0.90,
        cleanup_temp: bool = True
    ):
        """
        Initialize the agent pipeline.
        
        Args:
            backbone_critic: LLM for critic role
            backbone_refiner: LLM for refiner role
            max_steps: Maximum refinement iterations
            score_threshold: Minimum score ratio to accept refinement
            cleanup_temp: Whether to delete temporary files
        """
        self.backbone_critic = backbone_critic
        self.backbone_refiner = backbone_refiner
        self.max_steps = max_steps
        self.score_threshold = score_threshold
        self.cleanup_temp = cleanup_temp
    
    def generate(self, data_dir: Path) -> str:
        """
        Generate HTML code using the agent pipeline.
        
        Args:
            data_dir: Directory containing Figma data (processed_metadata.json, root.png, assets/)
        
        Returns:
            Final HTML string
        """
        data_dir = Path(data_dir)
        
        tmp_prefix = "_tmp_"
        metadata_json_path = data_dir / "processed_metadata.json"
        ref_image_path = data_dir / "root.png"
        
        # Process reference image
        ref_img = process_ref_image(ref_image_path, data_dir)
        
        # Paths for intermediate files
        ir_json_path = data_dir / f"{tmp_prefix}ir.json"
        initial_html_path = data_dir / f"{tmp_prefix}rule_conversion.html"
        
        # Step 1: Rule-based initial generation
        logger.info("Running Figma-to-IR conversion...")
        alt_nodes = FigmatoIR(str(metadata_json_path), str(ir_json_path))
        
        logger.info("Running IR-to-Tailwind conversion...")
        initial_html = IRtoTailwind(alt_nodes, str(initial_html_path))
        
        initial_score = calc_visual_score(ref_img, initial_html_path)
        logger.info(f"Initial visual score: {initial_score}")
        
        # Image key for hosted reference
        ref_image_key = data_dir.name
        
        # Refinement loop
        html_to_refine = initial_html
        best_score = None
        
        for step in range(self.max_steps):
            logger.debug(f"Running critic, step {step}...")
            
            try:
                # Save current HTML for screenshot
                current_html_path = data_dir / f"{tmp_prefix}current_to_refine.html"
                with open(current_html_path, "w", encoding="utf-8") as f:
                    f.write(html_to_refine)
                
                # Generate screenshot
                rendered_image = html2shot(
                    str(current_html_path),
                    str(current_html_path.with_suffix(".png")),
                    use_viewport=True,
                    viewport={'width': ref_img.width, 'height': ref_img.height}
                )
                
                # Run critic
                critique_json = run_critic(
                    self.backbone_critic,
                    html_to_refine,
                    ref_image_key,
                    rendered_image,
                    metrics=["visual"]
                )
                
                # Save critique for debugging
                with open(data_dir / f"{tmp_prefix}critique_{step}.json", "w", encoding="utf-8") as f:
                    json.dump(critique_json, f, indent=2, ensure_ascii=False)
                
                if critique_json.get("error"):
                    logger.info(f"Critic returned error: {critique_json['error']}, skipping refinement")
                    continue
                
                if len(critique_json.get("critique", [])) == 0:
                    logger.info(f"Step {step}: No critique found, skipping refinement")
                    continue
                
                # Run refiner
                logger.debug("Running refiner...")
                refined_html = run_refiner(
                    self.backbone_refiner,
                    html_to_refine,
                    ref_image_key,
                    critique_json
                )
                
            except Exception as e:
                logger.error(f"Exception during critic/refiner steps: {e}")
                continue
            
            # Save and evaluate refined HTML
            refined_html_path = data_dir / f"{tmp_prefix}refined_agent_{step}.html"
            with open(refined_html_path, "w", encoding="utf-8") as f:
                f.write(refined_html)
            
            new_score = calc_visual_score(ref_img, refined_html_path)
            logger.info(f"New visual score: {new_score}")
            
            # Check if refinement is acceptable
            if new_score["mix"] <= initial_score["mix"] * self.score_threshold:
                logger.debug(
                    f"Step {step}: Significant decrease in visual score "
                    f"({new_score['mix']:.4f} <= {initial_score['mix'] * self.score_threshold:.4f}), aborting"
                )
            else:
                if not best_score or new_score["mix"] > best_score["mix"]:
                    logger.debug(
                        f"Step {step}: Best visual score improved from "
                        f"{best_score['mix'] if best_score else 0:.4f} to {new_score['mix']:.4f}"
                    )
                    best_score = new_score
                    html_to_refine = refined_html
        
        # Cleanup temporary files
        if self.cleanup_temp:
            logger.info(f"Cleaning up temporary files with prefix '{tmp_prefix}' in {data_dir}...")
            for temp_file in data_dir.glob(f"{tmp_prefix}*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except OSError as e:
                    logger.error(f"Error deleting file {temp_file}: {e}")
        
        return str(html_to_refine)


def agent_generation(
    data_dir: Path,
    backbone_critic: BaseLLM,
    backbone_refiner: BaseLLM,
    max_steps: int = 5
) -> str:
    """
    Convenience function for agent-based generation.
    
    Args:
        data_dir: Directory containing Figma data
        backbone_critic: LLM for critic role
        backbone_refiner: LLM for refiner role
        max_steps: Maximum refinement iterations
    
    Returns:
        Generated HTML string
    """
    pipeline = AgentPipeline(
        backbone_critic=backbone_critic,
        backbone_refiner=backbone_refiner,
        max_steps=max_steps
    )
    return pipeline.generate(data_dir)
