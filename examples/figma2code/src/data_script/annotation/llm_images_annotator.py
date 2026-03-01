"""
Image annotation using LLMs.

Annotates Figma design images with json format metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from ...llm import OpenRouterLLM
from ...utils.console_logger import logger

def annotate_single_image(
    image_path: Path,
    prompt: str,
    model: str = "openai/gpt-4o-mini",
    llm: Optional[OpenRouterLLM] = None
) -> Dict[str, Any]:
    """
    Annotate a single image using an LLM.
    
    Args:
        image_path: Path to image file
        model: LLM model identifier
        llm: Optional LLM instance (creates one if not provided)
    
    Returns:
        Annotation dictionary (may contain "error" key on failure)
    """
    try:
        if llm is None:
            llm = OpenRouterLLM(model=model)
        
        image = Image.open(image_path)
        response = llm(prompt, texts_imgs=[image])
        
        # Try to parse as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response}
            
    except Exception as e:
        logger.warning(f"Failed to annotate {image_path}: {e}")
        return {"error": str(e)}


def annotate_images(
    base_dir: Path,
    prompt: str,
    output_file: str = "annotations.json",
    model: str = "openai/gpt-4o-mini",
    max_workers: int = 1,
    force: bool = False,
    selected_pages: Optional[Dict[str, set]] = None,
) -> None:
    """
    Annotate all images in a directory structure.
    
    Expects structure:
        base_dir/
            filekey1/
                page1.png
                page2.png
            filekey2/
                ...
    
    Args:
        base_dir: Base directory containing filekey subdirectories
        prompt: Prompt for LLM annotation
        output_file: Output filename for annotations (per filekey)
        model: LLM model identifier
        max_workers: Number of parallel workers
        force: If True, re-annotate all images
        selected_pages: Optional dict {filekey: set of page_ids} to only annotate specified pages
    """
    filekey_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(filekey_dirs)} filekey directories")
    
    total_images, total_done = 0, 0
    
    llm = OpenRouterLLM(model=model)
    
    with tqdm(total=len(filekey_dirs), desc="Processing filekeys", unit="filekey") as filekey_pbar:
        for fk_dir in filekey_dirs:
            filekey = fk_dir.name
            out_path = fk_dir / output_file
            
            # Load existing annotations
            annotations = {}
            if out_path.exists():
                try:
                    annotations = json.loads(out_path.read_text(encoding="utf-8"))
                except Exception:
                    annotations = {}
            
            # Find images to process
            img_paths = sorted(fk_dir.glob("*.png"))
            if not img_paths:
                continue
            
            # Filter by selected_pages if provided
            if selected_pages is not None:
                if filekey not in selected_pages:
                    logger.debug(f"Skipping {filekey} (not in selected_pages)")
                    filekey_pbar.update(1)
                    continue
                selected_set = selected_pages[filekey]
                # 图片文件名使用 '_' 和 '-' 格式，selected_set 中已包含文件名格式
                img_paths = [p for p in img_paths if p.stem in selected_set]
                if not img_paths:
                    logger.debug(f"No selected images found in {filekey}")
                    filekey_pbar.update(1)
                    continue
            
            if force:
                # Re-annotate all
                to_process = img_paths
            else:
                # Only unannotated or errored
                to_process = [
                    p for p in img_paths
                    if p.stem not in annotations or "error" in annotations.get(p.stem, {})
                ]

            if not to_process:
                logger.debug(f"All images in {fk_dir.name} already annotated")
                continue

            done_count = 0 if force else sum(1 for v in annotations.values() if "error" not in v)
            total_count = len(to_process) + done_count
            
            logger.info(f"Processing {len(to_process)} images in {fk_dir.name}")
            
            # Process images
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(annotate_single_image, p, prompt, model, llm) : p for p in to_process}

                with tqdm(total=len(futures), desc=f"Annotating images for {fk_dir.name}", unit="image") as pbar:
                    for future in as_completed(futures):
                        try:
                            page_id = futures[future].stem
                            result = future.result()
                        except Exception as e:
                            logger.error(f"Failed to annotate {futures[future]}: {e}")
                            continue
                        annotations[page_id] = result
                        done_count += 1
                        pbar.update(1)
                        
                        # Save after each annotation
                        out_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
                        
            total_images += total_count
            total_done += done_count
            logger.info(f"Saved annotations to {out_path}")

            filekey_pbar.update(1)
    
    if total_images > 0:
        logger.info(f"{total_done}/{total_images} images annotated")