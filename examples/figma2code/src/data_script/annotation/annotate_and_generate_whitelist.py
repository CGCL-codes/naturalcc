"""
Annotate images and generate whitelist from split final annotations.

This script:
1. Reads final_annotation.json from each split directory to identify kept pages (keep='yes')
2. Annotates those pages with theme, language, content, and description using LLM
3. Generates whitelist_filter.csv and selected_filter directory
"""

import json
import csv
from pathlib import Path
from typing import Dict, Set, Any, List

from ...utils.console_logger import logger
from .llm_images_annotator import annotate_images

# Constants
ANNOTATION_FILE = "final_annotation.json"
API_ANNOTATION_FILE = "annotations.json"
API_ATTRIBUTES_KEEP = ["theme", "language", "content", "description"]

ANNOTATION_PROMPT = """You are a professional UI image analyst. Analyze the provided UI image and output a strict JSON annotation. 
You must return a single, complete JSON object and nothing else. Do not use markdown, explanations, or extra text.

Fields (strictly fill all fields):
- theme: choose one from ["light", "dark", "unknown"]Strictly choose from these values for theme: "light", "dark", or "unknown".
Never use "mix" or other variants.
- language: choose one from ["zh", "en", "mix", "unknown"]
- content: choose or create a functional category. Reference: Login/Auth, Dashboard, List/Feed, Detail, View, Form/Input, Settings, Profile, Ecommerce, LandingPage, Onboarding, Search, Modal/Dialog, Notification/Alert, Error/EmptyState, Navigation/TabBar, Checkout, Payment, ContentEditor, Unknown
- description: concise description of UI elements, layout, and function

Instructions:
1. Examine the layout, text, colors, and design carefully.
2. Make sure every field is present and has a valid value.
3. For content, if none of the reference categories fit well, create an appropriate one.
4. Ensure the JSON is fully parsable. Do not include "null" or leave fields empty.

Example JSON:
{
  "theme": "light",
  "language": "en",
  "content": "Messaging",
  "description": "A mobile UI for a messaging app, showing a list of recent chats with profile pictures and last message snippets."
}

{
  "theme": "dark",
  "language": "en",
  "content": "Dashboard",
  "description": "A desktop admin dashboard showing analytics charts, side navigation, and summary cards in dark mode."
}
"""


def get_split_dirs(split_set_dir: Path) -> List[Path]:
    """
    Get all split directories in the split set directory.
    """
    split_dirs = []
    for split_path in split_set_dir.iterdir():
        if split_path.is_dir() and split_path.name.startswith("split_"):
            split_dirs.append(split_path)
    return split_dirs

def collect_kept_pages_from_splits(
    split_dirs: List[Path],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Collect kept pages from all split directories.
    
    Args:
        split_dirs: list of split directories
    
    Returns:
        Dictionary mapping filekey to page_id to page annotation (without 'keep' field)
    """
    all_annotation: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for split_path in split_dirs:
        annotation_path = split_path / ANNOTATION_FILE
        if not annotation_path.exists():
            logger.warning(f"final_annotation.json not found in {split_path}, skipping")
            continue
        
        logger.info(f"Processing {split_path.name}")
        
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {annotation_path}: {e}")
            continue
        
        keep_filekeys = []
        
        for filekey, filekey_annotation in annotation.items():
            filekey_pages = {}
            for page_id, page_annotation in filekey_annotation.items():
                if page_annotation.get("keep") == "yes":
                    # Create a copy without 'keep' field
                    page_data = {k: v for k, v in page_annotation.items() if k != "keep"}
                    filekey_pages[page_id] = page_data
            
            if filekey_pages:
                if filekey not in all_annotation:
                    all_annotation[filekey] = filekey_pages
                else:
                    all_annotation[filekey].update(filekey_pages)
                keep_filekeys.append(filekey)
        
        logger.info(f"  Found {len(keep_filekeys)} filekeys with kept pages")
    
    total_pages = sum(len(pages) for pages in all_annotation.values())
    logger.info(f"Total: {len(all_annotation)} filekeys, {total_pages} pages to annotate")
    
    return all_annotation


def annotate_kept_pages(
    split_dirs: List[Path],
    force: bool = False,
) -> None:
    """
    Annotate kept pages in each split directory.
    
    Args:
        split_dirs: list of split directories
        force: If True, re-annotate all images
    """
    for split_path in split_dirs:
        candidate_dir = split_path / "candidate"
        if not candidate_dir.exists():
            logger.warning(f"candidate directory not found in {split_path}, skipping")
            continue
        
        logger.info(f"Annotating images in {split_path.name}")
        
        # Build selected_pages dict for this split
        selected_pages:Dict[str, Set[str]] = {}
        annotation_path = split_path / ANNOTATION_FILE
        if annotation_path.exists():
            try:
                with open(annotation_path, "r", encoding="utf-8") as f:
                    split_annotation = json.load(f)
                
                for filekey, filekey_annotation in split_annotation.items():
                    if filekey not in selected_pages:
                        selected_pages[filekey] = set()
                    for page_id, page_annotation in filekey_annotation.items():
                        if page_annotation.get("keep") == "yes":
                            selected_pages[filekey].add(page_id)
            except Exception as e:
                logger.error(f"Failed to load {annotation_path}: {e}")
                continue
        
        if not selected_pages:
            logger.info(f"  No pages to annotate in {split_path.name}")
            continue
        
        # Annotate images in this split
        annotate_images(
            base_dir=candidate_dir,
            prompt=ANNOTATION_PROMPT,
            output_file=API_ANNOTATION_FILE,
            force=force,
            selected_pages=selected_pages,
        )


def generate_whitelist(
    split_set_dir: Path,
    kept_pages: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """
    Generate whitelist CSV and selected_filter directory.
    
    Args:
        split_set_dir: split set directory
        kept_pages: Dictionary mapping filekey to page_id to page annotation
    """
    
    split_dirs = get_split_dirs(split_set_dir)
    
    # Merge annotations with API attributes
    all_annotation = {}
    
    for split_path in split_dirs:
        candidate_dir = split_path / "candidate"
        if not candidate_dir.exists():
            continue
        
        for filekey in kept_pages.keys():
            api_annotation_path = candidate_dir / filekey / API_ANNOTATION_FILE
            if not api_annotation_path.exists():
                logger.warning(f"annotations.json not found for {filekey} in {split_path.name}")
                continue
            
            try:
                with open(api_annotation_path, "r", encoding="utf-8") as f:
                    api_annotation = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load {api_annotation_path}: {e}")
                continue
            
            # Update kept_pages with API attributes
            try:
                for page_id, page_annotation in kept_pages[filekey].items():
                    if page_id in api_annotation:
                        api_data = api_annotation[page_id]
                        # Only update if not already present or if we want to overwrite
                        if filekey not in all_annotation:
                            all_annotation[filekey] = {}
                        if page_id not in all_annotation[filekey]:
                            all_annotation[filekey][page_id] = page_annotation.copy()
                        
                        # Add API attributes
                        for api_attribute in API_ATTRIBUTES_KEEP:
                            if api_attribute in api_data:
                                all_annotation[filekey][page_id][api_attribute] = api_data[api_attribute]
            except Exception as e:
                logger.error(f"Error processing {filekey} in {split_path.name}: {e}")
                continue
    
    # Write whitelist CSV
    whitelist_path = split_set_dir / "whitelist_filter.csv"
    logger.info(f"Writing whitelist to {whitelist_path}")
    
    with open(whitelist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        header = ["file_key", "node_id", "platform", "complexity", "quality_rating"] + API_ATTRIBUTES_KEEP
        writer.writerow(header)
        
        # Write data rows
        for filekey, annotation in all_annotation.items():
            for page_id, page_annotation in annotation.items():
                row = [
                    filekey,
                    page_id,
                    page_annotation.get("platform", ""),
                    page_annotation.get("complexity", ""),
                    page_annotation.get("quality_rating", ""),
                ]
                # Add API attribute fields
                for api_attribute in API_ATTRIBUTES_KEEP:
                    row.append(page_annotation.get(api_attribute, ""))
                writer.writerow(row)
    
    logger.info(f"Whitelist written: {len(all_annotation)} filekeys, {sum(len(pages) for pages in all_annotation.values())} pages")


def run_annoate_and_generate_whitelist(
    split_set_dir: Path,
    force: bool = False,
) -> None:
    """
    annotate kept pages and generate whitelist.
    
    Args:
        split_set_dir: split_set directory
        force: If True, re-annotate all images
    """
    if not split_set_dir.exists():
        raise ValueError(f"Split set directory does not exist: {split_set_dir}")
    split_dirs = get_split_dirs(split_set_dir)
    logger.info(f"Found {len(split_dirs)} split directories")
    
    logger.info("=" * 60)
    logger.info("Step 1: Collecting kept pages from split directories")
    logger.info("=" * 60)
    kept_pages = collect_kept_pages_from_splits(split_dirs)
    
    if not kept_pages:
        logger.warning("No kept pages found, exiting")
        return
    
    logger.info("=" * 60)
    logger.info("Step 2: Annotating kept pages with LLM")
    logger.info("=" * 60)
    annotate_kept_pages(
        split_dirs=split_dirs,
        kept_pages=kept_pages,
        force=force,
    )
    
    logger.info("=" * 60)
    logger.info("Step 3: Generating whitelist and selected_filter directory")
    logger.info("=" * 60)
    generate_whitelist(
        split_set_dir=split_set_dir,
        kept_pages=kept_pages,
    )
    
    logger.info("=" * 60)
    logger.info("All steps completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    from ...configs.paths import enter_project_root, OUTPUT_DIR
    from ...utils.console_logger import setup_logging
    enter_project_root()
    setup_logging(logger, log_name="annotate_and_generate_whitelist")
    
    run_annoate_and_generate_whitelist(OUTPUT_DIR / "page_filter" / "split_set")
