"""
Similarity-based filtering for Figma designs.

Uses CLIP embeddings to detect and deduplicate similar pages.
Supports intra-filekey and inter-filekey filtering.

Output directory structure:
```
output/page_filter/clipsim_filtered/
├── intra_filekey/
│   ├── summary.json
│   ├── filekey1/
│   │   ├── summary.json
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   ├── filekey2/
│   │   └── ...
│   └── ...
├── inter_filekey/
│   ├── summary.json
│   ├── filekey1/
│   │   ├── summary.json
│   │   ├── page_id1.png
│   │   └── ...
│   ├── filekey2/
│   │   └── ...
│   └── ...
├── intra_similar_groups/          (optional, when save_similar_groups=True)
│   ├── filekey1/
│   │   ├── group_0/
│   │   │   ├── summary.json
│   │   │   └── *.png
│   │   └── ...
│   └── ...
└── inter_similar_groups/          (optional, when save_similar_groups=True)
    ├── group_0/
    │   ├── summary.json
    │   └── *.png
    └── ...
```
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from ...utils.console_logger import logger, setup_logging

# Image extensions for filtering
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Lazy load heavy dependencies
_clip_model = None
_clip_preprocess = None


def _load_clip():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    
    if _clip_model is None:
        try:
            import torch
            import clip
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
            _clip_model.eval()
        except ImportError:
            raise ImportError("clip and torch packages required for similarity filtering")
    
    return _clip_model, _clip_preprocess


@dataclass
class SimilarityConfig:
    """Configuration for similarity-based filtering."""
    threshold: float = 0.95  # Cosine similarity threshold
    batch_size: int = 32  # Batch size for embedding computation
    save_similar_groups: bool = True  # Whether to save similar groups to disk (for analysis)


def get_best_image_from_group(image_paths: List[str]) -> Optional[str]:
    """
    From a group of image paths, select the one with the largest file size.

    Args:
        image_paths: List of image file paths

    Returns:
        Path with largest file size, or None if empty
    """
    if not image_paths:
        return None
    return max(image_paths, key=lambda p: os.path.getsize(p))


def save_similar_group(
    group_paths: List[str],
    best_image_path: str,
    base_output_dir: Path,
    group_index: int,
) -> None:
    """
    Save a similar image group to disk (for analysis/debugging).

    Args:
        group_paths: All image paths in the group
        best_image_path: The retained (best) image path
        base_output_dir: Root directory for group folders
        group_index: Group index for folder name group_{index}
    """
    group_dir = base_output_dir / f"group_{group_index}"
    group_dir.mkdir(parents=True, exist_ok=True)
    group_node_ids = []
    for path in group_paths:
        try:
            shutil.copy(path, group_dir / Path(path).name)
            group_node_ids.append(Path(path).stem)
        except Exception as e:
            logger.warning("Copy %s to %s failed: %s", path, group_dir, e)
    retained_node_id = Path(best_image_path).stem
    removed_node_ids = sorted(nid for nid in group_node_ids if nid != retained_node_id)
    summary = {
        "group_index": group_index,
        "total_images_in_group": len(group_paths),
        "retained_node_id": retained_node_id,
        "removed_node_ids": removed_node_ids,
        "all_node_ids_in_group": sorted(group_node_ids),
    }
    try:
        with open(group_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Write summary %s failed: %s", group_dir / "summary.json", e)


def compute_clip_embeddings_batch(
    image_paths: List[str],
    config: Optional[SimilarityConfig] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute CLIP embeddings in batches (faster for many images).

    Args:
        image_paths: List of image file paths
        config: Similarity configuration (batch_size)

    Returns:
        Tuple of (embedding_matrix, valid_indices) where valid_indices are indices into image_paths
    """
    try:
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError("torch and PIL required for batch embedding")
    config = config or SimilarityConfig()
    model, preprocess = _load_clip()
    device = next(model.parameters()).device
    valid_indices: List[int] = []
    batch_tensors: List = []
    batch_indices: List[int] = []
    all_embeddings: List[np.ndarray] = []
    n = len(image_paths)
    for i in range(n):
        path = image_paths[i]
        try:
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image)
            batch_tensors.append(tensor)
            batch_indices.append(i)
        except Exception as e:
            logger.warning("Load image %s failed: %s", path, e)
            continue
        if len(batch_tensors) >= config.batch_size:
            with torch.no_grad():
                stack = torch.stack(batch_tensors).to(device)
                emb = model.encode_image(stack)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu().numpy())
            valid_indices.extend(batch_indices)
            batch_tensors, batch_indices = [], []
    if batch_tensors:
        with torch.no_grad():
            stack = torch.stack(batch_tensors).to(device)
            emb = model.encode_image(stack)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())
        valid_indices.extend(batch_indices)
    if not all_embeddings:
        return np.zeros((0, 0)), []
    return np.vstack(all_embeddings), valid_indices


def _similarity_matrix_from_embeddings(
    embeddings: np.ndarray,
    valid_indices: List[int],
    n_total: int,
) -> np.ndarray:
    """Build full N x N similarity matrix from embeddings of a subset of indices."""
    sim_sub = np.dot(embeddings, embeddings.T)
    full_sim = np.zeros((n_total, n_total))
    for i, vi in enumerate(valid_indices):
        for j, vj in enumerate(valid_indices):
            full_sim[vi, vj] = sim_sub[i, j]
    return full_sim


def group_by_similarity_and_retain_best(
    image_paths: List[str],
    sim_matrix: np.ndarray,
    config: SimilarityConfig,
    save_groups_base_dir: Optional[Path] = None,
    group_counter_start: int = 0,
) -> Tuple[Set[str], int]:
    """
    Group images by similarity threshold and retain the best (by file size) per group.

    Args:
        image_paths: List of image paths (same order as sim_matrix)
        sim_matrix: Full N x N similarity matrix
        config: Similarity configuration (threshold, save_similar_groups)
        save_groups_base_dir: If set and config.save_similar_groups, save group folders here
        group_counter_start: Starting index for group folder names

    Returns:
        Tuple of (retained_paths, next_group_counter)
    """
    n = len(image_paths)
    visited: Set[int] = set()
    retained: Set[str] = set()
    group_counter = group_counter_start
    for i in range(n):
        if i in visited:
            continue
        similar_indices = np.where(sim_matrix[i] >= config.threshold)[0]
        similar_indices = similar_indices.tolist()
        group_paths = [image_paths[idx] for idx in similar_indices]
        best_path = get_best_image_from_group(group_paths)
        if best_path:
            retained.add(best_path)
        if len(group_paths) > 1 and save_groups_base_dir and config.save_similar_groups and best_path:
            save_similar_group(group_paths, best_path, save_groups_base_dir, group_counter)
            group_counter += 1
        for idx in similar_indices:
            visited.add(idx)
    return retained, group_counter


def _collect_image_paths_in_filekey(
    filekey_path: Path,
    retained_node_ids: Optional[Set[str]] = None,
) -> List[str]:
    """Collect image paths under a filekey dir, optionally filtered by node IDs (stem)."""
    all_paths = [
        str(p) for p in filekey_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if retained_node_ids is None:
        return all_paths
    return [p for p in all_paths if Path(p).stem in retained_node_ids]


def filter_intra_filekey(
    filekey_path: Path,
    json_filter_dir: Path,
    output_dir: Path,
    config: SimilarityConfig,
) -> Optional[Dict[str, Any]]:
    """
    Run intra-filekey similarity filtering for one filekey.

    Reads JSON filter summary if present to restrict to retained_node_ids,
    computes CLIP similarity within the filekey, groups by threshold,
    keeps best per group (by file size), copies to output_dir/filekey_name.

    Args:
        filekey_path: Path to filekey directory (candidate)
        json_filter_dir: Directory containing filekey/summary.json from JSON filter
        output_dir: Output base (will write to output_dir/filekey_name)
        config: Similarity configuration

    Returns:
        Summary dict or None if skipped/error
    """
    filekey_name = filekey_path.name
    json_summary_path = json_filter_dir / filekey_name / "summary.json"
    retained_node_ids: Optional[Set[str]] = None
    if json_summary_path.exists():
        try:
            with open(json_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            retained_node_ids = set(summary.get("retained_node_ids", []))
        except Exception as e:
            logger.warning("Read %s failed: %s", json_summary_path, e)

    img_paths = _collect_image_paths_in_filekey(filekey_path, retained_node_ids)
    filekey_output_dir = output_dir / filekey_name
    filekey_output_dir.mkdir(parents=True, exist_ok=True)

    if len(img_paths) < 2:
        for p in img_paths:
            shutil.copy(p, filekey_output_dir / Path(p).name)
        retained_ids = [Path(p).stem for p in img_paths]
        summary_out = {
            "total_images": len(img_paths),
            "retained_images_count": len(retained_ids),
            "removed_images_count": 0,
            "retained_node_ids": retained_ids,
            "removed_node_ids": [],
        }
        with open(filekey_output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2, ensure_ascii=False)
        return summary_out

    # Batch embeddings and similarity
    embeddings, valid_indices = compute_clip_embeddings_batch(img_paths, config)
    if len(valid_indices) == 0:
        logger.warning("No valid embeddings for filekey %s", filekey_name)
        return None
    n = len(img_paths)
    sim_matrix = _similarity_matrix_from_embeddings(embeddings, valid_indices, n)
    # Fill diagonal for indices that had no embedding (avoid grouping with others)
    for i in range(n):
        if i not in valid_indices:
            sim_matrix[i, i] = 1.0

    save_dir = (output_dir / "intra_similar_groups" / filekey_name) if config.save_similar_groups else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    retained_paths, _ = group_by_similarity_and_retain_best(
        img_paths, sim_matrix, config,
        save_groups_base_dir=save_dir,
        group_counter_start=0,
    )
    for p in retained_paths:
        shutil.copy(p, filekey_output_dir / Path(p).name)
    retained_ids = sorted(Path(p).stem for p in retained_paths)
    removed_ids = sorted(Path(p).stem for p in set(img_paths) - retained_paths)
    summary_out = {
        "total_images": len(img_paths),
        "retained_images_count": len(retained_ids),
        "removed_images_count": len(removed_ids),
        "retained_node_ids": retained_ids,
        "removed_node_ids": removed_ids,
    }
    with open(filekey_output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    return summary_out


def run_intra_similarity_filter(
    input_dir: Path,
    json_filter_dir: Path,
    output_base_dir: Path,
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    """
    Run intra-filekey similarity filter for all filekeys under input_dir.

    Writes to output_base_dir/intra_filekey/<filekey_name>/.
    """
    config = config or SimilarityConfig()
    intra_output = output_base_dir / "intra_filekey"
    intra_output.mkdir(parents=True, exist_ok=True)
    filekey_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    total_retained, total_removed = 0, 0
    for filekey_dir in tqdm(filekey_dirs, desc="Intra-filekey similarity filter"):
        summary = filter_intra_filekey(filekey_dir, json_filter_dir, intra_output, config)
        if summary:
            total_retained += summary.get("retained_images_count", 0)
            total_removed += summary.get("removed_images_count", 0)
    global_summary = {
        "total_retained_images": total_retained,
        "total_removed_images": total_removed,
    }
    with open(intra_output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    logger.info(
        "Intra-filekey: retained=%d removed=%d -> %s",
        total_retained, total_removed, intra_output,
    )
    return global_summary


def filter_inter_filekey(
    intra_output_dir: Path,
    output_base_dir: Path,
    config: SimilarityConfig,
) -> Dict[str, Any]:
    """
    Run inter-filekey similarity filtering.

    Collects all images from intra_output_dir, computes global CLIP similarity,
    groups by threshold, keeps best per group, then copies retained images
    back to output_base_dir/inter_filekey/<filekey_name>/.
    """
    all_img_paths = [
        str(p) for p in intra_output_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    path_to_filekey: Dict[str, str] = {p: Path(p).parent.name for p in all_img_paths}
    inter_output = output_base_dir / "inter_filekey"
    groups_dir = output_base_dir / "inter_similar_groups"
    inter_output.mkdir(parents=True, exist_ok=True)

    if len(all_img_paths) < 2:
        for path in all_img_paths:
            fk = path_to_filekey.get(path)
            if fk:
                (inter_output / fk).mkdir(parents=True, exist_ok=True)
                shutil.copy(path, inter_output / fk / Path(path).name)
        global_summary = {
            "total_retained_images": len(all_img_paths),
            "total_removed_images": 0,
        }
        with open(inter_output / "summary.json", "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2, ensure_ascii=False)
        logger.info("Inter-filekey: fewer than 2 images, copied all to %s", inter_output)
        return global_summary

    embeddings, valid_indices = compute_clip_embeddings_batch(all_img_paths, config)
    if len(valid_indices) == 0:
        logger.warning("No valid embeddings for inter-filekey")
        return {}
    n = len(all_img_paths)
    sim_matrix = _similarity_matrix_from_embeddings(embeddings, valid_indices, n)
    for i in range(n):
        if i not in valid_indices:
            sim_matrix[i, i] = 1.0

    save_dir = groups_dir if config.save_similar_groups else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    retained_paths, _ = group_by_similarity_and_retain_best(
        all_img_paths, sim_matrix, config,
        save_groups_base_dir=save_dir,
        group_counter_start=0,
    )
    for path in retained_paths:
        fk = path_to_filekey.get(path)
        if fk:
            dest_dir = inter_output / fk
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest_dir / Path(path).name)

    filekey_original: Dict[str, Set[str]] = {}
    for p in all_img_paths:
        fk = path_to_filekey.get(p)
        if fk:
            filekey_original.setdefault(fk, set()).add(p)
    total_retained_final, total_removed_final = 0, 0
    for filekey_name, original_paths in filekey_original.items():
        retained_in_fk = {p for p in retained_paths if path_to_filekey.get(p) == filekey_name}
        removed_in_fk = original_paths - retained_in_fk
        retained_ids = sorted(Path(p).stem for p in retained_in_fk)
        removed_ids = sorted(Path(p).stem for p in removed_in_fk)
        summary = {
            "total_images_before_inter_filter": len(original_paths),
            "retained_images_count": len(retained_ids),
            "removed_images_count": len(removed_ids),
            "retained_node_ids": retained_ids,
            "removed_node_ids": removed_ids,
        }
        if retained_in_fk:
            fk_dir = inter_output / filekey_name
            fk_dir.mkdir(parents=True, exist_ok=True)
            with open(fk_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        total_retained_final += len(retained_ids)
        total_removed_final += len(removed_ids)
    global_summary = {
        "total_retained_images": total_retained_final,
        "total_removed_images": total_removed_final,
    }
    with open(inter_output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    logger.info(
        "Inter-filekey: retained=%d removed=%d -> %s",
        total_retained_final, total_removed_final, inter_output,
    )
    return global_summary


def run_similarity_filter(
    input_dir: str,
    json_filter_dir: str,
    output_dir: str,
    config: Optional[SimilarityConfig] = None,
) -> Dict[str, Any]:
    """
    Run full similarity filter pipeline: intra-filekey then inter-filekey.

    - input_dir: Directory of filekey subdirs (candidate pages, same as JSON filter input).
    - json_filter_dir: Directory where JSON filter wrote filekey/summary.json.
    - output_dir: Base output (e.g. output/page_filter/clipsim_filtered).
      Writes intra_filekey/ and inter_filekey/ under output_dir.
    """
    input_path = Path(input_dir)
    json_path = Path(json_filter_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    config = config or SimilarityConfig()
    run_intra_similarity_filter(input_path, json_path, output_path, config)
    inter_summary = filter_inter_filekey(output_path / "intra_filekey", output_path, config)
    return inter_summary

if __name__ == "__main__":
    """Run similarity filter with direct configuration."""
    from ...configs.paths import enter_project_root, OUTPUT_DIR
    enter_project_root()
    setup_logging(logger, log_name="similarity_filter")

    result = run_similarity_filter(
        input_dir=OUTPUT_DIR / "page_filter" / "candidate",
        json_filter_dir=OUTPUT_DIR / "page_filter" / "json_filtered",
        output_dir=OUTPUT_DIR / "page_filter" / "clipsim_filtered",
    )
    logger.info(result)
