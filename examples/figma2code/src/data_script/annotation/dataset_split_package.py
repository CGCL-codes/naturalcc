"""
Dataset split and package for Gradio annotation.

Splits base_dir into k parts by image count, creates split_i/candidate|selected|trash
and final_annotation.json, report.json; then packages each split with the Gradio
annotation script into zip files. Run = split + package.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

from ...utils.console_logger import logger


def split_dataset(base_dir: Path, k: int, output_dir: Path) -> List[Path]:
    """
    Evenly split subfolders under base_dir into k parts by image count; copy to output_dir.

    Each split_i/ contains:
        candidate/   (assigned filekey subfolders)

    Returns:
        List of split directories (output_dir / "split_0", ...).
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    fk_dirs: List[tuple] = []
    for d in base_dir.iterdir():
        if d.is_dir():
            img_count = len(list(d.glob("*.png")))
            fk_dirs.append((d, img_count))

    if not fk_dirs:
        logger.warning("No subfolders or images found under %s", base_dir)
        return []

    fk_dirs.sort(key=lambda x: x[1], reverse=True)
    splits: List[List[Path]] = [[] for _ in range(k)]
    split_sizes = [0] * k
    for folder, count in fk_dirs:
        idx = split_sizes.index(min(split_sizes))
        splits[idx].append(folder)
        split_sizes[idx] += count

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dirs: List[Path] = []
    for i, split in enumerate(splits):
        split_dir = output_dir / f"split_{i}"
        split_dir.mkdir(exist_ok=True)
        candidate_dir = split_dir / "candidate"
        candidate_dir.mkdir(exist_ok=True)
        for folder in split:
            dest = candidate_dir / folder.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(folder, dest)
        split_dirs.append(split_dir)
        logger.info("split_%d: %d images, %d folders", i, split_sizes[i], len(splits[i]))

    logger.info("Dataset split complete -> %s", output_dir.resolve())
    return split_dirs


def package_splits(
    output_dir: Path,
    gradio_script_path: Path,
    k: Optional[int] = None,
    split_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    Package each split_i under output_dir with the Gradio annotation script into a zip.

    Zip layout: split_i/... (candidate) + script at root (basename).

    Args:
        output_dir: Directory containing split_0, split_1, ...
        gradio_script_path: Path to the Gradio annotation script (e.g. gradio_filter_and_annotation.py).
        k: Number of splits (used if split_indices not set: package split_0 .. split_{k-1}).
        split_indices: Which split indices to package (default: all existing split_* dirs).

    Returns:
        List of created zip paths.
    """
    output_dir = Path(output_dir)
    gradio_script_path = Path(gradio_script_path).resolve()
    if not gradio_script_path.is_file():
        logger.error("Gradio script not found: %s", gradio_script_path)
        return []

    if split_indices is not None:
        indices = list(split_indices)
    elif k is not None:
        indices = list(range(k))
    else:
        indices = []
        for d in output_dir.iterdir():
            if d.is_dir() and d.name.startswith("split_") and d.name[6:].isdigit():
                indices.append(int(d.name[6:]))
        indices.sort()

    created: List[Path] = []
    for i in indices:
        split_dir = output_dir / f"split_{i}"
        if not split_dir.is_dir():
            logger.warning("Split dir missing: %s", split_dir)
            continue
        zip_path = output_dir / f"split_{i}_with_gradio.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _dirs, files in os.walk(split_dir):
                    root_p = Path(root)
                    for f in files:
                        file_path = root_p / f
                        rel = file_path.relative_to(split_dir)
                        arcname = Path(f"split_{i}") / rel
                        zipf.write(file_path, arcname)
                zipf.write(gradio_script_path, gradio_script_path.name)
            created.append(zip_path)
            logger.info("Created %s", zip_path)
        except Exception as e:
            logger.exception("Failed to create %s: %s", zip_path, e)
    return created


def run_split_package(
    base_dir: Path,
    output_dir: Path,
    k: int = 4,
    gradio_script_path: Optional[Path] = None,
    run_split: bool = True,
    run_package: bool = True,
) -> None:
    """
    Split dataset then package each split with the Gradio script (split + package).

    Args:
        base_dir: Original dataset (filekey subfolders with *.png).
        output_dir: Where to write split_0, split_1, ... and *.zip.
        k: Number of splits.
        gradio_script_path: Path to Gradio annotation script (required if run_package).
        run_split: Whether to run split.
        run_package: Whether to run package after split.
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    if gradio_script_path is not None:
        gradio_script_path = Path(gradio_script_path).resolve()
    elif run_package:
        # Default: script next to this file
        gradio_script_path = Path(__file__).parent / "gradio_filter_and_annotation.py"
        if not gradio_script_path.is_file():
            logger.error("Default Gradio script not found: %s", gradio_script_path)
            run_package = False

    if run_split:
        split_dataset(base_dir, k, output_dir)
    if run_package and gradio_script_path is not None:
        package_splits(output_dir, gradio_script_path, k=k)


if __name__ == "__main__":
    from ...configs.paths import enter_project_root, OUTPUT_DIR
    from ...utils.console_logger import setup_logging

    enter_project_root()
    setup_logging(logger, log_name="dataset_split_package")

    base_dir = OUTPUT_DIR / "page_filter" / "clipsim_filtered" / "inter_filekey"
    output_dir = OUTPUT_DIR / "page_filter" / "split_set"
    # Path to Gradio annotation script (must be set for packaging)
    gradio_script_path = Path(__file__).resolve().parent / "gradio_filter_and_annotation.py"

    run_split_package(
        base_dir=base_dir,
        output_dir=output_dir,
        k=4,
        gradio_script_path=gradio_script_path,
    )
