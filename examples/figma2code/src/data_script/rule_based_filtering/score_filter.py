"""
Score-based filtering using LLM filter.json.

Uses llm_images_annotator to annotate images with score prompt -> filter.json,
then filters by score thresholds, moves removed samples, and plots before/after distributions.
"""

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

from ..annotation.llm_images_annotator import annotate_images
from ...utils.console_logger import logger

FILTER_KEYS = [
    "structure_complexity_confidence",
    "multi_page_confidence",
    "widget_page_confidence",
    "total_sum",
]

SCORE_PROMPT = """
You are a professional UI image analyst. Carefully analyze the provided UI image and output a structured JSON containing the following information:

---

1. structure_complexity_confidence
- Definition: Your confidence that the UI structure is *too simple*.
- Scale (1–5):
  - 1 = Very complex, rich layout, clearly not simple
  - 3 = Moderately simple, some structure but limited variety of elements
  - 5 = Extremely simple (very few elements, almost no hierarchy)(but some pages like landing page,etc should not be marked 5 ,they has some hierarchy)

---

2. multi_page_confidence
- Definition: Your confidence that the UI represents *multiple pages/screens*.
  - "Multiple pages/screens" means the image shows two or more distinct UI layouts side by side or within one canvas.
  - If the image very clearly shows multiple devices (e.g., mobile, tablet, desktop) or multiple screen mockups at once, this should be rated as 5.
- Scale (1–5):
  - 1 = Clearly a single page
  - 3 = Possibly multiple views, uncertain
  - 5 = Very clearly multiple pages/screens (e.g., multiple mobile/tablet/desktop screens shown together)

---

3. widget_page_confidence
- Definition: Your confidence that the UI is a *widget/control page*.
  - A widget/control page mainly consists of independent UI components placed together without a unified layout (e.g., many buttons, cards, or form elements scattered on the canvas).
  - It looks more like a component collection rather than a complete page design.
  - Important rule: If the UI clearly represents a structured screen for mobile, tablet, or desktop, it should not be considered a widget/control page, even if it contains components.

- Scale (1–5):
  - 1 = Clearly not a widget/control page (it’s a proper page design for a device)
  - 3 = Some scattered components but still some structure
  - 5 = Definitely a widget/control/component collection (unless inside a design tool panel)

---

4. total_sum
- Definition: The integer sum of the three confidence scores above (structure_complexity_confidence + multi_page_confidence + widget_page_confidence).

---

Strict requirements:
- Return a single JSON object.
- Do not include any explanations, reasoning, or extra text.
- Each confidence score must be an integer from 1 to 5.
- Include the field "total_sum" and remove any "keep" field.
- JSON must strictly follow the format.

Example output:
{
  "structure_complexity_confidence": 4,
  "multi_page_confidence": 3,
  "widget_page_confidence": 5,
  "total_sum": 12
}
"""


@dataclass
class ScoreFilterConfig:
    """Configuration for filtering by filter.json scores."""
    total_sum_threshold: int = 12  # Remove if total_sum >= this
    max_single_score_threshold: int = 4  # Remove if max(s, m, w) >= this


def collect_filter_scores(base_dir: Path) -> Tuple[List[int], List[int], List[int]]:
    """
    Collect structure_complexity_confidence, multi_page_confidence, widget_page_confidence
    from all filter.json under base_dir/filekey/.

    Returns:
        (structure_scores, multi_scores, widget_scores)
    """
    structure_scores: List[int] = []
    multi_scores: List[int] = []
    widget_scores: List[int] = []
    for fk_dir in base_dir.iterdir():
        if not fk_dir.is_dir():
            continue
        filter_path = fk_dir / "filter.json"
        if not filter_path.exists():
            continue
        try:
            data = json.loads(filter_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", filter_path, e)
            continue
        for v in data.values():
            if not isinstance(v, dict):
                continue
            if not all(k in v for k in FILTER_KEYS):
                continue
            structure_scores.append(v["structure_complexity_confidence"])
            multi_scores.append(v["multi_page_confidence"])
            widget_scores.append(v["widget_page_confidence"])
    return structure_scores, multi_scores, widget_scores


def _should_remove(entry: Dict[str, Any], config: ScoreFilterConfig) -> bool:
    """True if this entry should be removed (filtered out)."""
    if not all(k in entry for k in FILTER_KEYS):
        return False
    s = entry["structure_complexity_confidence"]
    m = entry["multi_page_confidence"]
    w = entry["widget_page_confidence"]
    t_sum = entry["total_sum"]
    return t_sum >= config.total_sum_threshold or max(s, m, w) >= config.max_single_score_threshold


def apply_filter_by_scores(
    base_dir: Path,
    removed_dir: Path,
    config: Optional[ScoreFilterConfig] = None,
    global_summary_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Filter samples by filter.json scores: move removed images to removed_dir,
    update filter.json and annotations.json in place (remove entries for removed samples),
    write summary.json per filekey and optionally a global summary.

    Returns:
        Global summary dict (total counts, retained/removed node_ids).
    """
    config = config or ScoreFilterConfig()
    removed_dir = Path(removed_dir)
    removed_dir.mkdir(parents=True, exist_ok=True)
    global_retained: List[str] = []
    global_removed: List[str] = []
    total_images_global = 0

    filekey_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    for fk_dir in tqdm(filekey_dirs, desc="Filter by scores"):
        filter_path = fk_dir / "filter.json"
        if not filter_path.exists():
            continue
        try:
            filters = json.loads(filter_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", filter_path, e)
            continue

        removed_filters: Dict[str, Any] = {}
        for node_id, entry in list(filters.items()):
            if not isinstance(entry, dict):
                continue
            if not all(k in entry for k in FILTER_KEYS):
                continue
            total_images_global += 1
            if _should_remove(entry, config):
                img_path = fk_dir / f"{node_id}.png"
                out_fk = removed_dir / fk_dir.name
                out_fk.mkdir(parents=True, exist_ok=True)
                if img_path.exists():
                    shutil.move(str(img_path), str(out_fk / img_path.name))
                removed_filters[node_id] = filters.pop(node_id, None)

        filter_path.write_text(json.dumps(filters, ensure_ascii=False, indent=2), encoding="utf-8")
        if removed_filters:
            out_fk = removed_dir / fk_dir.name
            (out_fk / "filter.json").write_text(
                json.dumps(removed_filters, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        before_count = len(filters) + len(removed_filters)
        fk_summary = {
            "total_images_before_filter": before_count,
            "retained_images_count": len(filters),
            "removed_images_count": len(removed_filters),
            "retained_node_ids": sorted(filters.keys()),
            "removed_node_ids": sorted(removed_filters.keys()),
        }
        (fk_dir / "summary.json").write_text(
            json.dumps(fk_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        global_retained.extend(filters.keys())
        global_removed.extend(removed_filters.keys())

    global_summary = {
        "total_images_before_filter": total_images_global,
        "retained_images_count": len(global_retained),
        "removed_images_count": len(global_removed),
        "retained_node_ids": global_retained,
        "removed_node_ids": global_removed,
    }
    if global_summary_file:
        global_summary_file.parent.mkdir(parents=True, exist_ok=True)
        global_summary_file.write_text(
            json.dumps(global_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Global summary saved to %s", global_summary_file)
    return global_summary


def plot_filter_distributions(
    before_scores: Tuple[List[int], List[int], List[int]],
    after_scores: Tuple[List[int], List[int], List[int]],
    output_path: Path,
) -> None:
    """
    Plot score distributions before and after filtering (3 metrics x before/after).
    Saves a figure with 2 rows x 3 columns.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping distribution plot")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ("Structure Complexity", "Multi Page", "Widget Page")
    keys = list(range(1, 6))
    for col, (title, before, after) in enumerate(zip(titles, before_scores, after_scores)):
        c_before = Counter(before)
        c_after = Counter(after)
        vals_before = [c_before.get(k, 0) for k in keys]
        vals_after = [c_after.get(k, 0) for k in keys]
        axes[0, col].bar(keys, vals_before)
        axes[0, col].set_xticks(keys)
        axes[0, col].set_xlabel("Score")
        axes[0, col].set_ylabel("Count")
        axes[0, col].set_title(f"{title} (before filter)")
        axes[1, col].bar(keys, vals_after)
        axes[1, col].set_xticks(keys)
        axes[1, col].set_xlabel("Score")
        axes[1, col].set_ylabel("Count")
        axes[1, col].set_title(f"{title} (after filter)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Distribution plot saved to %s", output_path)


def run_score_filter(
    base_dir: Path,
    removed_dir: Optional[Path] = None,
    plot_output_path: Optional[Path] = None,
    filter_config: Optional[ScoreFilterConfig] = None,
    model: str = "openai/gpt-4o-mini",
    run_annotate_scores: bool = True,
    force_annotate: bool = False,
) -> Dict[str, Any]:
    """
    Score annotation and filter pipeline:
    1. Annotate all images with SCORE_PROMPT -> filter.json (if run_annotate_scores).
    2. Collect before scores, apply filter by scores (move removed, update files), collect after scores.
    3. Plot before/after distributions.
    """
    base_dir = Path(base_dir)
    removed_dir = Path(removed_dir) if removed_dir else base_dir.parent / f"{base_dir.name}_removed"
    plot_output_path = plot_output_path or base_dir.parent / "score_distributions.png"
    filter_config = filter_config or ScoreFilterConfig()

    if run_annotate_scores:
        logger.info("Step 1: Annotate with score prompt -> filter.json")
        annotate_images(base_dir, SCORE_PROMPT, "filter.json", model=model, force=force_annotate)

    logger.info("Step 2: Collect before scores and apply filter")
    before_scores = collect_filter_scores(base_dir)
    global_summary_file = base_dir.parent / "score_filter_global_summary.json"
    summary = apply_filter_by_scores(base_dir, removed_dir, filter_config, global_summary_file)
    after_scores = collect_filter_scores(base_dir)

    if before_scores[0] or after_scores[0]:
        logger.info("Step 3: Plot before/after distributions")
        plot_filter_distributions(before_scores, after_scores, plot_output_path)

    return summary


if __name__ == "__main__":
    from ...configs.paths import enter_project_root, OUTPUT_DIR
    from ...utils.console_logger import setup_logging
    enter_project_root()
    setup_logging(logger, log_name="score_filter")

    base_dir = OUTPUT_DIR / "page_filter" / "clipsim_filtered" / "inter_filekey"
    removed_dir = base_dir.parent / "inter_filekey_removed"
    plot_output_path = base_dir.parent / "score_distributions.png"
    result = run_score_filter(
        base_dir=base_dir,
        removed_dir=removed_dir,
        plot_output_path=plot_output_path,
    )
    logger.info(result)
