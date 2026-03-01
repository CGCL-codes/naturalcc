from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
from ...utils.files import load_json, save_json
from ...utils.console_logger import logger


# ============================================================
# MOS Processing  (Z-score normalization)
# ============================================================


def collect_user_data(
    input_dirs: List[str],
    prefix_filter: str = "results_",
) -> Dict[str, dict]:
    """
    Collect all user annotation data from one or more directories.
    {prefix_filter}_{username}.json are loaded.
    Same-named users across directories are merged (``dict.update``).
    """
    all_users_data: Dict[str, dict] = defaultdict(dict)

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logger.warning(f"Directory {input_dir} does not exist, skipping.")
            continue

        json_files = [
            f for f in input_dir.glob("*.json") if f.name.startswith(prefix_filter) 
        ]
        logger.info(f"Found {len(json_files)} annotation files in {input_dir}.")

        for file in json_files:
            username = file.stem.split("_", 1)[1]
            data = load_json(file)
            all_users_data[username].update(data)

    return dict(all_users_data)


def _extract_sample_id(full_key: str) -> str:
    return full_key.split("__", 1)[1] if "__" in full_key else full_key


def collect_valid_keys(
    all_users_data: Dict[str, dict],
) -> Set[str]:
    """
    Collect the set of valid sample keys from user annotation data.

    Handles ``model__sample_id`` key format by splitting on ``"__"``.
    """
    valid_keys: Set[str] = set()

    for user_data in all_users_data.values():
        for full_key in user_data.keys():
            sample_id = _extract_sample_id(full_key)
            valid_keys.add(sample_id)

    logger.info(f"Collected {len(valid_keys)} unique valid sample keys.")
    return valid_keys


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Compute Z-scores for *arr*.  Returns zeros when std == 0."""
    m = np.mean(arr)
    s = np.std(arr)
    return np.zeros_like(arr) if s == 0 else (arr - m) / s


def calculate_z_scores_single(
    all_users_data: Dict[str, dict],
    valid_keys: Set[str],
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """
    Z-score normalise single-value annotations (visual evaluation).

    Returns:
        sample_pool: ``{sample_id: [z_score_from_user1, z_score_from_user2, ...]}``
        user_scores: ``{username: {sample_id: z_score}}``

    """
    sample_pool: Dict[str, List[float]] = defaultdict(list)
    user_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for username, user_data in all_users_data.items():
        raw_scores = list(user_data.values())
        if not raw_scores:
            continue

        raw_np = np.array(raw_scores, dtype=float)
        z = _zscore(raw_np)

        for i, full_key in enumerate(user_data.keys()):
            sid = _extract_sample_id(full_key)
            if valid_keys and sid not in valid_keys:
                continue
            sample_pool[sid].append(float(z[i]))
            user_scores[username][sid] = float(z[i])
    
    return sample_pool, user_scores


def calculate_z_scores_per_metric(
    all_users_data: Dict[str, dict],
    valid_keys: Set[str],
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Z-score normalise multi-metric annotations (code evaluation).

    Returns:
        metric_pools:  ``{metric: {sample_id: [z1, z2, ...]}}``
        user_scores:   ``{metric: {username: {sample_id: z}}}``
    """
    metric_pools: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    user_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    # Auto-discover metrics
    metrics: Set[str] = set()
    for user_data in all_users_data.values():
        for val in user_data.values():
            if isinstance(val, dict):
                metrics.update(val.keys())
        if metrics:
            break
    print(f"Detected metrics: {sorted(metrics)}")

    for metric in metrics:
        for username, user_data in all_users_data.items():
            raw_scores, sample_keys = [], []
            for key, val in user_data.items():
                if isinstance(val, dict) and metric in val:
                    try:
                        raw_scores.append(float(val[metric]))
                        sample_keys.append(key)
                    except (ValueError, TypeError):
                        logger.warning(f"Failed to parse score for metric {metric} from user {username} for key {key}")
                        continue
            if not raw_scores:
                continue

            z = _zscore(np.array(raw_scores))
            for i, full_key in enumerate(sample_keys):
                sid = _extract_sample_id(full_key)
                if valid_keys and sid not in valid_keys:
                    continue
                zv = round(float(z[i]), 4)
                metric_pools[metric][sid].append(zv)
                user_scores[metric][username][sid] = zv

    return metric_pools, user_scores


def calculate_summed_scores(all_users_data: Dict[str, dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Sum all per-metric sub-scores into a single ``Summed Score`` per sample.
    Output: 
    ```json
    {
        "username": {
            "sample_key": {
                "metric": score,
                ...
                "Summed Score": total
            },
        }
    }
    ```
    """
    summed: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for username, user_data in all_users_data.items():
        for sample_key, val in user_data.items():
            if isinstance(val, dict):
                total = 0.0
                for metric, score in val.items():
                    summed[username][sample_key][metric] = float(score)
                    total += float(score)
                summed[username][sample_key]["Summed Score"] = total
            else:
                summed[username][sample_key]["Summed Score"] = float(val)

    return summed


def calculate_mos_single(sample_pool: Dict[str, List[float]]) -> Dict[str, float]:
    """Aggregate per-sample z-score lists into MOS (mean)."""
    return {sid: round(float(np.mean(scores)), 4) for sid, scores in sample_pool.items()}


def calculate_mos_multi(metric_pools: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample z-score lists into MOS (mean), per metric."""
    return {metric: {sid: round(float(np.mean(scores)), 4) for sid, scores in pool.items()} for metric, pool in metric_pools.items()}

# ---------- Full MOS pipelines ----------

def compute_mos_visual(
    input_dirs: List[str],
    output_file: Path,
) -> None:
    """
    Full pipeline for **visual** (image-similarity) evaluation.

    Annotation files: each JSON maps ``key → score (number)``.
    Output file: 
    ```json
    {   
        "username": {
            "sample_id": z_score,
            ...
        },
        "all": {
            "sample_id": mos
        }
    }
    ```
    """
    all_users = collect_user_data(input_dirs)
    if not all_users:
        logger.error("No annotation data found, exiting.")
        return

    valid_keys = collect_valid_keys(all_users)
    sample_pool, user_scores = calculate_z_scores_single(all_users, valid_keys)
    mos = calculate_mos_single(sample_pool)

    user_scores.update({"all": mos})

    save_json(user_scores, output_file)
    logger.info(f"Final MOS results saved to: {output_file}")


def compute_mos_code(
    input_dirs: List[str],
    output_file: Path,
) -> None:
    """
    Full pipeline for **code-quality** evaluation (per-metric MOS).

    Annotation files: each JSON maps ``key → {metric: score, ...}``.
    Output file:
    ```json
    {
        "metric": {
            "username": {
                "sample_id": z_score,
                ...
            },
            "all": {
                "sample_id": mos,
            }
        },
        "summed_scores": {
            "username": {
                "sample_id": summed_score,
            },
            "all": {
                "sample_id": mos,
            }
        }
    ```
    """
    all_users = collect_user_data(input_dirs, prefix_filter="")
    if not all_users:
        logger.error("No annotation data found, exiting.")
        return

    valid_keys = collect_valid_keys(all_users)
    summed = calculate_summed_scores(all_users)
    metric_pools, user_scores = calculate_z_scores_per_metric(summed, valid_keys)
    mos = calculate_mos_multi(metric_pools)

    for metric, metric_scores in user_scores.items():
        metric_scores.update({"all": mos[metric]})

    save_json(user_scores, output_file)
    logger.info(f"Final MOS results saved to: {output_file}")

