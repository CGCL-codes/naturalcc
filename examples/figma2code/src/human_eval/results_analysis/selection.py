from typing import Optional, Set, List, Dict
from pathlib import Path
import shutil
import pandas as pd
from ...utils.console_logger import logger

# ============================================================
# Data Filtering & Sample Selection
# ============================================================

def filter_metrics_from_results(
    results_csv: Path,
    output_csv: Path,
    target_result_name: str,
    sample_ids: Optional[Set[str]] = None,
    metric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter a full experiment results CSV by model and (optionally) sample IDs,
    keeping only the specified metric columns.
    """
    df = pd.read_csv(results_csv)

    df_f = df[df["result_name"] == target_result_name].copy()
    if sample_ids:
        df_f = df_f[df_f["folder"].isin(sample_ids)]
    
    if metric_columns:
        keep = ["folder"] + [m for m in metric_columns if m in df_f.columns]
        df_f = df_f[keep]
    
    df_f.to_csv(output_csv, index=False)
    logger.info(f"✔ Filtered metrics saved to {output_csv} (extracted {len(df_f)} rows)")
    return df_f


# Quadrant names: (mae_level, mos_level)
QUADRANTS = [
    "high_mae_high_mos",
    "high_mae_low_mos",
    "low_mae_high_mos",
    "low_mae_low_mos",
]


def select_extreme_samples(
    mos_data: Dict[str, float],
    metrics_file: Path,
    output_dir: Path,
    samples_per_quadrant: int = 4,
    source_cases_dir: Optional[Path] = None,
    cases_output_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Select samples in four quadrants (by MAE/MOS median): high_mae_high_mos,
    high_mae_low_mos, low_mae_high_mos, low_mae_low_mos. Picks up to
    samples_per_quadrant (default 4) in each quadrant. 
    Copy case folders to a target directory (with subdirs per quadrant).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_mos = pd.DataFrame(list(mos_data.items()), columns=["folder", "MOS"])

    df_metric = pd.read_csv(metrics_file)
    df_metric["folder"] = df_metric["folder"].astype(str)

    df = pd.merge(df_mos, df_metric[["folder", "MAE_px"]], on="folder", how="inner")
    logger.info(f"Merged data shape: {df.shape}")

    if df.empty:
        logger.error("Error: No overlapping keys found.")
        return None

    mae_thr = df["MAE_px"].quantile(0.5)
    mos_thr = df["MOS"].quantile(0.5)
    logger.info(f"MAE median (threshold): {mae_thr:.4f}")
    logger.info(f"MOS median (threshold): {mos_thr:.4f}")

    high_mae = df["MAE_px"] >= mae_thr
    low_mae = df["MAE_px"] < mae_thr
    high_mos = df["MOS"] >= mos_thr
    low_mos = df["MOS"] < mos_thr

    # (quadrant_name, mask, sort_mae_asc, sort_mos_asc) for picking "most extreme" in quadrant
    quadrant_specs = [
        ("high_mae_high_mos", high_mae & high_mos, False, False),
        ("high_mae_low_mos", high_mae & low_mos, False, True),
        ("low_mae_high_mos", low_mae & high_mos, True, False),
        ("low_mae_low_mos", low_mae & low_mos, True, True),
    ]

    selected_rows: List[pd.DataFrame] = []
    selected_keys: List[str] = []
    key_to_quadrant: Dict[str, str] = {}

    for name, mask, mae_asc, mos_asc in quadrant_specs:
        sub = df.loc[mask].copy()
        sub = sub.sort_values(
            by=["MAE_px", "MOS"],
            ascending=[mae_asc, mos_asc],
        ).head(samples_per_quadrant)
        if len(sub) > 0:
            selected_rows.append(sub.assign(quadrant=name))
            keys = sub["folder"].tolist()
            selected_keys.extend(keys)
            for k in keys:
                key_to_quadrant[k] = name
            logger.info(f"  {name}: selected {len(sub)} samples")

    if not selected_rows:
        logger.error("No samples selected in any quadrant.")
        return None

    sel = pd.concat(selected_rows, ignore_index=True)

    keys_file = output_dir / "selected_keys.txt"
    with open(keys_file, "w") as f:
        for k in selected_keys:
            f.write(f"{k}\n")

    summary_file = output_dir / "selected_by_quadrant.csv"
    sel.to_csv(summary_file, index=False)
    logger.info(f"Selected total {len(sel)} samples across quadrants:")
    logger.info(sel.to_string())
    logger.info(f"✔ Saved keys to {keys_file}, summary to {summary_file}")

    # Copy case folders if paths are provided
    if source_cases_dir is not None and cases_output_dir is not None:
        source_cases_dir = Path(source_cases_dir)
        cases_output_dir = Path(cases_output_dir)
        cases_output_dir.mkdir(parents=True, exist_ok=True)
        for key in selected_keys:
            quad = key_to_quadrant[key]
            src_path = source_cases_dir / key
            dst_dir = cases_output_dir / quad
            dst_path = dst_dir / key
            if src_path.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                logger.info(f"  Copied {key} -> {quad}/")
            else:
                logger.warning(f"Source folder not found: {src_path}")

    return sel