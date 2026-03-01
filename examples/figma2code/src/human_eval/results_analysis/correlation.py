from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit
import itertools
from ...utils.files import load_json
from ...utils.console_logger import logger


# ============================================================
# Correlation Analysis
# ============================================================

def compute_srocc(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> float:
    """Absolute Spearman Rank-Order Correlation Coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    rho, _ = spearmanr(x, y)
    return abs(rho) if not np.isnan(rho) else np.nan


def compute_plcc(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> float:
    """Absolute Pearson Linear Correlation Coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    r, _ = pearsonr(x, y)
    return abs(r) if not np.isnan(r) else np.nan


def compute_krcc(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> float:
    """Absolute Kendall τ Rank Correlation Coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    tau, _ = kendalltau(x, y)
    return abs(tau) if not np.isnan(tau) else np.nan


def compute_cc_v(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    variances: Optional[Union[List[float], np.ndarray]] = None,
) -> float:
    """Weighted linear regression correlation coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if variances is None:
        variances = np.ones_like(y)
    else:
        variances = np.asarray(variances, dtype=float)
        variances = variances[mask]
    weights = 1.0 / variances

    X = np.vstack([x, np.ones(len(x))]).T
    W = np.diag(weights)
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    predicted = X @ beta
    return np.corrcoef(y, predicted)[0, 1]


def compute_cc_n(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> float:
    """Logistic non-linear regression correlation coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    def _logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    try:
        p0 = [1, 1 / np.std(x), np.median(x)]
        popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=10000)
        predicted = _logistic(x, *popt)
        return np.corrcoef(y, predicted)[0, 1]
    except Exception:
        a, b = np.polyfit(x, y, 1)
        predicted = a * x + b
        return np.corrcoef(y, predicted)[0, 1]


def calculate_correspondance_metrics(
    objective_scores: np.ndarray,
    subjective_scores: np.ndarray,
    variances: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Correspondence metrics between objective scores and subjective MOS.

    Returns:
        CC_V  – Pearson r after *weighted linear regression*
        CC_N  – Pearson r after *logistic non-linear regression* (fallback: linear)
        SROCC – abs(Spearman ρ)
    """
    cc_v = compute_cc_v(objective_scores, subjective_scores, variances)
    cc_n = compute_cc_n(objective_scores, subjective_scores)
    srocc = compute_srocc(objective_scores, subjective_scores)

    return {"CC_V": cc_v, "CC_N": cc_n, "SROCC": srocc}


# ---------- High-level correlation matrices ----------

VISUAL_METRIC_COLUMNS: List[str] = ["MAE_px", "DINOv2", "PSNR", "SSIM", "CLIP", "LPIPS"]
VISUAL_INVERT_METRICS: List[str] = ["MAE_px", "LPIPS"]

# Default column renaming for code quality metrics
CODE_METRIC_RENAME: Dict[str, str] = {
    "RESP_relative_unit_share": "RUR",
    "RESP_absolute_or_fixed_positioning_rate": "APR",
    "RESP_flex_grid_on_containers_rate": "FU",
    "RESP_breakpoint_coverage_share": "BC",
    "MAINT_semantic_tag_share": "STR",
    "MAINT_arbitrary_value_usage_rate": "AVU",
    "MAINT_inline_style_rate": "ISR",
    "MAINT_custom_class_reuse_rate": "CCR",
}

CODE_METRIC_COLUMNS: List[str] = list(CODE_METRIC_RENAME.keys())

CODE_REVERSE_METRICS: List[str] = [
    "RESP_absolute_or_fixed_positioning_rate",
    "MAINT_arbitrary_value_usage_rate",
    "MAINT_inline_style_rate",
]


def compute_visual_correspondance_matrix(
    metrics_csv: Path,
    mos_file: Path,
    output_csv: Path,
    metric_columns: Optional[List[str]] = None,
    invert_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Visual correspondance matrix (CC_V, CC_N, SROCC) (MOS+per-annotator).
    Output columns: metric_obj, user, cc_v, cc_n, srocc 
    """
    df = pd.read_csv(metrics_csv)

    if metric_columns is None:
        metric_columns = VISUAL_METRIC_COLUMNS
    if invert_metrics is None:
        invert_metrics = VISUAL_INVERT_METRICS
    
    # Invert "lower-is-better" metrics
    for metric_obj in invert_metrics:
        if metric_obj in df.columns:
            df[metric_obj] = -df[metric_obj]

    # Only keep folder and relevant metric columns for processing
    available = [m for m in metric_columns if m in df.columns]
    if not available:
        logger.warning("No available metric columns, skipping visual correspondance matrix.")
        return pd.DataFrame()
    keep_cols = ["folder"] + available
    df = df[keep_cols]

    mos_data = load_json(mos_file)
    records = []

    for mos_name, mos_dict in mos_data.items():
        if len(mos_dict) != len(df):
            logger.error(f"MOS count ({len(mos_dict)}) does not match CSV rows ({len(df)})")
            return pd.DataFrame()

        mos_series = df["folder"].map(mos_dict)
        for metric_obj in available:
            metrics = df[metric_obj]
            correspondance_row = calculate_correspondance_metrics(metrics, mos_series)
            row = {
                "metric_obj": metric_obj,
                "user": mos_name,
                "cc_v": correspondance_row.get("CC_V", float('nan')),
                "cc_n": correspondance_row.get("CC_N", float('nan')),
                "srocc": correspondance_row.get("SROCC", float('nan')),
            }
            records.append(row)

    df_correspondance = pd.DataFrame.from_records(records, columns=["metric_obj", "user", "cc_v", "cc_n", "srocc"])
    df_correspondance.to_csv(output_csv, index=False)
    logger.info(f"✔ Visual correspondance matrix saved to {output_csv}")
    logger.info(df_correspondance)
    return df_correspondance


def compute_code_correspondance_matrix(
    code_metrics_csv: Path,
    mos_file: Path,
    output_csv: Path,
    metric_columns: Optional[List[str]] = None,
    reverse_metrics: Optional[List[str]] = None,
    rename_dict: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Full correspondence metrics (CC_V, CC_N, SROCC) between
    code quality metrics and per-user MOS (across all dimensions / users).
    Output columns: metric_obj, metric_sub, user, cc_v, cc_n, srocc
    """
    df = pd.read_csv(code_metrics_csv)
    
    if metric_columns is None:
        metric_columns = CODE_METRIC_COLUMNS
    if reverse_metrics is None:
        reverse_metrics = CODE_REVERSE_METRICS
    if rename_dict is None:
        rename_dict = CODE_METRIC_RENAME

    # Reverse "lower-is-better" metrics
    for metric_obj in reverse_metrics:
        if metric_obj in df.columns:
            df[metric_obj] = -df[metric_obj]

    # Only keep folder and relevant metric columns for processing, and rename accordingly
    available = [m for m in metric_columns if m in df.columns]
    if not available:
        logger.warning("No available metric columns, skipping code correspondance matrix.")
        return pd.DataFrame()
    keep_cols = ["folder"] + available
    df = df[keep_cols].rename(columns=rename_dict)
    renamed = [rename_dict.get(m, m) for m in available]

    mos_data = load_json(mos_file)
    records = []

    for metric_sub, metric_sub_mos in mos_data.items():
        for user_name, mos_dict in metric_sub_mos.items():
            if len(mos_dict) != len(df):
                logger.error(f"MOS count ({len(mos_dict)}) does not match CSV rows ({len(df)})")
                return pd.DataFrame()
            
            mos_series = df["folder"].map(mos_dict)
            for metric_obj in renamed:
                metrics = df[metric_obj]
                correspondance_row = calculate_correspondance_metrics(metrics, mos_series)
                row = {
                    "metric_obj": metric_obj,
                    "metric_sub": metric_sub,
                    "user": user_name,
                    "cc_v": correspondance_row.get("CC_V", float('nan')),
                    "cc_n": correspondance_row.get("CC_N", float('nan')),
                    "srocc": correspondance_row.get("SROCC", float('nan')),
                }
                records.append(row)

    df_correspondance = pd.DataFrame.from_records(
        records, columns=["metric_obj", "metric_sub", "user", "cc_v", "cc_n", "srocc"]
    )
    df_correspondance.to_csv(output_csv, index=False)
    logger.info(f"✔ Code correspondance matrix saved to {output_csv}")
    logger.info(df_correspondance)
    return df_correspondance


# ============================================================
# Inter-Rater Agreement
# ============================================================

def compute_inter_rater_correlation(
    users_data: Dict[str, Dict[str, Dict[str, float]]],
    output_csv: Path,
) -> List[dict]:
    """
    Pairwise inter-rater agreement (Pearson CC + Spearman ρ) per task.

    Expected users_data structure::

        { "metric_name": { "user1": { "sample_id": score }, ... }, ... }
    """
    all_results = []

    for metric_name, metric_data in users_data.items():
        logger.info(f"===== Processing metric: {metric_name} =====")
        desc = _compute_pairwise_correlation(metric_name, metric_data)
        for k, v in desc.items():
            logger.info(f"  {k}: {v}")
        all_results.append(desc)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"✔ Inter-rater results saved to {output_csv}")
        logger.info(df)

    return all_results


def _compute_pairwise_correlation(metric_name: str, metric_data: dict) -> dict:
    """
    Compute all pair-wise Pearson + Spearman for one metric.
    Expected metric_data structure:
        { "user1": { "sample_id": score }, ... }
    """
    users = list(metric_data.keys())
    pairs = list(itertools.combinations(users, 2))

    correlation_coefficients = []
    rank_order_correlations = []
    for u1, u2 in pairs:
        s1, s2 = metric_data[u1], metric_data[u2]
        common = set(s1.keys()) & set(s2.keys())
        if not common:
            continue
        x = [s1[k] for k in common]
        y = [s2[k] for k in common]
        # Pearson correlation
        corr = np.corrcoef(x, y)[0, 1]
        # Spearman rank-order correlation
        rank_corr, _ = spearmanr(x, y)
        if not np.isnan(corr):
            correlation_coefficients.append(corr)
        if not np.isnan(rank_corr):
            rank_order_correlations.append(rank_corr)

    if correlation_coefficients:
        average_correlation = float(np.mean(correlation_coefficients))
    else:
        average_correlation = None
    
    if rank_order_correlations:
        average_rank_order = float(np.mean(rank_order_correlations))
    else:
        average_rank_order = None

    return {
        "Metric Name": metric_name,
        "User Count": len(users),
        "Pair Count": len(pairs),
        "Average Correlation Coefficient": average_correlation,
        "**Average Rank-Order Correlation Coefficient**": average_rank_order,
        "Max Correlation Coefficient": max(correlation_coefficients) if correlation_coefficients else None,
        "Min Correlation Coefficient": min(correlation_coefficients) if correlation_coefficients else None,
        "Max Rank-Order Correlation Coefficient": max(rank_order_correlations) if rank_order_correlations else None,
        "Min Rank-Order Correlation Coefficient": min(rank_order_correlations) if rank_order_correlations else None,
    }