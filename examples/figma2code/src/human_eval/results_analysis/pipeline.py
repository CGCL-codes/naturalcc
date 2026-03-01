"""
Results analysis utilities for human evaluation.

Comprehensive pipeline for:
- MOS (Mean Opinion Score) processing from raw visual/code annotations
- Filtering metrics from full experiment results CSV by valid annotation keys
- Visual and code correspondance matrices (CC_V, CC_N, SROCC) vs MOS
- Inter-rater agreement (visual and code)
- MOS vs MAE_px plots
- Extreme-sample selection for case study

Usage:
    # Run full pipeline with built-in AnalysisConfig defaults
    python -m src.human_eval.results_analysis.pipeline
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from .mos import collect_user_data, collect_valid_keys, compute_mos_visual, compute_mos_code
from .correlation import VISUAL_METRIC_COLUMNS, CODE_METRIC_COLUMNS, compute_code_correspondance_matrix, compute_inter_rater_correlation, compute_visual_correspondance_matrix
from .plot import plot_mos_vs_metric
from .selection import filter_metrics_from_results, select_extreme_samples

from ...utils.files import load_json
from ...utils.console_logger import logger, setup_logging 

# ============================================================
# Main Pipeline
# ============================================================

@dataclass
class AnalysisConfig:
    """In-code configuration object for the full analysis pipeline."""

    output_dir: str = "output/human_eval/results_analysis"

    # MOS processing
    visual_annotation_dirs: List[str]= field(default_factory=lambda: ["data/human_annotations/similarity_eval"])
    code_annotation_dirs: List[str] = field(default_factory=lambda: ["data/human_annotations/code_eval"])

    # Pre-computed MOS files
    visual_mos_file: str = "final_mos_visual.json"
    code_mos_file: str = "final_mos_code.json"

    # Metrics
    results_csv: str = "output/results/exp1.csv" # Full experiment results CSV
    visual_metrics_csv: str = "filtered_visual_metrics.csv"
    visual_columns: List[str] = field(default_factory=lambda: VISUAL_METRIC_COLUMNS)
    code_metrics_csv: str = "filtered_code_metrics.csv"
    code_columns: List[str] = field(default_factory=lambda: CODE_METRIC_COLUMNS)
    mae_csv: Optional[str] = None

    # Options
    target_result_name: str = "figma_image_direct__ernie4_5_vl_424b_a47b"
    skip_mos_processing: bool = False
    skip_plots: bool = False


def run_all_analyses(config: AnalysisConfig) -> None:
    """
    Run every analysis step in sequence.
    """
    from ...configs.paths import enter_project_root
    enter_project_root()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logger, output_dir=output_dir, log_name="results_analysis")

    logger.info("Results Analysis Pipeline")

    # ------------------------------------------------------------------
    # MOS Processing (from raw annotations)
    # ------------------------------------------------------------------
    if not config.skip_mos_processing:
        v_dirs = config.visual_annotation_dirs
        if v_dirs:
            logger.info("[Step 1a] Processing visual evaluation annotations...")
            v_mos = output_dir / config.visual_mos_file
            compute_mos_visual(v_dirs, v_mos)

        c_dirs = config.code_annotation_dirs
        if c_dirs:
            logger.info("[Step 1b] Processing code evaluation annotations (per-metric)...")
            c_mos = output_dir / config.code_mos_file
            compute_mos_code(c_dirs, c_mos)

    # ------------------------------------------------------------------
    # Filter metrics from full results CSV 
    # ------------------------------------------------------------------
    results_csv = Path(config.results_csv)
    valid_keys = collect_valid_keys(collect_user_data(config.visual_annotation_dirs))
    if results_csv.exists():
        logger.info("[Step 2] Filtering code metrics from results CSV...")
        filtered_code_csv = output_dir / config.code_metrics_csv
        filtered_visual_csv = output_dir / config.visual_metrics_csv
        filter_metrics_from_results(results_csv, filtered_code_csv, config.target_result_name, valid_keys, config.code_columns)
        filter_metrics_from_results(results_csv, filtered_visual_csv, config.target_result_name, valid_keys, config.visual_columns)

    # ------------------------------------------------------------------
    # Visual correspondance matrix (CC_V, CC_N, SROCC) (MOS+per-annotator)
    # ------------------------------------------------------------------
    v_metrics = output_dir / config.visual_metrics_csv
    v_mos = output_dir / config.visual_mos_file
    mae_csv = output_dir / config.mae_csv if config.mae_csv else None

    if v_metrics.exists() and v_mos.exists():
        logger.info("[Step 3] Computing visual correspondance matrix (CC_V, CC_N, SROCC)...")
        v_correspondance_out = output_dir / "visual_correspondance_matrix.csv"
        compute_visual_correspondance_matrix(v_metrics, v_mos, v_correspondance_out, config.visual_columns)

    # ------------------------------------------------------------------
    # Code correspondance matrix (CC_V, CC_N, SROCC) (MOS+per-annotator)
    # ------------------------------------------------------------------
    c_metrics = output_dir / config.code_metrics_csv
    c_mos = output_dir / config.code_mos_file

    if c_metrics.exists() and c_mos.exists():
        logger.info("[Step 4] Computing code correspondance matrix (CC_V, CC_N, SROCC)...")
        code_out = output_dir / "code_correspondance_matrix.csv"
        compute_code_correspondance_matrix(c_metrics, c_mos, code_out, config.code_columns)

    # ------------------------------------------------------------------
    # Inter-rater agreement
    # ------------------------------------------------------------------
    if v_mos.exists():
        logger.info("[Step 5a] Computing inter-rater agreement (visual)...")
        inter_rater_out = output_dir / "inter_rater_visual.csv"
        v_mos_data = load_json(v_mos)
        v_user_data = {
            "visual_similarity": {
                user: data for user, data in v_mos_data.items() if user != "all"
            }
        }
        compute_inter_rater_correlation(v_user_data, inter_rater_out)

    if c_mos.exists():
        logger.info("[Step 5b] Computing inter-rater agreement (code)...")
        inter_rater_out = output_dir / "inter_rater_code.csv"
        c_mos_data = load_json(c_mos)
        c_user_data = {
            "code_summed_score": {
                user: data for user, data in c_mos_data["Summed Score"].items() if user != "all"
            }
        }
        compute_inter_rater_correlation(c_user_data, inter_rater_out)

    # ------------------------------------------------------------------
    # Plots MOS vs MAE_px
    # ------------------------------------------------------------------
    if not config.skip_plots:
        if v_mos.exists() and v_metrics.exists():
            logger.info("[Step 6] Plotting MOS vs MAE_px...")
            v_mos_data = load_json(v_mos)["all"]
            plot_mos_vs_metric(
                v_mos_data,
                v_metrics,
                metric_column="MAE_px",
                output_plot=output_dir / "mos_vs_mae.png",
                title="MOS vs MAE_px",
                xlabel="MAE_px (Lower is better)",
            )

    # ------------------------------------------------------------------
    # Sample selection for case study
    # ------------------------------------------------------------------
    if v_mos.exists() and v_metrics.exists():
        logger.info("[Step 7] Selecting extreme samples for case study...")
        v_mos_data = load_json(v_mos)["all"]
        source_cases_dir = Path("data/data_test")
        cases_output_dir = output_dir / "selected_samples"
        select_extreme_samples(
            v_mos_data, v_metrics, cases_output_dir, source_cases_dir=source_cases_dir,
            cases_output_dir=cases_output_dir)

    logger.info(f"  All results saved to: {output_dir}")

if __name__ == "__main__":
    config = AnalysisConfig()
    run_all_analyses(config)
