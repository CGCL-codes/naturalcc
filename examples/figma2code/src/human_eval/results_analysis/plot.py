from typing import Optional, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ...utils.console_logger import logger

# ============================================================
# Plotting
# ============================================================

def plot_mos_vs_metric(
    mos_data: Dict[str, float],
    metrics_file: Path,
    metric_column: str,
    output_plot: Path,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "MOS (Higher is better)",
) -> None:
    """
    Scatter plot of MOS vs. one automated metric.

    *mos_data* is a dictionary ``{sample_id: score}``.
    *metrics_file* is a CSV with a ``folder`` column.
    """
    df_mos = pd.DataFrame(list(mos_data.items()), columns=["folder", "MOS"])

    df_m = pd.read_csv(metrics_file)
    df_m["folder"] = df_m["folder"].astype(str)

    df = pd.merge(df_mos, df_m[["folder", metric_column]], on="folder", how="inner")
    logger.info(f"Merged data points for plot: {len(df)}")

    if title is None:
        title = f"Scatter Plot: MOS vs {metric_column}"
    if xlabel is None:
        xlabel = metric_column

    plt.figure(figsize=(10, 6))
    plt.scatter(df[metric_column], df["MOS"], alpha=0.7, edgecolors="w", s=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"✔ Plot saved to {output_plot}")


def plot_correlation_heatmap(
    correlation_df: pd.DataFrame,
    output_plot: Path,
    title: str = "Correlation Heatmap",
) -> None:
    """Heatmap from a correlation DataFrame (numeric columns only)."""
    num_df = correlation_df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, max(8, len(num_df) * 0.4)))

    im = ax.imshow(num_df.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(num_df.columns)))
    ax.set_xticklabels(num_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(num_df.index)))
    ax.set_yticklabels(num_df.index)
    ax.set_title(title)

    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✔ Heatmap saved to {output_plot}")
