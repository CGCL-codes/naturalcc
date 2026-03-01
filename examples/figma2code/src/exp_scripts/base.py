"""
Base class for experiment scripts.

Provides common functionality for generation and evaluation.
"""

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import pandas as pd

from ..configs import (
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    DATA_TEST_DIR,
    ensure_output_dirs,
)
from ..evaluation.evaluator import (
    EvaluationConfig,
    evaluate_single,
    load_reference_image,
    save_results,
    load_existing_results,
    get_evaluated_keys,
    is_already_evaluated,
    merge_results,
)
from ..utils.console_logger import logger, create_progress, console


def setup_sample_directory(
    sample_key: str,
    source_data_dir: Path,
    experiments_output_dir: Optional[Path] = None,
    force_copy: bool = False
) -> Path:
    """
    Create/update experiment output directory for a sample.
    
    Assets are COPIED (not symlinked) so packaging works correctly.
    Assets are copied only once if already present.
    
    Args:
        sample_key: Sample identifier (e.g., "Jn1TxN0MSnHA2746GTdfct_189_83")
        source_data_dir: Path to source data (e.g., data/data_test)
        experiments_output_dir: Path to experiments output (default: output/experiments)
        force_copy: If True, overwrite existing assets
    
    Returns:
        Path to the sample's output directory
    """
    if experiments_output_dir is None:
        experiments_output_dir = EXPERIMENTS_DIR
    
    source_sample = source_data_dir / sample_key
    target_dir = experiments_output_dir / sample_key
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Resources to copy (name, is_directory)
    resources = [
        ("assets", True),
        ("root.png", False),
        ("processed_metadata.json", False),
    ]
    
    for name, is_dir in resources:
        source = source_sample / name
        target = target_dir / name
        
        if not source.exists():
            logger.debug(f"Source {source} does not exist, skipping")
            continue
        
        if target.exists() and not force_copy:
            logger.debug(f"Target {target} exists, skipping (use force_copy=True to overwrite)")
            continue
        
        # Remove existing target if force_copy
        if target.exists():
            if is_dir:
                shutil.rmtree(target)
            else:
                target.unlink()
        
        # Copy resource
        logger.info(f"Copying {source} to {target}")
        if is_dir:
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    
    return target_dir


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    
    # Data settings
    data_dir: Path = field(default_factory=lambda: DATA_TEST_DIR)
    experiments_dir: Path = field(default_factory=lambda: EXPERIMENTS_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    save_prefix: str = "" # extra prefix for the saved file name
    
    # Run settings
    samples: Optional[List[str]] = None  # Specific samples to process
    replace: bool = False  # Replace existing HTML files
    skip_generation: bool = False  # Skip generation, only evaluate
    skip_evaluation: bool = False  # Skip evaluation, only generate
    
    # Evaluation settings
    save_rendered_image: bool = True
    replace_evaluation: bool = False  # If True, re-evaluate all; if False, skip already evaluated samples
    
    def __post_init__(self):
        """Convert paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.experiments_dir, str):
            self.experiments_dir = Path(self.experiments_dir)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)


class ExperimentBase(ABC):
    """
    Base class for experiment runners.
    
    Subclasses should implement:
    - experiment_name: Property returning the experiment name (e.g., "exp1")
    - description: Property returning a description of the experiment
    - get_run_configs: Method returning list of run configurations
    - generate_single: Method to generate HTML for a single sample
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or ExperimentConfig()
        ensure_output_dirs()
    
    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """Return the experiment name (e.g., 'exp1', 'exp2', 'exp3')."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the experiment."""
        pass
    
    @abstractmethod
    def get_run_configs(self) -> List[Dict[str, Any]]:
        """
        Return list of run configurations.
        
        Each configuration dict should contain parameters for a single experiment run.
        
        Returns:
            List of configuration dictionaries
        """
        pass
    
    @abstractmethod
    def generate_single(
        self,
        sample_dir: Path,
        run_config: Dict[str, Any]
    ) -> str:
        """
        Generate HTML for a single sample.
        
        Args:
            sample_dir: Directory containing sample data
            run_config: Configuration for this run
        
        Returns:
            Generated HTML string
        """
        pass
    
    @abstractmethod
    def get_output_filename(self, run_config: Dict[str, Any]) -> str:
        """
        Get output filename for a run configuration.
        
        Args:
            run_config: Configuration for this run
        
        Returns:
            Filename (without directory path)
        """
        pass
    
    def get_sample_keys(self) -> List[str]:
        """Get all sample keys from data directory."""
        if self.config.samples:
            return self.config.samples
        
        keys = []
        for item in self.config.data_dir.iterdir():
            if item.is_dir() and (item / "processed_metadata.json").exists():
                keys.append(item.name)
        return sorted(keys)
    
    def run_generation(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run generation for all samples with a specific configuration.
        
        Args:
            run_config: Configuration for this run
        
        Returns:
            Statistics dictionary with success/error counts
        """
        sample_keys = self.get_sample_keys()
        filename = self.get_output_filename(run_config)
        
        logger.info(f"Running generation: {filename}")
        logger.info(f"Processing {len(sample_keys)} samples...")
        
        progress = create_progress()
        task_id = progress.add_task(
            f"{self.experiment_name}: {filename}",
            total=len(sample_keys),
            status="Starting..."
        )
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        with progress:
            for sample_key in sample_keys:
                progress.update(task_id, status=f"Processing {sample_key}...")
                
                # Setup output directory
                output_dir = setup_sample_directory(
                    sample_key, 
                    self.config.data_dir,
                    self.config.experiments_dir
                )
                
                # Check if already exists
                html_path = output_dir / filename
                if html_path.exists() and not self.config.replace:
                    logger.debug(f"HTML already exists for {sample_key}, skipping.")
                    skipped_count += 1
                    progress.update(task_id, advance=1)
                    continue
                
                try:
                    html = self.generate_single(output_dir, run_config)
                    
                    # Save HTML
                    html_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    
                    logger.debug(f"Successfully saved HTML at {html_path}")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {sample_key}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    error_count += 1
                
                progress.update(task_id, advance=1)
        
        return {
            "success": success_count,
            "error": error_count,
            "skipped": skipped_count,
            "total": len(sample_keys)
        }
    
    def _get_evaluation_output_path(self, suffix: str = "", add_timestamp: bool = False) -> Path:
        """
        Get the output path for evaluation results (without saving).
        
        Args:
            suffix: Optional suffix for filename
            add_timestamp: Whether to add timestamp to filename
        
        Returns:
            Path to the evaluation results file
        """
        if add_timestamp:
            timestamp = "_" + time.strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = ""
        if suffix:
            filename = f"{self.config.save_prefix}{self.experiment_name}_{suffix}{timestamp}.csv"
        else:
            filename = f"{self.config.save_prefix}{self.experiment_name}{timestamp}.csv"
        
        return self.config.results_dir / filename

    def run_evaluation(self, run_configs: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Run evaluation for all generated HTML files with incremental evaluation support.
        
        Supports:
        - Loading existing evaluation results to avoid re-computation
        - Saving results after each sample_key is fully evaluated
        - Replace mode to force re-evaluation of all samples
        
        Args:
            run_configs: Optional list of run configurations to evaluate.
                        If None, evaluates all configurations from get_run_configs().
        
        Returns:
            DataFrame with evaluation results (includes failed generations with null metrics)
        """
        if run_configs is None:
            run_configs = self.get_run_configs()
        
        sample_keys = self.get_sample_keys()
        eval_config = EvaluationConfig(save_rendered_image=self.config.save_rendered_image)
        
        logger.info(f"Running evaluation for {self.experiment_name}")
        logger.info(f"Evaluating {len(sample_keys)} samples x {len(run_configs)} configurations")
        
        # Get output path for incremental saving
        output_path = self._get_evaluation_output_path()
        
        # Load existing results if not in replace mode
        existing_df = None
        evaluated_keys = set()
        if not self.config.replace_evaluation:
            existing_df = load_existing_results(output_path)
            evaluated_keys = get_evaluated_keys(existing_df)
            if evaluated_keys:
                logger.info(f"Found {len(evaluated_keys)} already evaluated (sample_key, result_name) pairs")
        else:
            logger.info("Replace evaluation mode: will re-evaluate all samples")
        
        rows = []
        total_tasks = len(sample_keys) * len(run_configs)
        
        progress = create_progress()
        task_id = progress.add_task(
            f"Evaluating {self.experiment_name}",
            total=total_tasks,
            status="Starting..."
        )
        
        # Track which sample_keys need saving
        samples_processed = 0
        
        with progress:
            for sample_key in sample_keys:
                sample_rows = []  # Rows for this sample_key
                
                # Setup sample directory and load reference
                sample_dir = self.config.experiments_dir / sample_key
                if not sample_dir.exists():
                    logger.warning(f"Sample directory not found: {sample_dir}")
                    # Record failed entries for all run_configs
                    for run_config in run_configs:
                        filename = self.get_output_filename(run_config)
                        result_name = Path(filename).stem
                        
                        # Skip if already evaluated
                        if is_already_evaluated(sample_key, result_name, evaluated_keys):
                            logger.debug(f"Skipping already evaluated: {sample_key}/{result_name}")
                            progress.update(task_id, advance=1)
                            continue
                        
                        row = {
                            "sample_key": sample_key,
                            "result_name": result_name,
                            "generation_failed": True,
                            "failure_reason": "sample_dir_not_found",
                            **self._extract_run_info(run_config),
                        }
                        sample_rows.append(row)
                        progress.update(task_id, advance=1)
                    
                    # Add sample_rows to main rows and save incrementally
                    if sample_rows:
                        rows.extend(sample_rows)
                        samples_processed += 1
                        self._save_incremental_results(output_path, existing_df, rows)
                    continue
                
                ref_img, width, height = load_reference_image(sample_dir)
                if ref_img is None:
                    logger.warning(f"Reference image not found for {sample_key}")
                    # Record failed entries for all run_configs
                    for run_config in run_configs:
                        filename = self.get_output_filename(run_config)
                        result_name = Path(filename).stem
                        
                        # Skip if already evaluated
                        if is_already_evaluated(sample_key, result_name, evaluated_keys):
                            logger.debug(f"Skipping already evaluated: {sample_key}/{result_name}")
                            progress.update(task_id, advance=1)
                            continue
                        
                        row = {
                            "sample_key": sample_key,
                            "result_name": result_name,
                            "generation_failed": True,
                            "failure_reason": "ref_image_not_found",
                            **self._extract_run_info(run_config),
                        }
                        sample_rows.append(row)
                        progress.update(task_id, advance=1)
                    
                    # Add sample_rows to main rows and save incrementally
                    if sample_rows:
                        rows.extend(sample_rows)
                        samples_processed += 1
                        self._save_incremental_results(output_path, existing_df, rows)
                    continue
                
                for run_config in run_configs:
                    filename = self.get_output_filename(run_config)
                    result_name = Path(filename).stem
                    html_path = sample_dir / filename
                    
                    progress.update(task_id, status=f"{sample_key}/{filename}")
                    
                    # Skip if already evaluated
                    if is_already_evaluated(sample_key, result_name, evaluated_keys):
                        logger.debug(f"Skipping already evaluated: {sample_key}/{result_name}")
                        progress.update(task_id, advance=1)
                        continue
                    
                    if not html_path.exists():
                        # Record failed generation with null metrics
                        logger.debug(f"HTML file not found: {html_path}")
                        row = {
                            "sample_key": sample_key,
                            "result_name": result_name,
                            "generation_failed": True,
                            "failure_reason": "html_not_generated",
                            **self._extract_run_info(run_config),
                        }
                        sample_rows.append(row)
                        progress.update(task_id, advance=1)
                        continue
                    
                    try:
                        metrics = evaluate_single(html_path, ref_img, width, height, eval_config)
                        row = {
                            "sample_key": sample_key,
                            "result_name": result_name,
                            "generation_failed": False,
                            "failure_reason": None,
                            **self._extract_run_info(run_config),
                            **metrics
                        }
                        sample_rows.append(row)
                    except Exception as e:
                        logger.error(f"Failed to evaluate {html_path}: {e}")
                        # Record evaluation failure
                        row = {
                            "sample_key": sample_key,
                            "result_name": result_name,
                            "generation_failed": False,
                            "failure_reason": f"evaluation_error: {str(e)}",
                            **self._extract_run_info(run_config),
                        }
                        sample_rows.append(row)
                    
                    progress.update(task_id, advance=1)
                
                # Add sample_rows to main rows and save incrementally after each sample_key
                if sample_rows:
                    rows.extend(sample_rows)
                    samples_processed += 1
                    self._save_incremental_results(output_path, existing_df, rows)
                    logger.debug(f"Saved incremental results after {sample_key} ({samples_processed} samples processed)")
        
        # Merge with existing results to get final DataFrame
        final_df = merge_results(existing_df, rows)
        
        logger.info(f"Evaluation complete: {len(rows)} new evaluations, {len(final_df)} total rows")
        
        return final_df
    
    def _save_incremental_results(
        self,
        output_path: Path,
        existing_df: Optional[pd.DataFrame],
        new_rows: list
    ) -> None:
        """
        Save incremental evaluation results by merging with existing results.
        
        Args:
            output_path: Path to save results
            existing_df: Existing DataFrame to merge with
            new_rows: List of new result rows
        """
        merged_df = merge_results(existing_df, new_rows)
        save_results(merged_df, output_path)
    
    def _extract_run_info(self, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract run information for result rows.
        
        Override in subclasses to add experiment-specific columns.
        
        Args:
            run_config: Run configuration
        
        Returns:
            Dictionary of column name -> value
        """
        return {}
    
    def save_evaluation_results(self, df: pd.DataFrame, suffix: str = "", add_timestamp: bool = False) -> Path:
        """
        Save evaluation results to CSV.
        
        Args:
            df: DataFrame with evaluation results
            suffix: Optional suffix for filename
        
        Returns:
            Path to saved CSV file
        """
        if add_timestamp:
            timestamp = "_" + time.strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = ""
        if suffix:
            filename = f"{self.config.save_prefix}{self.experiment_name}_{suffix}{timestamp}.csv"
        else:
            filename = f"{self.config.save_prefix}{self.experiment_name}{timestamp}.csv"
        
        output_path = self.config.results_dir / filename
        save_results(df, output_path)
        
        return output_path
    
    def run(self, run_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run the complete experiment (generation + evaluation).
        
        Args:
            run_configs: Optional list of specific run configurations.
                        If None, uses all configurations from get_run_configs().
        
        Returns:
            Dictionary with generation stats and evaluation results path
        """
        if run_configs is None:
            run_configs = self.get_run_configs()
        
        results = {
            "experiment": self.experiment_name,
            "generation_stats": [],
            "evaluation_results_path": None
        }
        
        # Run generation
        if not self.config.skip_generation:
            console.print(f"\n[bold blue]===== {self.experiment_name}: Generation =====[/bold blue]")
            for run_config in run_configs:
                stats = self.run_generation(run_config)
                results["generation_stats"].append({
                    "config": run_config,
                    "stats": stats
                })
                console.print(
                    f"  {self.get_output_filename(run_config)}: "
                    f"[green]Success: {stats['success']}[/green], "
                    f"[red]Error: {stats['error']}[/red], "
                    f"[dim]Skipped: {stats['skipped']}[/dim]"
                )
        
        # Run evaluation
        if not self.config.skip_evaluation:
            console.print(f"\n[bold blue]===== {self.experiment_name}: Evaluation =====[/bold blue]")
            df = self.run_evaluation(run_configs)
            
            if not df.empty:
                output_path = self.save_evaluation_results(df)
                results["evaluation_results_path"] = str(output_path)
                console.print(f"  Results saved to: {output_path}")
                
                # Print summary statistics and save merge results
                merge_df = self._print_evaluation_summary(df)
                if not merge_df.empty:
                    merge_path = self.save_merge_results(merge_df, output_path)
                    if merge_path:
                        results["merge_results_path"] = str(merge_path)
        
        return results
    
    def _print_evaluation_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Print comprehensive summary of evaluation results grouped by result_name and log key metrics.
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            DataFrame with aggregated statistics by result_name (merge table)
        """
        logger.info("=" * 60)
        logger.info("Evaluation Summary (Grouped by result_name)")
        logger.info("=" * 60)
        
        if df.empty:
            logger.warning("No evaluations to summarize.")
            return pd.DataFrame()
        
        # Get unique result_names
        if "result_name" not in df.columns:
            logger.warning("No 'result_name' column found in DataFrame.")
            return pd.DataFrame()
        
        result_names = df["result_name"].unique()
        logger.info(f"Total methods/variants: {len(result_names)}")
        
        # Define metric columns
        visual_cols = ["PSNR", "SSIM", "LPIPS", "MAE_px", "MSE_px", "CLIP", "MAE_embed", "DINOv2"]
        resp_cols = sorted([col for col in df.columns if col.startswith("RESP_")])
        maint_cols = sorted([col for col in df.columns if col.startswith("MAINT_")])
        
        # Collect aggregated statistics for merge table
        merge_rows = []
        
        for result_name in sorted(result_names):
            console.print(f"\n[bold cyan]{'='*50}[/bold cyan]")
            console.print(f"[bold cyan]Method: {result_name}[/bold cyan]")
            console.print(f"[bold cyan]{'='*50}[/bold cyan]")
            
            df_method = df[df["result_name"] == result_name]
            total_samples = len(df_method)
            
            # Initialize row for merge table
            row = {"result_name": result_name}
            
            # Extract additional info columns (model, method, ablation, etc.)
            info_cols = ["model", "method", "ablation"]
            for col in info_cols:
                if col in df_method.columns:
                    unique_vals = df_method[col].unique()
                    row[col] = unique_vals[0] if len(unique_vals) == 1 else str(list(unique_vals))
            
            # === Failure Rate Statistics ===
            console.print("\n[bold yellow]Generation Statistics:[/bold yellow]")
            if "generation_failed" in df_method.columns:
                failed_count = int(df_method["generation_failed"].sum())
                success_count = total_samples - failed_count
                failure_rate = (failed_count / total_samples * 100) if total_samples > 0 else 0
                success_rate = 100 - failure_rate
                
                row["total_samples"] = total_samples
                row["success_count"] = success_count
                row["failed_count"] = failed_count
                row["success_rate"] = round(success_rate, 2)
                
                logger.info(f"[{result_name}] Total: {total_samples}, Success: {success_count} ({success_rate:.1f}%), Failed: {failed_count} ({failure_rate:.1f}%)")
                
                # Filter to successful samples for metric calculation
                df_success = df_method[df_method["generation_failed"] == False]
            else:
                row["total_samples"] = total_samples
                row["success_count"] = total_samples
                row["failed_count"] = 0
                row["success_rate"] = 100.0
                logger.info(f"[{result_name}] Total samples: {total_samples}")
                df_success = df_method
            
            if df_success.empty:
                logger.warning(f"[{result_name}] No successful evaluations.")
                merge_rows.append(row)
                continue
            
            # === Visual Metrics ===
            available_visual_cols = [col for col in visual_cols if col in df_success.columns]
            
            if available_visual_cols:
                console.print("\n[bold green]Visual Similarity Metrics:[/bold green]")
                logger.info(f"[{result_name}] Visual Metrics:")
                
                for col in available_visual_cols:
                    values = df_success[col].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        row[f"{col}_mean"] = round(mean_val, 4)
                        row[f"{col}_std"] = round(std_val, 4)
                        logger.info(f"  {col}: {mean_val:.4f} +/- {std_val:.4f}")
            
            # === Code Quality - Responsiveness Metrics ===
            if resp_cols:
                console.print("\n[bold blue]Responsiveness Metrics:[/bold blue]")
                logger.info(f"[{result_name}] Responsiveness Metrics:")
                
                for col in resp_cols:
                    values = df_success[col].dropna()
                    if len(values) > 0:
                        # Check if column is boolean/binary
                        if values.dtype == bool or set(values.unique()).issubset({0, 1, True, False}):
                            true_rate = values.mean() * 100
                            row[f"{col}_rate"] = round(true_rate, 2)
                            logger.info(f"  {col}: {true_rate:.1f}%")
                        else:
                            mean_val = values.mean()
                            std_val = values.std()
                            row[f"{col}_mean"] = round(mean_val, 4)
                            row[f"{col}_std"] = round(std_val, 4)
                            logger.info(f"  {col}: {mean_val:.4f} +/- {std_val:.4f}")
            
            # === Code Quality - Maintainability Metrics ===
            if maint_cols:
                console.print("\n[bold magenta]Maintainability Metrics:[/bold magenta]")
                logger.info(f"[{result_name}] Maintainability Metrics:")
                
                for col in maint_cols:
                    values = df_success[col].dropna()
                    if len(values) > 0:
                        # Check if column is boolean/binary
                        if values.dtype == bool or set(values.unique()).issubset({0, 1, True, False}):
                            true_rate = values.mean() * 100
                            row[f"{col}_rate"] = round(true_rate, 2)
                            logger.info(f"  {col}: {true_rate:.1f}%")
                        else:
                            mean_val = values.mean()
                            std_val = values.std()
                            row[f"{col}_mean"] = round(mean_val, 4)
                            row[f"{col}_std"] = round(std_val, 4)
                            logger.info(f"  {col}: {mean_val:.4f} +/- {std_val:.4f}")
            
            merge_rows.append(row)
        
        logger.info("=" * 60)
        
        # Create merge DataFrame
        merge_df = pd.DataFrame(merge_rows)
        return merge_df
    
    def save_merge_results(self, merge_df: pd.DataFrame, base_output_path: Path) -> Path:
        """
        Save aggregated statistics to a merge CSV file.
        
        Args:
            merge_df: DataFrame with aggregated statistics by result_name
            base_output_path: Path to the base evaluation results CSV
        
        Returns:
            Path to saved merge CSV file
        """
        if merge_df.empty:
            logger.warning("No merge results to save")
            return None
        
        # Generate merge filename: exp1.csv -> exp1_merge.csv
        merge_filename = base_output_path.stem + "_merge" + base_output_path.suffix
        merge_path = base_output_path.parent / merge_filename
        
        merge_df.to_csv(merge_path, index=False, encoding="utf-8-sig")
        logger.info(f"Merge results saved to {merge_path}")
        console.print(f"  Merge results saved to: {merge_path}")
        
        return merge_path
