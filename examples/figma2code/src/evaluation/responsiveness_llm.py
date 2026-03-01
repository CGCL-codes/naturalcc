"""
LLM-based responsiveness evaluation for HTML pages.

This module provides tools to evaluate the responsive design quality of HTML pages
by rendering them at multiple viewport sizes and using an LLM to assess the results.

The evaluation produces a score from 1-5 based on:
- Layout adaptability across screen sizes
- Content readability at all sizes
- Element integrity (no truncation or overflow)
- Spacing consistency
- Functional usability of interactive elements

Usage:
    python -m src.evaluation.responsiveness_llm
"""

import re
import signal
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import tqdm
from PIL import Image

from ..configs.models import MODELS
from ..llm.openrouter import create_llm
from ..utils.console_logger import logger
from ..utils.html_screenshot import html2shot


# ============================================================================
# Constants and Configuration
# ============================================================================

class ViewportPreset(Enum):
    """Common viewport size presets for responsive testing."""
    DESKTOP_HD = (1920, 1080)      # Desktop Full HD
    LAPTOP = (1366, 768)           # Standard laptop
    TABLET_PORTRAIT = (768, 1024)  # iPad portrait
    MOBILE = (375, 667)            # iPhone SE


# Default viewport configurations for responsiveness testing
DEFAULT_VIEWPORTS: List[Tuple[int, int]] = [
    ViewportPreset.DESKTOP_HD.value,
    ViewportPreset.LAPTOP.value,
    ViewportPreset.TABLET_PORTRAIT.value,
    ViewportPreset.MOBILE.value,
]


RESPONSIVENESS_EVALUATION_PROMPT = """You are a professional front-end development expert skilled at evaluating web page responsive design quality.

I will show you screenshots of the same HTML page rendered at different screen sizes. Please evaluate the responsiveness quality of this page based on the following criteria and provide a score from 1-5:

Scoring Criteria:
- **5 (Excellent)**: The page adapts perfectly to all screen sizes with smooth and natural layout, appropriately adjusted element sizes and spacing, no overflow or overlap, and consistent excellent user experience
- **4 (Good)**: The page performs well on most screen sizes with minor layout issues that don't affect usability, overall good user experience
- **3 (Average)**: The page has basic responsive capability but shows obvious layout issues at certain screen sizes, such as element compression, text overflow, or improper spacing
- **2 (Poor)**: The page has limited responsive design with serious layout issues on small screens, some content may not display properly or requires horizontal scrolling
- **1 (Very Poor)**: The page has almost no responsive design, layout completely breaks down at different screen sizes with massive overflow or overlap, unusable

Evaluation Points:
1. Layout Adaptability: Whether elements can reasonably reflow based on screen size
2. Content Readability: Whether text is clearly readable at all sizes
3. Element Integrity: Whether images, buttons and other elements scale correctly without truncation or overflow
4. Spacing Consistency: Whether element spacing remains reasonable at all sizes
5. Functional Usability: Whether interactive elements are easy to click/operate at all sizes

The images correspond to the following screen sizes (width×height) in order: {viewport_sizes}

Please briefly analyze the page's performance at each screen size, then provide your final score.
Output format requirement: After completing your analysis, you must output the score on a separate line at the end in the format: **Score: X** (where X is an integer from 1-5)
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EvaluationResult:
    """Result of a single responsiveness evaluation."""
    sample_id: str
    html_prefix: str
    score: Optional[int]
    response: str
    error: Optional[str] = None


@dataclass 
class EvaluationState:
    """
    Manages evaluation state with support for interruption handling.
    
    This class tracks the progress of batch evaluations and provides
    graceful handling of interrupts (SIGINT/SIGTERM) to save partial results.
    """
    rows: List[Dict] = field(default_factory=list)
    output_dir: Optional[Path] = None
    interrupted: bool = False
    
    def save_results(self, force: bool = False) -> None:
        """
        Save current results to CSV file.
        
        Args:
            force: If True, save even if no new results
        """
        if not self.output_dir or (not self.rows and not force):
            return
        
        try:
            df = merge_results(self.output_dir, self.rows)
            if not df.empty:
                output_csv = self.output_dir / "results.csv"
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                logger.info(f"Results saved to {output_csv} ({len(df)} records)")
        except Exception as e:
            logger.error(f"Failed to save results on interrupt: {e}")
    
    def handle_interrupt(self, signum, frame) -> None:
        """
        Signal handler for graceful interrupt handling.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.interrupted = True
        print("\n\n[Interrupted] Ctrl+C detected, saving current results...")
        logger.warning("Evaluation interrupted by user (SIGINT)")
        self.save_results(force=True)
        print(f"[Interrupted] Saved {len(self.rows)} new evaluation results")
        sys.exit(0)


@dataclass
class ResponsivenessConfig:
    """Configuration for responsiveness evaluation."""
    viewports: List[Tuple[int, int]] = field(default_factory=lambda: DEFAULT_VIEWPORTS.copy())
    save_responses: bool = True
    save_renders: bool = True
    overwrite: bool = False
    model_name: str = "gpt5"
    max_tokens: int = 2000
    temperature: float = 0
    img_max_width: int = 1024


# ============================================================================
# Utility Functions
# ============================================================================

def get_render_image_path(
    output_dir: Path,
    sample_id: str,
    html_prefix: str,
    width: int,
    height: int
) -> Path:
    """
    Get the file path for a rendered screenshot.
    
    Args:
        output_dir: Base output directory
        sample_id: Sample identifier
        html_prefix: HTML file prefix (without extension)
        width: Viewport width
        height: Viewport height
    
    Returns:
        Path to the rendered image file
    """
    sample_dir = output_dir / sample_id
    return sample_dir / f"{html_prefix}__{width}_{height}.png"


def load_existing_results(output_dir: Path) -> Set[Tuple[str, str]]:
    """
    Load previously completed evaluation results.
    
    Args:
        output_dir: Directory containing results.csv
    
    Returns:
        Set of (sample_id, html_prefix) tuples for completed evaluations
    """
    results_csv = output_dir / "results.csv"
    if not results_csv.exists():
        return set()
    
    try:
        df = pd.read_csv(results_csv, encoding="utf-8-sig")
        if 'sample_id' in df.columns and 'html_prefix' in df.columns:
            # Only return records with valid scores
            valid_df = df[df['responsiveness_score'].notna()]
            return set(zip(valid_df['sample_id'], valid_df['html_prefix']))
    except Exception as e:
        logger.warning(f"Failed to load existing results: {e}")
    
    return set()


def merge_results(output_dir: Path, new_rows: List[Dict]) -> pd.DataFrame:
    """
    Merge new evaluation results with existing results.
    
    Args:
        output_dir: Directory containing results.csv
        new_rows: List of new result dictionaries
    
    Returns:
        Combined DataFrame with duplicates removed (keeping newest)
    """
    results_csv = output_dir / "results.csv"
    existing_df = None
    
    if results_csv.exists():
        try:
            existing_df = pd.read_csv(results_csv, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"Failed to load existing results for merge: {e}")
    
    new_df = pd.DataFrame(new_rows)
    
    if existing_df is not None and not existing_df.empty:
        # Merge with new results overwriting old ones
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['sample_id', 'html_prefix'],
            keep='last'
        )
        return combined
    
    return new_df


def parse_score_from_response(response: str) -> Optional[int]:
    """
    Extract the responsiveness score from LLM response text.
    
    Args:
        response: The model's response text
    
    Returns:
        Score (1-5), or None if parsing fails
    """
    # Try multiple patterns in order of specificity
    patterns = [
        r'\*\*[Ss]core[:：]?\s*(\d)\*\*',  # **Score: X**
        r'[Ss]core[:：]?\s*(\d)',           # Score: X
        r'(\d)\s*/\s*5',                    # X/5
        r'\*\*(\d)\*\*\s*/\s*5',            # **X**/5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    
    # Fallback: find the last 1-5 number in the response
    numbers = re.findall(r'\b([1-5])\b', response)
    if numbers:
        return int(numbers[-1])
    
    return None


def iter_html_files(folder: Path) -> List[Path]:
    """
    List all HTML files in a directory, sorted alphabetically.
    
    Args:
        folder: Directory to search
    
    Returns:
        Sorted list of HTML file paths
    """
    return sorted([p for p in folder.glob("*.html") if p.is_file()])


def format_viewport_description(viewports: List[Tuple[int, int]]) -> str:
    """
    Format viewport sizes for display in prompt.
    
    Args:
        viewports: List of (width, height) tuples
    
    Returns:
        Formatted string like "1920×1080, 1366×768, ..."
    """
    return ", ".join([f"{w}×{h}" for w, h in viewports])


# ============================================================================
# Core Evaluation Classes
# ============================================================================

class ResponsivenessEvaluator:
    """
    Evaluates HTML page responsiveness using LLM-based assessment.
    
    This class renders HTML pages at multiple viewport sizes and uses
    a vision-capable LLM to assess the responsive design quality.
    
    Example:
        evaluator = ResponsivenessEvaluator(model_name="gpt5")
        score, response = evaluator.evaluate("page.html")
        print(f"Responsiveness score: {score}/5")
    """
    
    def __init__(
        self,
        model_name: str = "gpt5",
        viewports: Optional[List[Tuple[int, int]]] = None,
        max_tokens: int = 2000,
        temperature: float = 0,
        img_max_width: int = 1024
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: LLM model name from MODELS config
            viewports: List of (width, height) viewport sizes to test
            max_tokens: Maximum tokens in LLM response
            temperature: LLM sampling temperature (0 = deterministic)
            img_max_width: Maximum image width for LLM input
        
        Raises:
            ValueError: If model_name is not found in MODELS
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        
        self.model_name = model_name
        self.viewports = viewports or DEFAULT_VIEWPORTS.copy()
        self.img_max_width = img_max_width
        
        # Initialize LLM client
        self.llm = create_llm(
            model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            img_max_width=img_max_width
        )
    
    def render_at_viewports(
        self,
        html_path: Union[str, Path],
        shot_dir: Optional[Path] = None,
        sample_id: Optional[str] = None,
        html_prefix: Optional[str] = None,
        overwrite: bool = False
    ) -> List[Image.Image]:
        """
        Render HTML at multiple viewport sizes.
        
        Args:
            html_path: Path to HTML file
            shot_dir: Directory to save/load rendered images
            sample_id: Sample identifier for file naming
            html_prefix: HTML file prefix for file naming
            overwrite: If True, re-render even if images exist
        
        Returns:
            List of PIL Images, one per viewport
        """
        html_path = Path(html_path)
        images = []
        save_images = shot_dir is not None and sample_id is not None and html_prefix is not None
        
        # Ensure sample directory exists
        if save_images:
            sample_dir = shot_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
        
        for width, height in self.viewports:
            img = None
            img_path = None
            
            if save_images:
                img_path = get_render_image_path(shot_dir, sample_id, html_prefix, width, height)
            
            # Try to load existing render if not in overwrite mode
            if img_path and img_path.exists() and not overwrite:
                try:
                    img = Image.open(img_path).convert("RGB")
                    logger.debug(f"Loaded existing render: {img_path}")
                except Exception as e:
                    logger.warning(f"Failed to load existing image {img_path}: {e}")
                    img = None
            
            # Render if no existing image
            if img is None:
                try:
                    img = html2shot(
                        html_file_path=html_path,
                        use_viewport=True,
                        viewport={'width': width, 'height': height}
                    )
                    if isinstance(img, Image.Image):
                        img = img.convert("RGB")
                        # Save rendered image
                        if img_path:
                            img.save(img_path, format='PNG', optimize=True)
                            logger.debug(f"Saved render: {img_path}")
                    else:
                        logger.warning(f"Failed to render {html_path} at {width}x{height}")
                        img = Image.new('RGB', (width, height), color='white')
                except Exception as e:
                    logger.error(f"Error rendering {html_path} at {width}x{height}: {e}")
                    img = Image.new('RGB', (width, height), color='white')
            
            images.append(img)
        
        return images
    
    def evaluate(
        self,
        html_path: Union[str, Path],
        shot_dir: Optional[Path] = None,
        sample_id: Optional[str] = None,
        html_prefix: Optional[str] = None,
        overwrite: bool = False
    ) -> Tuple[Optional[int], str]:
        """
        Evaluate the responsiveness of an HTML page.
        
        Args:
            html_path: Path to HTML file
            shot_dir: Directory for saving/loading renders
            sample_id: Sample identifier
            html_prefix: HTML file prefix
            overwrite: If True, re-render images
        
        Returns:
            Tuple of (score, raw_response) where score is 1-5 or None on failure
        """
        # Render at all viewport sizes
        images = self.render_at_viewports(
            html_path,
            shot_dir=shot_dir,
            sample_id=sample_id,
            html_prefix=html_prefix,
            overwrite=overwrite
        )
        
        # Build prompt with viewport information
        viewport_desc = format_viewport_description(self.viewports)
        prompt = RESPONSIVENESS_EVALUATION_PROMPT.format(viewport_sizes=viewport_desc)
        
        # Build multimodal content list
        texts_imgs = []
        for img, (width, height) in zip(images, self.viewports):
            texts_imgs.append(f"Screen size {width}×{height}:")
            texts_imgs.append(img)
        
        try:
            response = self.llm(prompt, texts_imgs)
            score = parse_score_from_response(response)
            return score, response
        except Exception as e:
            logger.error(f"Error evaluating {html_path}: {e}")
            return None, str(e)


# ============================================================================
# Batch Evaluation Functions
# ============================================================================

# Global evaluation state for interrupt handling
_eval_state = EvaluationState()


def evaluate_folder(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "gpt5",
    viewports: Optional[List[Tuple[int, int]]] = None,
    save_responses: bool = True,
    html_prefixes: Optional[List[str]] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Evaluate responsiveness for all HTML files in a folder structure.
    
    Expected folder structure:
        input_dir/
            sample1/
                prefix1.html
                prefix2.html
            sample2/
                prefix1.html
                prefix2.html
                ...
    
    Args:
        input_dir: Input directory containing sample folders
        output_dir: Output directory for results and renders
        model_name: LLM model name for evaluation
        viewports: List of viewport sizes to test
        save_responses: If True, save detailed LLM responses
        html_prefixes: Filter to specific HTML prefixes (None = all)
        overwrite: If True, overwrite existing evaluations
    
    Returns:
        DataFrame with evaluation results
    """
    global _eval_state
    
    viewports = viewports or DEFAULT_VIEWPORTS
    
    # Initialize evaluator
    evaluator = ResponsivenessEvaluator(
        model_name=model_name,
        viewports=viewports,
        max_tokens=2000,
        temperature=0,
        img_max_width=1024
    )
    
    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "responses"
    shot_dir = output_dir / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    if save_responses:
        responses_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize state for interrupt handling
    _eval_state.output_dir = output_dir
    _eval_state.rows = []
    _eval_state.interrupted = False
    
    # Load existing results for skip logic
    existing_results = set() if overwrite else load_existing_results(output_dir)
    if existing_results:
        logger.info(f"Found {len(existing_results)} existing evaluations, will skip them")
    
    skipped_count = 0
    evaluated_count = 0
    
    # Process all sample directories
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    try:
        for sample_dir in tqdm.tqdm(sample_dirs, desc="Evaluating samples"):
            if _eval_state.interrupted:
                break
            
            sample_id = sample_dir.name
            html_files = iter_html_files(sample_dir)
            
            if not html_files:
                logger.warning(f"No HTML files found in {sample_dir}")
                continue
            
            for html_path in html_files:
                if _eval_state.interrupted:
                    break
                
                html_prefix = html_path.stem
                
                # Filter by html_prefix if specified
                if html_prefixes is not None and html_prefix not in html_prefixes:
                    continue
                
                # Skip if already evaluated
                if (sample_id, html_prefix) in existing_results:
                    skipped_count += 1
                    logger.debug(f"Skipping {sample_id}/{html_prefix} (already evaluated)")
                    continue
                
                tqdm.tqdm.write(f"Evaluating {sample_id}/{html_prefix}...")
                
                score, response = evaluator.evaluate(
                    html_path,
                    shot_dir=shot_dir,
                    sample_id=sample_id,
                    html_prefix=html_prefix,
                    overwrite=overwrite
                )
                
                row = {
                    "sample_id": sample_id,
                    "html_prefix": html_prefix,
                    "responsiveness_score": score
                }
                _eval_state.rows.append(row)
                evaluated_count += 1
                
                # Save detailed response
                if save_responses:
                    response_file = responses_dir / f"{sample_id}_{html_prefix}.txt"
                    with open(response_file, 'w', encoding='utf-8') as f:
                        f.write(f"Sample: {sample_id}\n")
                        f.write(f"HTML: {html_prefix}\n")
                        f.write(f"Score: {score}\n")
                        f.write("=" * 50 + "\n")
                        f.write(response)
                
                # Incremental save (merge with existing results)
                df_temp = merge_results(output_dir, _eval_state.rows)
                df_temp.to_csv(output_dir / "results_temp.csv", index=False, encoding="utf-8-sig")
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by KeyboardInterrupt")
        _eval_state.save_results(force=True)
        raise
    
    logger.info(f"Evaluated: {evaluated_count}, Skipped: {skipped_count}")
    
    # Merge and return final results
    df = merge_results(output_dir, _eval_state.rows)
    return df


def run_evaluation(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = "gpt5",
    viewports: Optional[List[Tuple[int, int]]] = None,
    html_prefixes: Optional[List[str]] = None,
    overwrite: bool = False,
    save_responses: bool = True
) -> pd.DataFrame:
    """
    Run responsiveness evaluation with full result saving and summary output.
    
    This is the main entry point for batch evaluation, handling all setup,
    signal registration, and result persistence.
    
    Args:
        input_dir: Input directory containing sample folders
        output_dir: Output directory for results
        model_name: LLM model name
        viewports: Viewport sizes to test (None = defaults)
        html_prefixes: Filter to specific HTML prefixes (None = all)
        overwrite: If True, overwrite existing evaluations
        save_responses: If True, save detailed LLM responses
    
    Returns:
        DataFrame with evaluation results
    """
    global _eval_state
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    viewports = viewports or DEFAULT_VIEWPORTS
    
    # Register signal handlers for graceful interrupt
    signal.signal(signal.SIGINT, _eval_state.handle_interrupt)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _eval_state.handle_interrupt)
    
    # Validate input
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not html_prefixes:
        # initialize html_prefixes with all exp1 results in input_dir
        from ..exp_scripts.exp1_model_comparison import Exp1ModelComparison
        exp1 = Exp1ModelComparison()
        html_prefixes = [exp1.get_output_filename(run_config).replace(".html", "") for run_config in exp1.get_run_configs()]
    
    # Log configuration
    logger.info("Starting responsiveness evaluation")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Viewports: {[f'{w}x{h}' for w, h in viewports]}")
    logger.info(f"HTML prefixes: {html_prefixes if html_prefixes else 'all'}")
    logger.info(f"Overwrite mode: {overwrite}")
    
    try:
        df = evaluate_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=model_name,
            viewports=viewports,
            save_responses=save_responses,
            html_prefixes=html_prefixes,
            overwrite=overwrite
        )
        
        # Save final results
        output_csv = output_dir / "results.csv"
        if not df.empty:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info(f"Results saved to {output_csv}")
            
            # Print summary
            logger.info("\n=== Summary ===")
            logger.info(f"Total evaluations: {len(df)}")
            logger.info("Score distribution:")
            score_counts = df['responsiveness_score'].value_counts().sort_index()
            logger.info(score_counts)
            
            # Compute and save aggregated statistics by html_prefix
            if 'html_prefix' in df.columns:
                logger.info("\n=== Statistics by html_prefix ===")
                
                # Group by html_prefix and compute statistics
                merge_rows = []
                for prefix in sorted(df['html_prefix'].unique()):
                    df_prefix = df[df['html_prefix'] == prefix]
                    scores = df_prefix['responsiveness_score'].dropna()
                    
                    row = {
                        'html_prefix': prefix,
                        'count': len(df_prefix),
                        'score_mean': round(scores.mean(), 4) if len(scores) > 0 else None,
                        'score_std': round(scores.std(), 4) if len(scores) > 0 else None,
                        'score_min': scores.min() if len(scores) > 0 else None,
                        'score_max': scores.max() if len(scores) > 0 else None,
                    }
                    
                    # Add score distribution
                    for score_val in range(1, 6):  # Scores 1-5
                        count = (scores == score_val).sum()
                        row[f'score_{score_val}_count'] = count
                        row[f'score_{score_val}_rate'] = round(count / len(scores) * 100, 2) if len(scores) > 0 else 0
                    
                    merge_rows.append(row)
                    logger.info(f"  {prefix}: mean={row['score_mean']}, std={row['score_std']}, n={row['count']}")
                
                # Save merge results
                merge_df = pd.DataFrame(merge_rows)
                merge_csv = output_dir / "results_merge.csv"
                merge_df.to_csv(merge_csv, index=False, encoding="utf-8-sig")
                logger.info(f"\nMerge results saved to {merge_csv}")
        else:
            logger.warning("No evaluation results generated")
        
        # Clean up temporary file
        temp_csv = output_dir / "results_temp.csv"
        if temp_csv.exists():
            temp_csv.unlink()
        
        return df
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted, results saved")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        _eval_state.save_results(force=True)
        raise


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for responsiveness evaluation."""
    import argparse
    from ..configs.paths import enter_project_root, EXPERIMENTS_DIR, OUTPUT_DIR, LOGS_DIR
    from ..utils.console_logger import setup_logging
    
    enter_project_root()
    
    parser = argparse.ArgumentParser(
        description="Evaluate HTML responsiveness using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all exp1 results (default)
  python -m src.evaluation.responsiveness_llm
  
  # Evaluate specific models' outputs
  python -m src.evaluation.responsiveness_llm --html-prefix figma_image_direct__gpt4o figma_image_direct__claude_opus_4_1
  
  # Use different evaluation model
  python -m src.evaluation.responsiveness_llm --model gpt5
  
  # Custom viewports
  python -m src.evaluation.responsiveness_llm --viewports 1920x1080 768x1024
  
  # Overwrite existing evaluations
  python -m src.evaluation.responsiveness_llm --overwrite
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory path containing sample folders (default: output/experiments)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory path for results (default: output/resp_eval)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt5",
        choices=list(MODELS.keys()),
        help="LLM model to use for evaluation (default: gpt5)"
    )
    parser.add_argument(
        "--viewports",
        type=str,
        nargs="+",
        default=None,
        help="Viewport sizes to test, format WxH (e.g., 1920x1080 768x1024)"
    )
    parser.add_argument(
        "--html-prefix",
        type=str,
        nargs="+",
        default=None,
        help="HTML prefixes to evaluate (e.g., figma_image_direct__gpt4o), all if not specified"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing renders and evaluation results"
    )
    parser.add_argument(
        "--no-save-responses",
        action="store_true",
        help="Do not save detailed LLM responses"
    )
    
    args = parser.parse_args()
    
    # Set default paths
    input_dir = Path(args.input) if args.input else EXPERIMENTS_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR / "resp_eval"
    
    # Setup logging
    setup_logging(logger, output_dir=LOGS_DIR, log_name="resp_eval")
    
    # Parse viewport arguments
    viewports = None
    if args.viewports:
        viewports = []
        for vp in args.viewports:
            try:
                w, h = vp.lower().split('x')
                viewports.append((int(w), int(h)))
            except ValueError:
                logger.error(f"Invalid viewport format: {vp}. Expected WxH (e.g., 1920x1080)")
                sys.exit(1)
    
    # Run evaluation
    run_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=args.model,
        viewports=viewports,
        html_prefixes=args.html_prefix,
        overwrite=args.overwrite,
        save_responses=not args.no_save_responses
    )


if __name__ == "__main__":
    main()
