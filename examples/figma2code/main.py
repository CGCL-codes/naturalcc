#!/usr/bin/env python3
"""
Figma2Code CLI - Unified interface for running experiments.

This CLI provides a unified interface for running Figma-to-Code experiments:
- exp1: Model comparison (testing different LLM models)
- exp2: Method comparison (testing different generation methods)
- exp3: Ablation study (testing importance of different input components)

Each experiment runs both generation and evaluation phases, and saves results.

Usage:
    # Run exp1 with default models
    python main.py exp1

    # Run exp1 anonther time with save_prefix (Inference Stability and Reproducibility)
    python main.py exp1 --save-prefix run2_
    
    # Run exp1 with specific models
    python main.py exp1 --models gpt4o claude_opus_4_1
    
    # Run exp2 with default methods
    python main.py exp2

    # Run exp2 with guidelines (robustness to explicit prompt perturbations)
    python main.py exp2 --add-guidelines
    
    # Run exp2 with specific methods and model
    python main.py exp2 --methods image_direct figma_direct --model gpt4o
    
    # Run exp3 ablation study
    python main.py exp3
    
    # Run exp3 with specific ablation types
    python main.py exp3 --ablations geometry style --methods figma_direct
    
    # Common options
    python main.py exp1 --samples sample1 sample2  # Process specific samples
    python main.py exp1 --replace                  # Replace existing HTML files
    python main.py exp1 --skip-generation          # Only run evaluation
    python main.py exp1 --skip-evaluation          # Only run generation
    python main.py exp1 --save-prefix run2_         # Extra prefix for the saved file name
    
    # List available options
    python main.py list models
    python main.py list methods
    python main.py list ablations
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.configs import (
    MODELS,
    enter_project_root,
    ensure_output_dirs,
)
from src.utils.console_logger import logger, setup_logging, console
from src.generation import list_ablation_types

# Import experiment functions and constants
from src.exp_scripts.exp1_model_comparison import run_exp1, DEFAULT_MODELS
from src.exp_scripts.exp2_method_comparison import (
    run_exp2, 
    DEFAULT_METHODS as EXP2_DEFAULT_METHODS,
    DEFAULT_MODEL as EXP2_DEFAULT_MODEL,
    METHODS,
)
from src.exp_scripts.exp3_ablation import (
    run_exp3,
    DEFAULT_ABLATION_TYPES,
    DEFAULT_MODEL as EXP3_DEFAULT_MODEL,
    DEFAULT_METHODS as EXP3_DEFAULT_METHODS,
    ABLATION_METHODS,
)


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Figma2Code CLI - Run experiments and evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run exp1 (model comparison)
  python main.py exp1
  python main.py exp1 --models gpt4o claude_opus_4_1
  
  # Run exp1 another time with save_prefix (Inference Stability and Reproducibility)
  python main.py exp1 --models gpt4o --save-prefix run2_
  
  # Run exp2 (method comparison)
  python main.py exp2
  python main.py exp2 --methods image_direct figma_direct --model gpt4o
  
  # Run exp2 with guidelines (robustness to explicit prompt perturbations)
  python main.py exp2 --add-guidelines
  
  # Run exp3 (ablation study)
  python main.py exp3
  python main.py exp3 --ablations geometry style --methods figma_direct
  
  # List available options
  python main.py list models
  python main.py list methods
  python main.py list ablations
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ==================== exp1: Model Comparison ====================
    exp1_parser = subparsers.add_parser(
        "exp1", 
        help="Run exp1: Model comparison experiment"
    )
    exp1_parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=None,
        help=f"Models to test. Default: {', '.join(DEFAULT_MODELS)}"
    )
    _add_common_args(exp1_parser)
    
    # ==================== exp2: Method Comparison ====================
    exp2_parser = subparsers.add_parser(
        "exp2",
        help="Run exp2: Method comparison experiment"
    )
    exp2_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help=f"Methods to test. Default: {', '.join(EXP2_DEFAULT_METHODS)}"
    )
    exp2_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=f"Model to use. Default: {EXP2_DEFAULT_MODEL}"
    )
    exp2_parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=5,
        help="Max refinement steps for agent method (default: 5)"
    )
    exp2_parser.add_argument(
        "--add-guidelines",
        action="store_true",
        help="Add generation guidelines to prompt (for robustness testing). Only applies to figma_direct and figma_image_direct methods."
    )
    _add_common_args(exp2_parser)
    
    # ==================== exp3: Ablation Study ====================
    exp3_parser = subparsers.add_parser(
        "exp3",
        help="Run exp3: Ablation study experiment"
    )
    exp3_parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=None,
        help=f"Ablation types to test. Default: {', '.join(DEFAULT_ABLATION_TYPES)}"
    )
    exp3_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=list(ABLATION_METHODS.keys()),
        help=f"Methods to use. Default: {', '.join(EXP3_DEFAULT_METHODS)}"
    )
    exp3_parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help=f"Model to use. Default: {EXP3_DEFAULT_MODEL}"
    )
    _add_common_args(exp3_parser)
    
    # ==================== list: List available options ====================
    list_parser = subparsers.add_parser(
        "list",
        help="List available options"
    )
    list_parser.add_argument(
        "what",
        choices=["models", "methods", "ablations", "experiments"],
        help="What to list"
    )
    
    return parser


def _add_common_args(parser):
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: data/data_test)"
    )
    parser.add_argument(
        "--samples",
        type=str,
        nargs="+",
        default=None,
        help="Specific sample keys to process"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing HTML files"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation, only run evaluation"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation, only run generation"
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="",
        help="Extra prefix for the saved file name"
    )
    parser.add_argument(
        "--replace-evaluation",
        action="store_true",
        help="Re-evaluate all samples (default: skip already evaluated samples)"
    )


def handle_exp1(args):
    """Handle exp1 command."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Experiment 1: Model Comparison[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    models = args.models
    console.print(f"Models: {', '.join(models) if models else 'default'}")
    if args.save_prefix:
        console.print(f"Save prefix: {args.save_prefix}")
    if args.replace_evaluation:
        console.print("[yellow]Replace evaluation mode: will re-evaluate all samples[/yellow]")
    console.print(f"Data directory: {args.data_dir or 'default'}")
    
    return run_exp1(
        models=models,
        data_dir=args.data_dir,
        samples=args.samples,
        replace=args.replace,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
        save_prefix=args.save_prefix,
        replace_evaluation=args.replace_evaluation,
    )


def handle_exp2(args):
    """Handle exp2 command."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Experiment 2: Method Comparison[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    console.print(f"Model: {args.model or EXP2_DEFAULT_MODEL}")
    console.print(f"Methods: {', '.join(args.methods) if args.methods else 'default'}")
    if args.add_guidelines:
        console.print("[yellow]Guidelines mode enabled (testing robustness to prompt perturbations)[/yellow]")
    if args.save_prefix:
        console.print(f"Save prefix: {args.save_prefix}")
    if args.replace_evaluation:
        console.print("[yellow]Replace evaluation mode: will re-evaluate all samples[/yellow]")
    console.print(f"Data directory: {args.data_dir or 'default'}")
    
    return run_exp2(
        model=args.model,
        methods=args.methods,
        data_dir=args.data_dir,
        samples=args.samples,
        replace=args.replace,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
        agent_max_steps=args.agent_max_steps,
        save_prefix=args.save_prefix,
        add_guidelines=args.add_guidelines,
        replace_evaluation=args.replace_evaluation,
    )


def handle_exp3(args):
    """Handle exp3 command."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Experiment 3: Ablation Study[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    
    console.print(f"Model: {args.model or EXP3_DEFAULT_MODEL}")
    console.print(f"Methods: {', '.join(args.methods) if args.methods else 'default'}")
    console.print(f"Ablation types: {', '.join(args.ablations) if args.ablations else 'default'}")
    if args.save_prefix:
        console.print(f"Save prefix: {args.save_prefix}")
    if args.replace_evaluation:
        console.print("[yellow]Replace evaluation mode: will re-evaluate all samples[/yellow]")
    console.print(f"Data directory: {args.data_dir or 'default'}")
    
    return run_exp3(
        model=args.model,
        methods=args.methods,
        ablation_types=args.ablations,
        data_dir=args.data_dir,
        samples=args.samples,
        replace=args.replace,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
        save_prefix=args.save_prefix,
        replace_evaluation=args.replace_evaluation,
    )


def handle_list(args):
    """Handle list command."""
    if args.what == "models":
        console.print("\n[bold]Available Models:[/bold]")
        for name, config in MODELS.items():
            is_default = " [dim](default for exp1)[/dim]" if name in DEFAULT_MODELS else ""
            console.print(f"  {name}: {config['model']}{is_default}")
    
    elif args.what == "methods":
        console.print("\n[bold]exp2 Methods:[/bold]")
        for name, info in METHODS.items():
            is_default = " [dim](default)[/dim]" if name in EXP2_DEFAULT_METHODS else ""
            console.print(f"  {name}: {info['description']}{is_default}")
        
        console.print("\n[bold]exp3 Ablation Methods:[/bold]")
        for name, info in ABLATION_METHODS.items():
            console.print(f"  {name}: {info['description']}")
    
    elif args.what == "ablations":
        console.print("\n[bold]Ablation Types:[/bold]")
        ablation_descriptions = {
            "geometry": "Remove layout information (x, y, width, height, transforms)",
            "style": "Remove visual styles (colors, fonts, effects)",
            "image_refs": "Remove image references",
            "structure": "Flatten node hierarchy",
            "text": "Remove text content",
        }
        for name in list_ablation_types():
            desc = ablation_descriptions.get(name, "")
            console.print(f"  {name}: {desc}")
    
    elif args.what == "experiments":
        console.print("\n[bold]Available Experiments:[/bold]")
        console.print("  exp1: Model comparison - Test different LLM models")
        console.print("        Options: --models, --save-prefix")
        console.print("        Example: python main.py exp1 --models gpt4o claude_opus_4_1")
        console.print("")
        console.print("  exp2: Method comparison - Test different generation methods")
        console.print("        Options: --methods, --model, --add-guidelines, --save-prefix")
        console.print("        Example: python main.py exp2 --add-guidelines")
        console.print("")
        console.print("  exp3: Ablation study - Test importance of input components")
        console.print("        Options: --methods, --ablations, --model, --save-prefix")
        console.print("        Example: python main.py exp3 --ablations geometry style")


def main():
    """Main entry point."""
    # Enter project root directory
    enter_project_root()
    
    # Ensure output directories exist
    ensure_output_dirs()
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Handle list command (no logging needed)
    if args.command == "list":
        handle_list(args)
        return
    
    # Setup logging for experiment commands
    prefix = getattr(args, 'save_prefix', '') or ''
    if not prefix and args.add_guidelines and args.command == "exp2":
        prefix = "guide_"
    log_name = f"{prefix}{args.command}"
    setup_logging(logger, log_name=log_name)
    
    # Run the appropriate experiment
    try:
        if args.command == "exp1":
            results = handle_exp1(args)
        elif args.command == "exp2":
            results = handle_exp2(args)
        elif args.command == "exp3":
            results = handle_exp3(args)
        else:
            parser.print_help()
            return
        
        # Print final summary
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Experiment completed![/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        
        if results and results.get("evaluation_results_path"):
            console.print(f"Results saved to: {results['evaluation_results_path']}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        logger.exception("Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
