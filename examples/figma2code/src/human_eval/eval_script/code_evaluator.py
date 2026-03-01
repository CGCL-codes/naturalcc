"""
Code quality evaluation interface using Gradio.

Provides a web interface for human evaluation of generated HTML code quality.
"""

import json
import shutil
import socket
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import gradio as gr

def package_evaluation(
    experiment_dir: str,
    output_zip: str,
    method_name: str,
    script_path: Optional[str] = None,
    samples_key: Optional[List[str]] = None
) -> None:
    """
    Package HTML files for distribution.
    structure:
    ./
    ├── code/
    |   ├── <method_name>__<filekey>_<node_id>.html
    |   └── ...
    └── <script_file_name>.py
    
    Args:
        experiment_dir: Experiment output directory
        output_zip: Output ZIP file path
        method_name: Method name for filtering
        script_path: Script to include (current file if None)
        samples_key: List of sample keys to include
    """
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        output_path.unlink()
    
    with zipfile.ZipFile(output_zip, "w") as zf:
        exp_path = Path(experiment_dir)
        
        for sample_dir in exp_path.iterdir():
            if not sample_dir.is_dir() and samples_key is not None and sample_dir.name not in samples_key:
                continue
            
            html_file = sample_dir / f"{method_name}.html"
            if html_file.exists():
                archive_name = f"code/{method_name}__{sample_dir.name}.html"
                zf.write(html_file, archive_name)
        
        # Include script
        if script_path is None:
            script_path = __file__
        zf.write(script_path, Path(script_path).name)
    
    print(f"Packaged to {output_zip}")


def unzip_package(zip_file_path: str, output_dir: str) -> None:
    """
    Extract packaged zip to a directory.

    Args:
        zip_file_path: Path to the zip file
        output_dir: Directory to extract to (will be cleared if exists)
    """
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"Extracted to {output_dir}")


@dataclass
class EvaluationMetric:
    """Configuration for an evaluation metric."""
    name: str
    options: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    description: str = ""


# Default metrics for code evaluation
DEFAULT_CODE_METRICS = [
    EvaluationMetric("HTML Structural Clarity"),
    EvaluationMetric("Class Readability"),
]


class CodeEvaluator:
    """
    Evaluates HTML code quality through a Gradio interface.
    
    Manages file loading, score caching, and result persistence.
    """
    
    def __init__(
        self,
        code_dir: str,
        save_file: str = "results.json",
        metrics: Optional[List[EvaluationMetric]] = None
    ):
        """
        Initialize code evaluator.
        
        Args:
            code_dir: Directory containing HTML files
            save_file: File to save evaluation results
            metrics: List of evaluation metrics (uses defaults if None)
        """
        self.code_dir = Path(code_dir)
        self.save_file = Path(save_file)
        self.metrics = metrics or DEFAULT_CODE_METRICS
        self.metric_names = [m.name for m in self.metrics]
        
        if not self.code_dir.exists():
            raise ValueError(f"Code directory not found: {code_dir}")
        
        # Load HTML files
        self.html_files: List[Tuple[str, Path]] = [] # [(<sample_id>, <html_path>)]
        for html_path in sorted(self.code_dir.glob("*.html")):
            sample_id = html_path.stem
            self.html_files.append((sample_id, html_path))
        
        if not self.html_files:
            raise ValueError(f"No HTML files found in {code_dir}")
        
        # Initialize evaluation cache
        self.evaluation_cache: Dict[str, Dict[str, Optional[int]]] = {} # {<sample_id>: {<metric_name>: <score>}}
        self.file_save_flag: List[bool] = [False] * len(self.html_files)
        
        self._load_existing_results()
    
    def _load_existing_results(self) -> None:
        """Load existing evaluation results from save file."""
        # Initialize cache for all samples
        for sample_id, _ in self.html_files:
            self.evaluation_cache[sample_id] = {
                metric: None for metric in self.metric_names
            }
        
        # Load from file if exists
        if self.save_file.exists():
            try:
                with open(self.save_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                
                for sample_id, evaluation in saved_data.items():
                    if sample_id not in self.evaluation_cache:
                        continue
                    
                    all_filled = True
                    for metric in self.metric_names:
                        if metric in evaluation and evaluation[metric] is not None:
                            self.evaluation_cache[sample_id][metric] = evaluation[metric]
                        else:
                            all_filled = False
                    
                    if all_filled:
                        idx = self._get_file_index(sample_id)
                        if idx is not None:
                            self.file_save_flag[idx] = True
                            
            except Exception as e:
                print(f"Warning: Failed to load {self.save_file}: {e}")
    
    def _get_file_index(self, sample_id: str) -> Optional[int]:
        """Get file index for a sample ID."""
        for i, (sid, _) in enumerate(self.html_files):
            if sid == sample_id:
                return i
        return None
    
    @property
    def total_files(self) -> int:
        """Total number of files."""
        return len(self.html_files)
    
    @property
    def first_unsaved_index(self) -> int:
        """Index of first unsaved file."""
        for i, saved in enumerate(self.file_save_flag):
            if not saved:
                return i
        return self.total_files - 1
    
    def get_file_info(self, index: int) -> Tuple[str, Dict[str, Any], str]:
        """
        Get file information for display.
        
        Args:
            index: File index
        
        Returns:
            Tuple of (html_content, evaluation_dict, file_info_string)
        """
        if not 0 <= index < self.total_files:
            raise IndexError(f"File index {index} out of range")
        
        sample_id, html_path = self.html_files[index]
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        evaluation = self.evaluation_cache[sample_id].copy()
        file_info = f"File {index + 1}/{self.total_files}"
        
        return html_content, evaluation, file_info
    
    def save_evaluation(
        self,
        index: int,
        evaluation: Dict[str, int]
    ) -> Tuple[bool, str]:
        """
        Save evaluation for a file.
        
        Args:
            index: File index
            evaluation: Dictionary mapping metric names to scores
        
        Returns:
            Tuple of (success, message)
        """
        # Validate all metrics filled
        missing = [m for m in self.metric_names if evaluation.get(m) is None]
        if missing:
            return False, f"Please fill in: {', '.join(missing)}"
        
        if not 0 <= index < self.total_files:
            return False, "File index out of range"
        
        sample_id, _ = self.html_files[index]
        
        # Update cache
        self.evaluation_cache[sample_id] = evaluation.copy()
        self.file_save_flag[index] = True
        
        # Save to file
        try:
            with open(self.save_file, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_cache, f, ensure_ascii=False, indent=2)
            return True, f"File {index + 1} saved successfully"
        except Exception as e:
            return False, f"Failed to save: {e}"


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def create_code_evaluation_interface(
    code_dir: str,
    save_file: str = "results.json",
    metrics: Optional[List[EvaluationMetric]] = None,
    start_index: int = -1
) -> gr.Blocks:
    """
    Create Gradio interface for code evaluation.
    
    Args:
        code_dir: Directory containing HTML files
        save_file: File to save results
        metrics: Evaluation metrics
        start_index: Starting file index (-1 for auto)
    
    Returns:
        Gradio Blocks application
    """
    evaluator = CodeEvaluator(code_dir, save_file, metrics)
    
    if start_index < 0:
        start_index = evaluator.first_unsaved_index
    
    def load_file(index: int):
        return evaluator.get_file_info(index)
    
    def navigate(current_index: int, eval_dict: dict, direction: int):
        """Navigate to previous/next file."""
        success, message = evaluator.save_evaluation(current_index, eval_dict)
        
        if direction == 0:  # Previous
            is_boundary = current_index == 0
            new_index = current_index if is_boundary or not success else current_index - 1
        else:  # Next
            is_boundary = current_index == evaluator.total_files - 1
            new_index = current_index if is_boundary or not success else current_index + 1
        
        if is_boundary and success:
            message = f"Boundary reached. {message}"
        
        html, eval_data, file_info = load_file(new_index)
        return html, file_info, eval_data, new_index, message
    
    def update_metric(eval_dict: dict, metric: str, value: int):
        eval_dict[metric] = value
        return eval_dict
    
    with gr.Blocks(title="Code Quality Evaluation", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 100% !important;
        }
        </style>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                html_display = gr.Code(
                    label="📝 HTML Source Code",
                    language="html",
                    max_lines=30,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                with gr.Row():
                    file_info_box = gr.Textbox(label="📄 Progress", interactive=False)
                    message_box = gr.Textbox(label="📊 Output Info", interactive=False)
                
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", variant="primary")
                    next_btn = gr.Button("➡️ Next", variant="primary")
                
                radio_widgets = {}
                for metric in evaluator.metrics:
                    radio_widgets[metric.name] = gr.Radio(
                        label=metric.name,
                        choices=metric.options,
                        value=None
                    )
        
        # State
        file_index = gr.State(start_index)
        eval_dict_state = gr.State({})
        
        # Load initial file
        demo.load(
            fn=lambda: load_file(start_index),
            outputs=[html_display, file_info_box, eval_dict_state]
        ).then(
            fn=lambda: start_index,
            outputs=[file_index]
        ).then(
            fn=lambda d: tuple(d.get(m.name) for m in evaluator.metrics),
            inputs=[eval_dict_state],
            outputs=list(radio_widgets.values())
        )
        
        # Navigation
        for btn, direction in [(prev_btn, 0), (next_btn, 1)]:
            btn.click(
                fn=lambda i, d, dir=direction: navigate(i, d, dir),
                inputs=[file_index, eval_dict_state],
                outputs=[html_display, file_info_box, eval_dict_state, file_index, message_box]
            ).then(
                fn=lambda d: tuple(d.get(m.name) for m in evaluator.metrics),
                inputs=[eval_dict_state],
                outputs=list(radio_widgets.values())
            )
        
        # Metric updates
        for metric in evaluator.metrics:
            radio_widgets[metric.name].change(
                fn=lambda d, v, m=metric.name: update_metric(d, m, v),
                inputs=[eval_dict_state, radio_widgets[metric.name]],
                outputs=[eval_dict_state]
            )
    
    return demo


def launch_code_evaluator(
    code_dir: str = "./code",
    save_file: str = "results.json",
    host: str = "127.0.0.1",
    port: int = 0
) -> None:
    """
    Launch the code evaluation interface.
    
    Args:
        code_dir: Directory containing HTML files
        save_file: File to save results
        host: Server host
        port: Server port (0 for auto)
    """
    if port == 0:
        port = find_free_port()
    
    print(f"Starting Code Evaluation at http://{host}:{port}")
    
    demo = create_code_evaluation_interface(code_dir, save_file)
    demo.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="store_true", help="package code")
    parser.add_argument("--eval", action="store_true", help="evaluate code")
    parser.add_argument("--save_file", type=str, default="results.json", help="save file name")

    args = parser.parse_args()

    if args.package == args.eval:
        parser.error("Exactly one of --package or --eval must be specified")
    if args.package:
        from ...configs.paths import enter_project_root, EXPERIMENTS_DIR
        enter_project_root()
        zip_file_path = "./output/human_eval/gradio_output/human_eval_code.zip"
        method_name = "exp1__ernie4_5_vl_424b_a47b"

        # filter sample key
        import os
        keys = []
        sample_dir = "./output/rebuttal_80"
        for dir_name in os.listdir(sample_dir):
            if os.path.isdir(os.path.join(sample_dir, dir_name)):
                keys.append(dir_name)
        filter_dir = "./output/sampled_rebuttal_1"
        for dir_name in os.listdir(filter_dir):
            if os.path.isdir(os.path.join(filter_dir, dir_name)) and dir_name in keys:
                keys.remove(dir_name)
        print(f"Found {len(keys)} keys")

        package_evaluation(str(EXPERIMENTS_DIR), zip_file_path, method_name, samples_key=keys)
        unzip_package(zip_file_path, "./output/human_eval/gradio_output/human_eval_code/")
    elif args.eval:
        launch_code_evaluator(save_file=args.save_file)