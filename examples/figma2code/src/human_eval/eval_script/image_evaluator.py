"""
Image similarity evaluation interface using Gradio.

Provides a web interface for human evaluation of visual similarity
between Figma design images and generated code renderings.

Usage:
* Server: `python -m src.human_eval.image_evaluator --package` to package image pairs
* Local: Evaluator extracts the zip, then runs
  `python -m src.human_eval.image_evaluator --eval [--save_file <save_file>]`

Result format:
    {
        "<method_name>__<filekey>_<node_id>": <score>,  # 1-5 similarity score
        ...
    }
"""

import json
import shutil
import socket
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import gradio as gr


# Similarity score labels (1=lowest, 5=highest)
SIMILARITY_LABELS = [
    "Highly Dissimilar",
    "Dissimilar",
    "Moderately Similar",
    "Similar",
    "Highly Similar",
]

SCORE_OPTIONS = [1, 2, 3, 4, 5]


def package_evaluation(
    experiment_dir: str,
    output_zip: str,
    method_name: str,
    script_path: Optional[str] = None,
    samples_key: Optional[List[str]] = None
) -> None:
    """
    Package image pairs for distribution.

    Renders HTML files to images and packages them alongside reference images.

    Structure:
    ./
    ├── image_pairs/
    |   ├── <method_name>__<filekey>_<node_id>/
    |   |   ├── gold_image.png
    |   |   └── generated_image.png
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

    exp_path = Path(experiment_dir)

    with zipfile.ZipFile(output_zip, "w") as zf:
        for sample_dir in sorted(exp_path.iterdir()):
            if not sample_dir.is_dir():
                continue
            if samples_key is not None and sample_dir.name not in samples_key:
                continue

            # Find reference image
            gold_image = sample_dir / "root.png"
            if not gold_image.exists():
                print(f"Gold image not found: {gold_image}")
                continue

            # Find HTML file for rendering
            html_file = sample_dir / f"{method_name}.html"
            if not html_file.exists():
                print(f"HTML file not found: {html_file}")
                continue

            try:
                # Get dimensions from metadata
                json_path = sample_dir / "processed_metadata.json"
                if json_path.exists():
                    with open(json_path, "r") as f:
                        meta_data = json.load(f)
                    bbox = meta_data.get("document", {}).get("absoluteBoundingBox", {})
                    width = int(bbox.get("width", 1200))
                    height = int(bbox.get("height", 800))
                else:
                    width, height = 1200, 800

                from ...evaluation.evaluator import render_html_to_image
                generated_image = render_html_to_image(html_file, width, height)

                # Save rendered image to temp location
                temp_image_path = sample_dir / f"{method_name}_temp.png"
                generated_image.save(str(temp_image_path))

                # Add to zip
                pair_name = f"{method_name}__{sample_dir.name}"
                zf.write(str(gold_image), f"image_pairs/{pair_name}/gold_image.png")
                zf.write(str(temp_image_path), f"image_pairs/{pair_name}/generated_image.png")

                # Clean up temp file
                temp_image_path.unlink()
            except Exception as e:
                print(f"Failed to render HTML {html_file}: {e}")
                continue

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
class ImagePair:
    """A pair of images for comparison."""
    sample_id: str
    reference_path: Path
    generated_path: Path


class ImageEvaluator:
    """
    Evaluates visual similarity through a Gradio interface.

    Shows reference (gold) and generated images side by side for scoring.
    Manages file loading, score caching, and result persistence.
    """

    def __init__(
        self,
        image_pairs_dir: str,
        save_file: str = "results.json",
    ):
        """
        Initialize image evaluator.

        Args:
            image_pairs_dir: Directory containing image pair folders
            save_file: File to save evaluation results
        """
        self.image_pairs_dir = Path(image_pairs_dir)
        self.save_file = Path(save_file)

        if not self.image_pairs_dir.exists():
            raise ValueError(f"Image pairs directory not found: {image_pairs_dir}")

        # Find image pairs
        self.image_pairs: List[ImagePair] = []  # [(sample_id, ref_path, gen_path)]
        self._find_image_pairs()

        if not self.image_pairs:
            raise ValueError(f"No valid image pairs found in {image_pairs_dir}")

        # Initialize evaluation cache
        self.evaluation_cache: Dict[str, Optional[int]] = {}  # {<sample_id>: <score>}
        self.file_save_flag: List[bool] = [False] * len(self.image_pairs)

        self._load_existing_results()

    def _find_image_pairs(self) -> None:
        """Find valid reference-generated image pairs in directory."""
        for sample_dir in sorted(self.image_pairs_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            ref_path = sample_dir / "gold_image.png"
            gen_path = sample_dir / "generated_image.png"

            if ref_path.exists() and gen_path.exists():
                self.image_pairs.append(ImagePair(
                    sample_id=sample_dir.name,
                    reference_path=ref_path,
                    generated_path=gen_path,
                ))
            else:
                print(f"Warning: Missing images in {sample_dir}")

    def _load_existing_results(self) -> None:
        """Load existing evaluation results from save file."""
        # Initialize cache for all samples
        for pair in self.image_pairs:
            self.evaluation_cache[pair.sample_id] = None

        # Load from file if exists
        if self.save_file.exists():
            try:
                with open(self.save_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)

                for sample_id, score in saved_data.items():
                    if sample_id in self.evaluation_cache and score is not None:
                        self.evaluation_cache[sample_id] = score
                        idx = self._get_pair_index(sample_id)
                        if idx is not None:
                            self.file_save_flag[idx] = True

            except Exception as e:
                print(f"Warning: Failed to load {self.save_file}: {e}")

    def _get_pair_index(self, sample_id: str) -> Optional[int]:
        """Get index for a sample ID."""
        for i, pair in enumerate(self.image_pairs):
            if pair.sample_id == sample_id:
                return i
        return None

    @property
    def total_pairs(self) -> int:
        """Total number of image pairs."""
        return len(self.image_pairs)

    @property
    def first_unsaved_index(self) -> int:
        """Index of first unsaved pair."""
        for i, saved in enumerate(self.file_save_flag):
            if not saved:
                return i
        return self.total_pairs - 1

    def get_pair_info(self, index: int) -> Tuple[str, str, Optional[int], str]:
        """
        Get image pair information for display.

        Args:
            index: Pair index

        Returns:
            Tuple of (reference_image_path, generated_image_path, current_score, info_string)
        """
        if not 0 <= index < self.total_pairs:
            raise IndexError(f"Pair index {index} out of range")

        pair = self.image_pairs[index]
        score = self.evaluation_cache[pair.sample_id]
        info_string = f"File {index + 1}/{self.total_pairs}"

        return str(pair.reference_path), str(pair.generated_path), score, info_string

    def save_evaluation(
        self,
        index: int,
        score: Optional[int]
    ) -> Tuple[bool, str]:
        """
        Save evaluation score for a pair.

        Args:
            index: Pair index
            score: Similarity score

        Returns:
            Tuple of (success, message)
        """
        if score is None:
            return False, "Please select a similarity score"

        if not 0 <= index < self.total_pairs:
            return False, "Index out of range"

        pair = self.image_pairs[index]

        # Update cache
        self.evaluation_cache[pair.sample_id] = score
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


def create_image_evaluation_interface(
    image_pairs_dir: str,
    save_file: str = "results.json",
    start_index: int = -1
) -> gr.Blocks:
    """
    Create Gradio interface for image similarity evaluation.

    Args:
        image_pairs_dir: Directory containing image pair folders
        save_file: File to save results
        start_index: Starting index (-1 for auto)

    Returns:
        Gradio Blocks application
    """
    evaluator = ImageEvaluator(image_pairs_dir, save_file)

    if start_index < 0:
        start_index = evaluator.first_unsaved_index

    # Build labeled choices for radio
    labeled_choices = [
        (SIMILARITY_LABELS[i], SCORE_OPTIONS[i])
        for i in range(len(SCORE_OPTIONS))
    ]

    def load_pair(index: int):
        return evaluator.get_pair_info(index)

    def navigate(current_index: int, score: Optional[int], direction: int):
        """Navigate to previous/next pair."""
        success, message = evaluator.save_evaluation(current_index, score)

        if direction == 0:  # Previous
            is_boundary = current_index == 0
            new_index = current_index if is_boundary or not success else current_index - 1
        else:  # Next
            is_boundary = current_index == evaluator.total_pairs - 1
            new_index = current_index if is_boundary or not success else current_index + 1

        if is_boundary and success:
            message = f"Boundary reached. {message}"

        ref_path, gen_path, new_score, info = load_pair(new_index)
        return ref_path, gen_path, new_score, info, new_index, message

    with gr.Blocks(title="Image Similarity Evaluation", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 100% !important;
        }
        </style>
        """)

        with gr.Column(scale=1):
            # Top section: navigation and scoring
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        file_info_box = gr.Textbox(
                            label="📄 Progress",
                            interactive=False,
                            value=""
                        )
                        message_box = gr.Textbox(
                            label="📊 Output Info",
                            interactive=False,
                            value=""
                        )
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Previous", variant="primary", size="sm")
                        next_btn = gr.Button("➡️ Next", variant="primary", size="sm")

                with gr.Column(scale=1):
                    similarity_radio = gr.Radio(
                        label="Similarity Score",
                        info="Rate the similarity between the Figma design (gold) and the generated result",
                        choices=labeled_choices,
                        value=None
                    )

            # Bottom section: image pair display
            with gr.Row():
                with gr.Column(scale=1):
                    gold_image_display = gr.Image(
                        label="🎨 Figma Design (Gold Image)",
                        type="filepath",
                        interactive=False,
                        height=500
                    )
                with gr.Column(scale=1):
                    generated_image_display = gr.Image(
                        label="🖼️ Generated Code Rendering",
                        type="filepath",
                        interactive=False,
                        height=500
                    )

        # State
        file_index = gr.State(start_index)
        score_state = gr.State(None)

        # Load initial pair
        demo.load(
            fn=lambda: load_pair(start_index),
            outputs=[gold_image_display, generated_image_display, score_state, file_info_box]
        ).then(
            fn=lambda: start_index,
            outputs=[file_index]
        ).then(
            fn=lambda score: score,
            inputs=[score_state],
            outputs=[similarity_radio]
        )

        # Navigation
        for btn, direction in [(prev_btn, 0), (next_btn, 1)]:
            btn.click(
                fn=lambda i, s, d=direction: navigate(i, s, d),
                inputs=[file_index, score_state],
                outputs=[
                    gold_image_display, generated_image_display,
                    score_state, file_info_box, file_index, message_box
                ]
            ).then(
                fn=lambda score: score,
                inputs=[score_state],
                outputs=[similarity_radio]
            )

        # Score update
        similarity_radio.change(
            fn=lambda value: value,
            inputs=[similarity_radio],
            outputs=[score_state]
        )

    return demo


def launch_image_evaluator(
    image_pairs_dir: str = "./image_pairs",
    save_file: str = "results.json",
    host: str = "127.0.0.1",
    port: int = 0
) -> None:
    """
    Launch the image evaluation interface.

    Args:
        image_pairs_dir: Directory containing image pair folders
        save_file: File to save results
        host: Server host
        port: Server port (0 for auto)
    """
    if port == 0:
        port = find_free_port()

    print(f"Starting Image Similarity Evaluation at http://{host}:{port}")

    demo = create_image_evaluation_interface(image_pairs_dir, save_file)
    demo.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="store_true", help="package image pairs")
    parser.add_argument("--eval", action="store_true", help="evaluate image similarity")
    parser.add_argument("--save_file", type=str, default="results.json", help="save file name")

    args = parser.parse_args()

    if args.package == args.eval:
        parser.error("Exactly one of --package or --eval must be specified")

    if args.package:
        import os
        from ...configs.paths import enter_project_root, EXPERIMENTS_DIR
        enter_project_root()

        zip_file_path = "./output/human_eval/gradio_output/human_eval_image_sim.zip"
        method_name = "exp1__ernie4_5_vl_424b_a47b"

        # Filter sample keys
        keys = []
        sample_dir = "./output/sampled_5_2"
        for dir_name in os.listdir(sample_dir):
            if os.path.isdir(os.path.join(sample_dir, dir_name)):
                keys.append(dir_name)
        print(f"Found {len(keys)} keys")

        package_evaluation(
            str(EXPERIMENTS_DIR), zip_file_path, method_name, samples_key=keys
        )
        unzip_package(
            zip_file_path,
            "./output/human_eval/gradio_output/human_eval_image_sim/"
        )
    elif args.eval:
        launch_image_evaluator(save_file=args.save_file)
