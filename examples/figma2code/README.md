# Figma2Code

Figma-to-Code: generate front-end code (HTML + Tailwind CSS) from Figma designs using large language and vision-language models.  
This repository provides a unified codebase for **generation experiments**, **evaluation**, and **data processing**.

---

## 1. Project Overview

- **Goal**: Given a Figma design page, automatically generate high-quality front-end code (HTML + Tailwind CSS) with large language / vision-language models.
- **Main Features**:
  - Unified experiment runner (`main.py`) for different research settings
  - Multiple generation methods (direct, agent-based, ablation)
  - Automatic evaluation pipeline (code quality, visual quality, human evaluation helpers)
  - Full data processing pipeline from raw Figma pages to curated datasets

---

## 2. Repository Structure (High-level)

- `main.py`  
  Unified CLI entry point. Used to run all experiments (generation + evaluation).

- `src/configs/`  
  Basic configuration, model registry, paths and environment-related settings.

- `src/generation/`  
  Figma-to-code generation logic:
  - `design2code.py`, `direct.py`, `base.py`
  - `agent/`: agent-style multi-step generation (IR construction, refinement, Tailwind code generation, critic, etc.)
  - `ablation.py`: utilities for input ablation experiments.

- `src/evaluation/`  
  Automatic evaluation utilities, including code quality and visual quality metrics.

- `src/exp_scripts/`  
  High-level experiment scripts used by `main.py`:
  - `exp1_model_comparison.py`
  - `exp2_method_comparison.py`
  - `exp3_ablation.py`

- `src/human_eval/`  
  Human evaluation helpers (e.g., MOS scoring, result analysis, plotting).

- `src/llm/`  
  LLM wrappers (OpenRouter, base interfaces).

- `src/utils/`  
  Shared utilities (console logger, image utilities, file helpers, Figma helpers, HTML screenshot, parsing, etc.).

- `src/data_script/`  
  Full data processing pipeline (raw Figma collection → rule-based filtering → annotation → test-set construction → metadata refinement).  
  **Details are documented in** `src/data_script/readme.md` (see Section 5).

- `src/scripts/`  
  Standalone analysis scripts (e.g., sample balancing, stylelint analysis, annotation statistics).

---

## 3. Environment & Configuration

1. **Python environment**
   - Python 3.12 is recommended.
   - Create a virtual environment and install dependencies from the project root:

     ```bash
     pip install -r requirements.txt
     ```

     See `requirements.txt` for pinned versions; adjust if needed for your environment.

2. **Environment variables**
   - Copy `.env.example` to `.env` and fill in your own values:

     - `OPENROUTER_API_KEY`: API key for LLM access (required for running generation experiments)
     - `HTTP_PROXY`, `HTTPS_PROXY`: optional, for network proxy
     - `FIGMA_API_KEY`: Figma API key (only required if you want to run data collection scripts)
     - `FIGMA_EMAIL`, `FIGMA_PASSWORD`: Figma account credentials (used for crawling community pages)
     - `IMAGE_HOST_PREFIX`, `IMAGE_HOST_PREFIX_SMALL`: image hosting prefix (optional; defaults are provided in code)

3. **Project root**
   - `main.py` and many scripts assume the **project root** as working directory.
   - `src/configs/paths.py` and `src/configs/settings.py` define default paths and settings.

---

## 4. Running Experiments via `main.py`

`main.py` provides a unified CLI interface to run all experiments and evaluations:

```bash
# Model comparison (exp1)
python main.py exp1
python main.py exp1 --models gpt4o claude_opus_4_1

# Run exp1 again with a save prefix (for stability / reproducibility)
python main.py exp1 --models gpt4o --save-prefix run2_

# Method comparison (exp2)
python main.py exp2
python main.py exp2 --methods image_direct figma_direct --model gpt4o

# Method comparison with extra guidelines (prompt robustness)
python main.py exp2 --add-guidelines

# Ablation study (exp3)
python main.py exp3
python main.py exp3 --ablations geometry style --methods figma_direct
```

**exp2-only options:**

- `--agent-max-steps N`: max refinement steps for the agent method (default: 5)
- `--add-guidelines`: add generation guidelines to the prompt (for robustness testing; applies to `figma_direct` and `figma_image_direct`)

### 4.1 Common CLI Options

All three experiments share a set of common options (see `create_parser()` / `_add_common_args()` in `main.py`):

- `--data-dir`: data directory (default: `data/data_test`)
- `--samples`: list of sample keys to process
- `--replace`: replace existing HTML files
- `--skip-generation`: only run evaluation (skip generation)
- `--skip-evaluation`: only run generation (skip evaluation)
- `--save-prefix`: extra prefix for log / result filenames
- `--replace-evaluation`: re-evaluate all samples (default: skip already evaluated samples)

Example:

```bash
python main.py exp1 --samples sample1 sample2 --replace --save-prefix debug_
```

### 4.2 Listing Available Options

You can list available models, methods, ablation types, and experiments:

```bash
python main.py list models
python main.py list methods
python main.py list ablations
python main.py list experiments
```

The `list` command prints descriptions and default choices (e.g., default models for `exp1`, default methods for `exp2`, supported ablation types for `exp3`).

---

## 5. Data Processing Pipeline (`src/data_script/`)

The **full data processing pipeline** is documented in **`src/data_script/readme.md`**. That document includes directory layout under `output/page_filter/`, step-by-step commands, and a workflow summary.

**How to run data scripts:** Scripts in `src/data_script/` use relative imports, so run them from the **project root** in module mode:

```bash
python -m src.data_script.<submodule>.<script_name>
```

Example: `python -m src.data_script.rule_based_filtering.json_filter`

**Pipeline stages (overview):**

1. **Raw Data Collection**  
   Crawl Figma file keys, download screenshots and metadata, generate initial whitelists. Scripts: `filekey_crawl`, `filekeysplit`, `figma_page_filter`.

2. **Rule-based Filtering**  
   JSON-based filters, similarity filters (CLIP), and LLM score filters. Scripts: `json_filter`, `similarity_filter`, `score_filter`.

3. **Manual Selection & Annotation**  
   Gradio-based annotation UI, split packages for annotators, aggregation of annotations. Scripts: `dataset_split_package`, `annotate_and_generate_whitelist`, `content_category_mapper`.

4. **Test Set Sampling & Professional Selection**  
   Stratified sampling and final selection by professional designers. Script: `filter_test_data` (modes: `sample`, `generate`).

5. **Final Dataset Processing & Metadata Refinement**  
   Download final data, process Figma JSON, collect resources, generate metadata and reports. Script: `figma_metadata_process`.

For **reproducing the dataset** or customizing a stage, follow the detailed commands and paths in `src/data_script/readme.md`.

---

## 6. Human Evaluation & Result Analysis

- `src/human_eval/eval_script/`: scripts for human evaluation (e.g., image-level / code-level evaluators).
- `src/human_eval/results_analysis/`: utilities for analyzing human scores and plotting correlations, MOS curves, etc.

These modules assume that generation results and evaluation logs are already produced by running `main.py` experiments.

---

## 7. Notes

- Many scripts assume a particular directory layout under `data/` and `output/`. Check the corresponding script or `src/data_script/readme.md` for expected paths.
- Intermediate artifacts (downloaded images, JSON, temporary statistics) are intentionally kept for debugging and re-running partial stages.
- If you encounter issues, please start by:
  - Checking `.env` and `src/configs/settings.py`
  - Confirming that `OPENROUTER_API_KEY` and (if applicable) Figma-related keys are correctly set
  - Verifying that your current working directory is the project root

---
