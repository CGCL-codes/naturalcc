# Code Tokenization Artifact (Public Subset)

This repository provides a lightweight public subset of the code used in our study on code tokenization for code language models (CLMs).

It is intended to document the data processing and tokenizer-construction workflow used in several experiments, while keeping the artifact compact and focused on reusable preprocessing components.

## What is included

- Dataset preparation scripts for the shared-token masking study.
- Example tokenizer-construction scripts for CodeT5 and CodeLlama style tokenizers.
- A simplified `1tasks/Code_Generation` subtree with public utility code, mapping helpers, and evaluation wrappers.
- Simplified `1tasks/Code_Summurization` and `2token_semantics` subtrees with public utility code and data-preparation scripts.
- A minimal environment description and command examples.

## What is not included

- Full training pipelines for all models and settings.
- Full evaluation pipelines and result-generation workflows.
- Experiment outputs, checkpoints, cached datasets, and logs.
- Internal orchestration scripts used for large-scale runs.
- Core unpublished algorithmic components that are still under active review.

## Directory layout

```text
code-tokenization/
  README.md
  LICENSE
  environment.yml
  docs/
  scripts/
```

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate code-tokenization
```

Prepare masking data for the shared-token study:

```bash
python scripts/prepare_codegen_mask_data.py --dataset-dir /path/to/concode
python scripts/prepare_codesumm_mask_data.py --dataset-root /path/to/codesearchnet --langs java,python
```

Train a modified tokenizer from a code-text corpus:

```bash
python scripts/build_tokenizer_codet5.py \
  --base-tokenizer /path/to/codet5 \
  --data-files /path/to/train.json /path/to/dev.json /path/to/test.json \
  --output-dir ./artifacts/codet5_concode
```

```bash
python scripts/build_tokenizer_codellama.py \
  --base-tokenizer /path/to/codellama \
  --data-files /path/to/train.json /path/to/dev.json /path/to/test.json \
  --output-dir ./artifacts/codellama_concode
```

## Notes

- The scripts assume JSON or JSONL inputs with fields such as `nl`, `code`, or `code_tokens`, depending on the task.
- File formats may need small adjustments for other datasets.
- This public subset is designed to be readable and self-contained rather than exhaustive.
- Paths are provided through command-line arguments; no local machine paths are required.
