# Code Tokenization 

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

