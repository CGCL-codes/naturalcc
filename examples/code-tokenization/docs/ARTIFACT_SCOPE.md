# Artifact Scope

This public artifact focuses on preprocessing and tokenizer construction.

Included components:

- Shared-token masking data preparation for code generation and code summarization.
- Example tokenizer retraining scripts based on Hugging Face tokenizers.

Excluded components:

- Full downstream training and evaluation orchestration.
- Large-scale experiment management code.
- Model checkpoints, cached outputs, and result tables.

The goal is to make the preprocessing logic understandable and reusable without bundling the full experiment workspace.
