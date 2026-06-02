# Code Summurization Subtree

This directory mirrors a simplified public subset of the code summarization workspace.

Included here are:

- Lightweight evaluation helpers such as BLEU and ROUGE wrappers.
- Mapping utilities and JSON templates used by tokenizer perturbation studies.
- Small data-preparation helpers that do not expose the full training pipeline.

Not included here are:

- Full summarization model training code.
- Internal transfer, embedding-replacement, and large-scale evaluation scripts.
- Checkpoints, outputs, and experiment logs.
