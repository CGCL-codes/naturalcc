# Metrics implementations

## RUBY & CodeBLEU

Our implementation of RUBY & CodeBLEU based on networkx's graph edit distance, Python AST module, and graph generation 
from [typilus](https://github.com/JetBrains-Research/typilus).

## SacreBLEU

We include a modified copy of [sacreBLEU](https://github.com/mjpost/sacrebleu) project in this repository.
We did so to include [our code tokenizer](sacrebleu_code/sacrebleu_methods/tokenizers/tokenizer_code.py) 
and inject it into the [BLEU implementation](sacrebleu_code/sacrebleu_methods/metrics/bleu.py).
