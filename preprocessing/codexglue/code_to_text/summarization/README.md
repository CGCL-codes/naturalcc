
## BPE tokenizer for vanilla Summarization task

#### Summarization dataset generation

```shell
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/go
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/java
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/javascript
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/php
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/python
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/ruby
```
