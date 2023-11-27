#!/usr/bin/env bash

# python -m dataset.csn.download
# python -m dataset.csn.flatten
#python -m dataset.csn.feature_extract -a code_tokens docstring_tokens raw_ast ast traversal
#python -m dataset.csn.filter -a code_tokens docstring_tokens traversal

python -m dataset.csn.codebert.preprocess_traversal_roberta -l javascript
#python -m dataset.csn.codebert.preprocess_traversal_roberta -l ruby
#python -m dataset.csn.codebert.preprocess_traversal_roberta -l php
#python -m dataset.csn.codebert.preprocess_traversal_roberta -l java
#python -m dataset.csn.codebert.preprocess_traversal_roberta -l python
#python -m dataset.csn.codebert.preprocess_traversal_roberta -l go
