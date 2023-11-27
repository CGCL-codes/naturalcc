```shell script
# single-lingual dataset
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/ruby
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/java
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/javascript
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/php
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/python
python -m dataset.codesearchnet_feng.sum_bin_ast.preprocess -f config/go
```