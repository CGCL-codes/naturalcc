# AST/Bianry-AST dataset generation

## Step 1

Generating ast/binary-ast data with multi-processing.

```shell script
# code_tokens/docstring_tokens
python -m dataset.python_wan.feature_extract
```

## Step 2

Generating raw/bin data with multi-processing. Before generating datasets, plz make
sure [config file](./sum_bin_ast/config/preprocess.yml) is set correctly.

```shell script
# code_tokens/docstring_tokens
python -m dataset.python_wan.sum_bin_ast.preprocess
```
