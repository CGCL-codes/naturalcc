# summerizaton dataset generation

## Step 1

Generating raw/bin data with multi-processing. Before generating datasets, plz make
sure [config file](./tokenization/config/preprocess.yml) is set correctly.

```shell script
# code_tokens/docstring_tokens
python -m dataset.python_wan.tokenization.preprocess
```
