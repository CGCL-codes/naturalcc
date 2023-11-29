# Dataset: CodeSearchNet(feng)

The authors of [CodeBERT:
A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf) shared
their [code and dataset](https://github.com/microsoft/CodeBERT).

<hr>

## Step 1: Download pre-processed and raw (code_search_net_feng) dataset.

```shell script
bash dataset/codesearchnet_feng/download.sh
```

## Step 2: Flatten attributes of code into ```~/code_search_net_feng/flatten/*```.

```shell script
python -m dataset.codesearchnet_feng.flatten -a code code_tokens docstring docstring_tokens repo
```

## Step 3: Generating raw/bin data with multi-processing.

Before generating datasets, plz make sure [config file](./config/ruby.yml) is set correctly. Here we use codesearchnet_feng_ruby
as exmaple.

```shell script
# code_tokens/docstring_tokens
python -m dataset.codesearchnet_feng.meta_sum.preprocess -f config/ruby
```