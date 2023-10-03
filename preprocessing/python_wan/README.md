# Dataset: [Python-code-docstring](https://arxiv.org/abs/1811.07234) [![Python-code-docstring](https://zenodo.org/badge/DOI/10.5281/zenodo.7202649.svg)](https://doi.org/10.5281/zenodo.7202649)   


For Python dataset, its original codes are not executable in python3. An optional way to deal with such problem is that we
can acquire runnable Python codes from [raw data](https://github.com/wanyao1992/code_summarization_public).

## Step 1: Download pre-processed and raw (python_wan) dataset.

```shell 
bash dataset/python_wan/download.sh
```

## Step 2: Clean raw code files.

```shell 
python -m dataset.python_wan.clean
```

## Step 3: Move **code/code_tokens/docstring/docstring_tokens** to ```~/python_wan/flatten/*```.

```shell 
python -m dataset.python_wan.attributes_cast
```

## Step 4 (optional): Or you can download our processed Python(Wan) dataset

```shell
bash dataset/python_wan/lazy_download.sh
```