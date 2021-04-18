# Dataset: Python_wan

For Python dataset, its original codes are not runnable in python3. An optional way to deal with such problem is that we
can acquire runnable Python codes from [raw data](https://github.com/wanyao1992/code_summarization_public).

## Step 1

Download pre-processed and raw (python_wan) dataset.

```shell 
bash dataset/python_wan/download.sh
```

## Step 2

Clean raw code files.

```shell 
python -m dataset.python_wan.clean
```

## Step 3

Move **code/code_tokens/docstring/docstring_tokens** to ```~/.ncc/python_wan/flatten/*```.

```shell 
python -m dataset.python_wan.flatten
```

## Step 4 (optional)

Or you can download our processed Python(Wan) dataset

```shell
bash dataset/python_wan/lazy_download.sh
```