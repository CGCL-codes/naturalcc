# Raw-Py150 dataset for *Code Completion* task

## Step 1: download raw-py150 dataset

```shell
bash dataset/raw_py150/download.sh
```

## Step 2: flatten attributes of raw-py150 files

```shell
python -m dataset.raw_py150.attributes_cast
```

## Step 3: preprocess raw-py150 files into code tokens

```shell
python -m dataset.raw_py150.completion.preprocess
```
