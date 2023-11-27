# Py150 dataset for *Code Completion* task

## step 1

download py150 dataset

```shell
bash dataset/py150/download.sh
```

## step 2

flatten py150 into new ast data

```shell
python -m dataset.py150.attributes_cast
```

## step 3

preprocess travtrans dataset

```shell
python -m dataset.py150.trav_trans.preprocess
```