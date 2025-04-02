# Py150 dataset for *Method Prediction(Code Summarization)* task

## step 1

flatten ast into path

```shell
python -m dataset.py150.code2seq.extract
```

## step 2

preprocess code2seq dataset

```shell
python -m dataset.py150.code2seq.method_prediction.preprocess
```