# Python(Wan) dataset for *Code(Path) Summarization* task

## step 1

generate path dataset

```shell
python -m dataset.python_wan.feature_extract -a raw_ast ast path
```

## step 1

preprocess code2seq dataset

```shell
python -m dataset.python_wan.code2seq.preprocess
```