# Code-to-Code dataset

## 1) preprocess dataset

```shell
bash dataset/codexglue/code_to_code/translation/preprocess.sh
```

## 2) flatten dataset

```shell
python -m dataset.codexglue.code_to_code.translation.attributes_cast
python -m dataset.codexglue.code_to_code.translation.feature_extract
```

vanilla

```shell
python -m dataset.codexglue.code_to_code.translation.vanilla.preprocess -f config/preprocess
```

transformer

```shell
# translation
python -m dataset.codexglue.code_to_code.translation.transformer.preprocess -f config/preprocess
```

codebert

```shell
# translation
python -m dataset.codexglue.code_to_code.translation.codebert.preprocess -f config/preprocess

# retrieval
python -m dataset.codexglue.code_to_code.retrieval.codebert.preprocess -f config/preprocess
```

graphcodebert

```shell
# translation
python -m dataset.codexglue.code_to_code.translation.graphcodebert.preprocess -f config/preprocess

# retrieval
python -m dataset.codexglue.code_to_code.retrieval.graphcodebert.preprocess -f config/preprocess
```

PLBART

```shell
# retrieval
python -m dataset.codexglue.code_to_code.retrieval.plbart.preprocess -f config/csharp
python -m dataset.codexglue.code_to_code.retrieval.plbart.preprocess -f config/java
```

CodeDisen & ParaBART

```shell
python -m dataset.codexglue.code_to_code.translation.codedisen.filter
python -m dataset.avatar.translation.codedisen.filter

python -m dataset.codexglue.code_to_code.translation.codedisen.build_dfs_dict

python -m dataset.codexglue.code_to_code.translation.codedisen.preprocess -f config/preprocess
```

ParaBART

```shell
python -m dataset.codexglue.code_to_code.translation.parabart.preprocess -f config/preprocess
```

Disen

```shell
python -m dataset.codexglue.code_to_code.translation.disen.preprocess -f config/preprocess
```

Noising + Disen

```shell
python -m dataset.codexglue.code_to_code.translation.noising_disen.preprocess -f config/preprocess
```

