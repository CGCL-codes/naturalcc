# AVATAR translation dataset

## step 1: download avatar dataset

```shell
bash dataset/avatar/translation/download.sh
```

## step 2: extract code attributes into different files

```shell
python -m dataset.avatar.translation.attributes_cast -k 5
python -m dataset.avatar.translation.attributes_cast -k 3
python -m dataset.avatar.translation.attributes_cast -k 1

python -m dataset.avatar.translation.feature_extract -k 5
python -m dataset.avatar.translation.feature_extract -k 3
python -m dataset.avatar.translation.feature_extract -k 1
```

## step 3(optional): probing statistics of dataset

```shell
python -m dataset.avatar.translation.probe.main -k 5
python -m dataset.avatar.translation.probe.main -k 3
python -m dataset.avatar.translation.probe.main -k 1
```

## step 4: preprocessing

### 1) vanilla tokenization

```shell
python -m dataset.avatar.translation.preprocessing.vanilla.preprocess -f config/topk5
python -m dataset.avatar.translation.preprocessing.vanilla.preprocess -f config/topk3
python -m dataset.avatar.translation.preprocessing.vanilla.preprocess -f config/topk1
```

### 2) codebert

```shell
python -m dataset.avatar.translation.preprocessing.codebert.preprocess -f config/topk5
python -m dataset.avatar.translation.preprocessing.codebert.preprocess -f config/topk3
python -m dataset.avatar.translation.preprocessing.codebert.preprocess -f config/topk1
```

### 3) graphcodebert

```shell
python -m dataset.avatar.translation.preprocessing.graphcodebert.preprocess -f config/topk5
python -m dataset.avatar.translation.preprocessing.graphcodebert.preprocess -f config/topk3
python -m dataset.avatar.translation.preprocessing.graphcodebert.preprocess -f config/topk1
```
