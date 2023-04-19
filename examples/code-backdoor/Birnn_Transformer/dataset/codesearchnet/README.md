# CodeSearchNet dataset

### Step 1

Download CSN raw dataset (```~/ncc_data/codesearchnet/raw```)

```shell
bash dataset/codesearchnet/download.sh
```

### Step 2

Flatten attributes of code snippets into different files. <br>
For instance, flatten ruby's code_tokens into ```~/ncc_data/codesearchnet/attributes/[train/valid/test].code_tokens```.

```shell
python -m dataset.codesearchnet.attributes_cast
```

### Step 3 (optional)

If you want to get AST/binary-AST etc. of code and so on. Plz run such command.

```shell
python -m dataset.codesearchnet.feature_extract
```

<hr>

| task | readme | description |
|------|--------|-------------|
|   hybrid retrieval   |    [README.md](dataset/codesearchnet/retrieval/READme.md)    |       ref: CodeSearchNet      |
|      |        |             |
|      |        |             |
|      |        |             |
|      |        |             |
|      |        |             |