CodeXGlue dataset

# Code Summarization

## BPE tokenizer for mBART

#### Step 1. Download CodeSearchNet(feng) dataset

```shell
bash dataset/codexglue/code_to_text/download.sh
```

#### Step 2. Cast attributes into files.

```shell
python -m dataset.codexglue.code_to_text.attributes_cast
```

#### Step 3. Tokenize code/docstring with SPM tokenizer

```shell
python -m dataset.codexglue.code_to_text.spm_tokenize
```


### step 3(optional). extract features of data attributes. For instance, AST, binary-AST etc. of code.
```shell
python -m dataset.codexglue.code_to_text.feature_extract -l [language] -f [flatten data directory] -s [parse file] -a [data attributes] -c [cpu cores]
```