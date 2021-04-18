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

#### Step 3. Generate MBART binarized dataset

```shell
#SRC_DIR=$NCC/ncc_data/codexglue/code-to-text/flatten
#DST_DIR=$NCC/ncc_data/codexglue/code-to-text/summarization/data-mmap
#DICT_FILE=$DST_DIR/dict.json
#cut -f1 dataset/byte-pair-encoding/sentence-level/csn/csn.spm.vocab | sed "s/$/ 100/g" >$DICT_FILE

python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/go
python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/java
python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/javascript
python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/php
python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/python
python -m dataset.codexglue.code_to_text.codebart.preprocess_codebart -f config/ruby

python -m dataset.codexglue.code_to_text.codebart.concate_docstring

# Convert sentencepiece vocab to fairseq format
#bash dataset/codexglue/code_to_text/preprocess/fairseq.sh
```

## BPE tokenizer for vanilla Summarization task

#### Summarization dataset generation

```shell
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/go
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/java
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/javascript
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/php
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/python
python -m dataset.codexglue.code_to_text.summarization.preprocess -f config/ruby
```
