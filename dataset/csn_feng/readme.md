# Dataset: CodeSearchNet(feng)

The authors of [CodeBERT:
A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf) shared their [code and dataset](https://github.com/microsoft/CodeBERT). 

<hr>

# Step 1 
Download pre-processed and raw (code_search_net_feng) dataset.
```shell script
bash dataset/csn_feng/download.sh
```
Once run this command, you will have code_search_net_feng directory like
```shell script
~/.ncc/code_search_net_feng/raw
├── code_search_net_feng.zip
├── go
│   ├── test.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── java
│   ├── test.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── javascript
│   ├── test.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── php
│   ├── test.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── python
│   ├── test.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
└── ruby
    ├── test.jsonl
    ├── train.jsonl
    └── valid.jsonl

6 directories, 19 files

```  

# Step 2
Flatten attributes of code into ```~/.ncc/code_search_net_feng/flatten/*```.
```shell script
python -m dataset.code_search_net_feng.flatten -l [langauge]
```
Once run this command, you will have python_wan directory like
```shell script
~/.ncc/code_search_net_feng/flatten
├── go
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   └── valid.docstring_tokens
├── java
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   └── valid.docstring_tokens
├── javascript
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   └── valid.docstring_tokens
├── php
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   └── valid.docstring_tokens
├── python
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   └── valid.docstring_tokens
└── ruby
    ├── test.code
    ├── test.code_tokens
    ├── test.docstring
    ├── test.docstring_tokens
    ├── train.code
    ├── train.code_tokens
    ├── train.docstring
    ├── train.docstring_tokens
    ├── valid.code
    ├── valid.code_tokens
    ├── valid.docstring
    └── valid.docstring_tokens

6 directories, 72 files
```

# Step 3
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./config/ruby.yml) is set correctly.  Here we use csn_feng_ruby as exmaple.
```shell script
# code_tokens/docstring_tokens
python -m dataset.python_wan.preprocess -f config/ruby
```
```shell script
# bin data directory
~/.ncc/code_search_net_feng/flatten/ruby
├── test.code
├── test.code_tokens
├── test.docstring
├── test.docstring_tokens
├── train.code
├── train.code_tokens
├── train.docstring
├── train.docstring_tokens
├── valid.code
├── valid.code_tokens
├── valid.docstring
└── valid.docstring_tokens

0 directories, 12 files

```
