# Dataset: CodeSearchNet

The authors of [CodeSearchNet Challenge: Evaluating the State of Semantic Code Search](https://arxiv.org/pdf/1909.09436.pdf) shared their [code and dataset](https://github.com/github/codesearchnet). 

<hr>

# Step 1 
Download pre-processed and raw (code_search_net) dataset.
```shell script
bash dataset/csn/download.sh
```
Once run this command, you will have code_search_net directory like
```shell script
/home/yang/.ncc/code_search_net/raw
├── go
│   ├── go_test_0.jsonl.gz
│   ├── go_train_0.jsonl.gz
│   ├── go_train_10.jsonl.gz
│   ├── go_train_1.jsonl.gz
│   ├── go_train_2.jsonl.gz
│   ├── go_train_3.jsonl.gz
│   ├── go_train_4.jsonl.gz
│   ├── go_train_5.jsonl.gz
│   ├── go_train_6.jsonl.gz
│   ├── go_train_7.jsonl.gz
│   ├── go_train_8.jsonl.gz
│   ├── go_train_9.jsonl.gz
│   └── go_valid_0.jsonl.gz
├── go.zip
├── java
│   ├── java_test_0.jsonl.gz
│   ├── java_train_0.jsonl.gz
│   ├── java_train_10.jsonl.gz
│   ├── java_train_11.jsonl.gz
│   ├── java_train_12.jsonl.gz
│   ├── java_train_13.jsonl.gz
│   ├── java_train_14.jsonl.gz
│   ├── java_train_15.jsonl.gz
│   ├── java_train_1.jsonl.gz
│   ├── java_train_2.jsonl.gz
│   ├── java_train_3.jsonl.gz
│   ├── java_train_4.jsonl.gz
│   ├── java_train_5.jsonl.gz
│   ├── java_train_6.jsonl.gz
│   ├── java_train_7.jsonl.gz
│   ├── java_train_8.jsonl.gz
│   ├── java_train_9.jsonl.gz
│   └── java_valid_0.jsonl.gz
├── javascript
│   ├── javascript_test_0.jsonl.gz
│   ├── javascript_train_0.jsonl.gz
│   ├── javascript_train_1.jsonl.gz
│   ├── javascript_train_2.jsonl.gz
│   ├── javascript_train_3.jsonl.gz
│   ├── javascript_train_4.jsonl.gz
│   └── javascript_valid_0.jsonl.gz
├── javascript.zip
├── java.zip
├── php
│   ├── php_test_0.jsonl.gz
│   ├── php_train_0.jsonl.gz
│   ├── php_train_10.jsonl.gz
│   ├── php_train_11.jsonl.gz
│   ├── php_train_12.jsonl.gz
│   ├── php_train_13.jsonl.gz
│   ├── php_train_14.jsonl.gz
│   ├── php_train_15.jsonl.gz
│   ├── php_train_16.jsonl.gz
│   ├── php_train_17.jsonl.gz
│   ├── php_train_1.jsonl.gz
│   ├── php_train_2.jsonl.gz
│   ├── php_train_3.jsonl.gz
│   ├── php_train_4.jsonl.gz
│   ├── php_train_5.jsonl.gz
│   ├── php_train_6.jsonl.gz
│   ├── php_train_7.jsonl.gz
│   ├── php_train_8.jsonl.gz
│   ├── php_train_9.jsonl.gz
│   └── php_valid_0.jsonl.gz
├── php.zip
├── python
│   ├── python_test_0.jsonl.gz
│   ├── python_train_0.jsonl.gz
│   ├── python_train_10.jsonl.gz
│   ├── python_train_11.jsonl.gz
│   ├── python_train_12.jsonl.gz
│   ├── python_train_13.jsonl.gz
│   ├── python_train_1.jsonl.gz
│   ├── python_train_2.jsonl.gz
│   ├── python_train_3.jsonl.gz
│   ├── python_train_4.jsonl.gz
│   ├── python_train_5.jsonl.gz
│   ├── python_train_6.jsonl.gz
│   ├── python_train_7.jsonl.gz
│   ├── python_train_8.jsonl.gz
│   ├── python_train_9.jsonl.gz
│   └── python_valid_0.jsonl.gz
├── python.zip
├── ruby
│   ├── final
│   │   └── jsonl
│   │       ├── test
│   │       │   └── ruby_test_0.jsonl.gz
│   │       ├── train
│   │       │   ├── ruby_train_0.jsonl.gz
│   │       │   └── ruby_train_1.jsonl.gz
│   │       └── valid
│   │           └── ruby_valid_0.jsonl.gz
│   ├── ruby_test_0.jsonl.gz
│   ├── ruby_train_0.jsonl.gz
│   ├── ruby_train_1.jsonl.gz
│   └── ruby_valid_0.jsonl.gz
└── ruby.zip

11 directories, 88 files
```  

# Step 2
Flatten attributes of code into ```~/.ncc/code_search_net/flatten/*```.
```shell script
python -m dataset.code_search_net.flatten -l [langauge]
```
Once run this command, you will have a directory like
```shell script
~/.ncc/code_search_net/flatten
├── go
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── test.func_name
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── train.func_name
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   ├── valid.docstring_tokens
│   └── valid.func_name
├── java
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── test.func_name
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── train.func_name
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   ├── valid.docstring_tokens
│   └── valid.func_name
├── javascript
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── test.func_name
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── train.func_name
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   ├── valid.docstring_tokens
│   └── valid.func_name
├── php
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── test.func_name
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── train.func_name
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   ├── valid.docstring_tokens
│   └── valid.func_name
├── python
│   ├── test.code
│   ├── test.code_tokens
│   ├── test.docstring
│   ├── test.docstring_tokens
│   ├── test.func_name
│   ├── train.code
│   ├── train.code_tokens
│   ├── train.docstring
│   ├── train.docstring_tokens
│   ├── train.func_name
│   ├── valid.code
│   ├── valid.code_tokens
│   ├── valid.docstring
│   ├── valid.docstring_tokens
│   └── valid.func_name
└── ruby
    ├── test.code
    ├── test.code_tokens
    ├── test.docstring
    ├── test.docstring_tokens
    ├── test.func_name
    ├── train.code
    ├── train.code_tokens
    ├── train.docstring
    ├── train.docstring_tokens
    ├── train.func_name
    ├── valid.code
    ├── valid.code_tokens
    ├── valid.docstring
    ├── valid.docstring_tokens
    └── valid.func_name

114 directories, 500 files
```

# Step 3
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./config/) is set correctly.  Here we use csn_ruby as exmaple.
```shell script
# code_tokens/docstring_tokens
python -m dataset.code_search_net.retrieval.preprocess -f config/ruby
```
```shell script
# bin data directory
~/.ncc/code_search_net/retrieval/ruby/data-mmap
├── code_tokens.dict.json
├── docstring_tokens.bpe.dict.json
├── test.code_tokens.idx
├── test.code_tokens.mmap
├── test.code_tokens_wo_func.idx
├── test.code_tokens_wo_func.mmap
├── test.docstring_tokens.bpe.idx
├── test.docstring_tokens.bpe.mmap
├── test.func_name.bpe.idx
├── test.func_name.bpe.mmap
├── train.code_tokens.idx
├── train.code_tokens.mmap
├── train.code_tokens_wo_func.idx
├── train.code_tokens_wo_func.mmap
├── train.docstring_tokens.bpe.idx
├── train.docstring_tokens.bpe.mmap
├── train.func_name.bpe.idx
├── train.func_name.bpe.mmap
├── valid.code_tokens.idx
├── valid.code_tokens.mmap
├── valid.code_tokens_wo_func.idx
├── valid.code_tokens_wo_func.mmap
├── valid.docstring_tokens.bpe.idx
├── valid.docstring_tokens.bpe.mmap
├── valid.func_name.bpe.idx
└── valid.func_name.bpe.mmap

0 directories, 26 files
```
