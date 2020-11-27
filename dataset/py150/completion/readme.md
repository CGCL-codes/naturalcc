# Dataset: [PY150](http://files.srl.inf.ethz.ch/data/py150.tar.gz)

Py150 contains 2 files: 100k train data and 50k test data. Each line of them is AST of python code save in JSON format. 

<hr>

# Step 1 
Download raw (py150) dataset.
```shell script
bash dataset/py150/download.sh
```
Once run this command, you will have code_search_net_feng directory like
```shell script
~/.ncc/py150/raw
├── parse_python.py
├── py150.tar.gz
├── python100k_train.json
└── python50k_eval.json

0 directories, 4 files
```

# Step 2
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./config/python.yml) is set correctly.
```shell script
# code_tokens/docstring_tokens
python -m dataset.py150.completion.preprocess -f config/python
```
```shell script
# bin data directory
~/.ncc/py150/completion/data-mmap
├── code_tokens.dict.json
├── test.code_tokens.ext.idx
├── test.code_tokens.ext.mmap
├── test.code_tokens.idx
├── test.code_tokens.mmap
├── train.code_tokens.ext.idx
├── train.code_tokens.ext.mmap
├── train.code_tokens.idx
└── train.code_tokens.mmap

0 directories, 9 files
```
