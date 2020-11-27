# Dataset: Python_wan

The authors of [A Transformer-based Approach for Source Code Summarizatio
n](https://arxiv.org/pdf/2005.00653.pdf) shared their [code and dataset](https://github.com/wasiahmad/NeuralCodeSum). 
In this repo., it offers original and runnable codes of Java dataset and therefore we can generate AST with Tree-Sitter.

However, as for Python dataset, its original codes are not runnable. An optional way to deal with such problem is that
  we can acquire runnable Python codes from [raw data](https://github.com/wanyao1992/code_summarization_public).

<hr>

# Step 1 
Download pre-processed and raw (python_wan) dataset.
```shell script
bash dataset/python_wan/download.sh
```
Once run this command, you will have python_wan directory like
```shell script
~/.ncc/python_wan/raw/
├── data_ps.declbodies
├── python.zip
├── test
│   ├── code.original
│   ├── code.original_subtoken
│   └── javadoc.original
├── train
│   ├── code.original
│   ├── code.original_subtoken
│   └── javadoc.original
└── valid
    ├── code.original
    ├── code.original_subtoken
    └── javadoc.original

3 directories, 11 files
```  
##### examples
code.original
```shell script
def get_flashed_messages with_categories False category_filter [] flashes _request_ctx_stack top flashesif flashes is None _request_ctx_stack top flashes flashes session pop '_flashes' if '_flashes' in session else [] if category_filter flashes list filter lambda f f[0] in category_filter flashes if not with_categories return [x[1] for x in flashes]return flashes
```
code.original_subtoken
```shell script
def get flashed messages with categories False category filter [] flashes request ctx stack top flashesif flashes is None request ctx stack top flashes flashes session pop ' flashes' if ' flashes' in session else [] if category filter flashes list filter lambda f f[ 0 ] in category filter flashes if not with categories return [x[ 1 ] for x in flashes]return flashes
```
javadoc.original
```shell script
pulls all flashed messages from the session and returns them .
```
data_ps.declbodies
```shell script
def get_flashed_messages(with_categories=False, category_filter=[]): DCNL  DCSP flashes = _request_ctx_stack.top.flashes DCNL DCSP if (flashes is None): DCNL DCSP  DCSP _request_ctx_stack.top.flashes = flashes = (session.pop('_flashes') if ('_flashes' in session) else []) DCNL DCSP if category_filter: DCNL DCSP  DCSP flashes = list(filter((lambda f: (f[0] in category_filter)), flashes)) DCNL DCSP if (not with_categories): DCNL DCSP  DCSP return [x[1] for x in flashes] DCNL DCSP return flashes
```

# Step 2
Clean raw code files.
```shell script
python -m dataset.python_wan.clean
```
##### examples
code with noise
```shell script
def get_flashed_messages(with_categories=False, category_filter=[]): DCNL  DCSP flashes = _request_ctx_stack.top.flashes DCNL DCSP if (flashes is None): DCNL DCSP  DCSP _request_ctx_stack.top.flashes = flashes = (session.pop('_flashes') if ('_flashes' in session) else []) DCNL DCSP if category_filter: DCNL DCSP  DCSP flashes = list(filter((lambda f: (f[0] in category_filter)), flashes)) DCNL DCSP if (not with_categories): DCNL DCSP  DCSP return [x[1] for x in flashes] DCNL DCSP return flashes
```
after clean and save cleaned codes at ```~/.ncc/python_wan/raw/code.json``` 
```shell script
def get_flashed_messages(with_categories=False, category_filter=[]):\n\tflashes = _request_ctx_stack.top.flashes\n\tif (flashes is None):\n\t\t_request_ctx_stack.top.flashes = flashes = (session.pop('_flashes') if ('_flashes' in session) else [])\n\tif category_filter:\n\t\tflashes = list(filter((lambda f: (f[0] in category_filter)), flashes))\n\tif (not with_categories):\n\t\treturn [x[1] for x in flashes]\n\treturn flashes\n
```

# Step 3
Move **code/code_tokens/docstring/docstring_tokens** to ```~/.ncc/python_wan/flatten/*```.
```shell script
python -m dataset.python_wan.flatten
```
Once run this command, you will have python_wan directory like
```shell script
~/.ncc/python_wan/flatten
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

# Step 4
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./config/preprocess.yml) is set correctly.
```shell script
# code_tokens/docstring_tokens
python -m dataset.python_wan.preprocess
```
```shell script
# raw data directory
~/.ncc/python_wan/summarization/data-raw/
├── code_tokens.dict.json
├── docstring_tokens.dict.json
├── test.code_tokens
├── test.docstring_tokens
├── train.code_tokens
├── train.docstring_tokens
├── valid.code_tokens
└── valid.docstring_tokens

0 directories, 8 files
```
```shell script
# bin data directory
~/.ncc/python_wan/summarization/data-mmap/
├── code_tokens.dict.json
├── docstring_tokens.dict.json
├── test.code_tokens.idx
├── test.code_tokens.mmap
├── test.docstring_tokens.idx
├── test.docstring_tokens.mmap
├── train.code_tokens.idx
├── train.code_tokens.mmap
├── train.docstring_tokens.idx
├── train.docstring_tokens.mmap
├── valid.code_tokens.idx
├── valid.code_tokens.mmap
├── valid.docstring_tokens.idx
└── valid.docstring_tokens.mmap

0 directories, 14 files
```
