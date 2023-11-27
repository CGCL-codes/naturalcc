# Dataset
We release the Python AST dataset in [Google Drive](https://drive.google.com/file/d/1BD8BCmGi3mds9su3eWMZYi8ybGB4UEJE/view?usp=sharing).


# Installing
Please install the following libraries.

```
pip install transformers
pip install numpy
pip install torch
pip install nltk
pip install scipy
```

# Experiments
This section describes how to reproduce the experiments in the paper.

## Attention analysis

First, we release the attention distribution of Python AST in [Google Drive](https://drive.google.com/file/d/1FCDcl7eRm_H30-huqnWe7rVSCd7Jx0nl/view?usp=share_link), where the value is 1 if two nodes are under the same parent node and are not neighbor.

Second, navigate to appropriate directory:

```
cd <project_root>/code_attention
python compute_edge_features.py
```

## Structural Probing

1. Use `scripts/convert_raw_to_bert.py` to convert the source code into BERT vectors and dump them to disk in hdf5 format.
2. Replace the data paths (and choose a results path) in the yaml configs in `example/config` with the paths that point to your `.ast` and `.hdf5` files as constructed in the above steps.

```
python code_probe_structural/example/prd_code_bert_mean.yaml
```

The `prd_code_bert_mean.yaml` file is configed as follows:


The parameters of dataset
```
dataset:
  observation_fieldnames: the fields (columns) of the corpus files to be used
    - code_ast
    - code_tokens
    - embeddings
  corpus: the location of the train, dev, and test  corpora files.
    root: ../../data/code_new/python_ast_new
    train_path: train.ast
    dev_path: valid.ast
    test_path: test.ast
  embeddings: the location of the train, dev, and test pre-computed embedding files
    type: token #{token,subword}
    root: ../../data/code_new/code_probe
    train_path: code_bert_train.hdf5
    dev_path: code_bert_valid.hdf5
    test_path: code_bert_test.hdf5
  batch_size: 40
```

The parameters of models
```
Model:
  hidden_dim: 768 # hidden dim
  model_type: CodeBert-disk # CodeBert-disk or GraphCodeBert-disk
  use_disk: True
  model_layer: 5
```


# Syntax Tree Induction

```
Python run.py --help
usage: run.py [-h] [--data-path DATA_PATH] [--result-path RESULT_PATH]
          [--from-scratch] [--gpu GPU] [--bias BIAS] [--seed SEED]
          [--token-heuristic TOKEN_HEURISTIC] [--use-coo-not-parser]

optional arguments:
  -h, --help    show this help message and exit
  --data-path DATA_PATH
  --result-path RESULT_PATH
  --gpu GPU
  --bias BIAS   the right-branching bias hyperparameter lambda
  --seed SEED
  --token-heuristic TOKEN_HEURISTIC     Available options: mean, first, last
```
