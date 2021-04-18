# NaturalCC *v0.5.0*

NaturalCC is a sequence modeling toolkit that allows researchers and developers to train custom models for many software
engineering tasks, e.g., code summarization, code retrieval, code completion, code clone detection and type inference.
Our vision is to bridge the gap between programming language and natural language through machine learning techniques.

<p align="center">
    <img src="https://img.shields.io/badge/version-0.5.0-green" alt="Version">
</p>
<hr>

## Features

- A collection of code corpus with data preprocessing
- Performance benchmark
- Mixed precision training
- Multi-gpu training

## Dataset

Currently, we have processed the following datasets:

- [Python (Wan et al.)](dataset/python_wan/README.md)
- [CodeSearchNet](dataset/codesearchnet/README.md)
- [CodeXGlue (Feng et al.)](dataset/codexglue/code_to_text/README.md)
- Py150 [(official processed)](dataset/py150/README.md) [(raw)](dataset/raw_py150/README.md)

## Requirements

- PyTorch version >= 1.4.0
- Python version >= 3.6
- GCC/G++ > 5.0
- For training new models, you'll also need an NVIDIA GPU and NCCL
- For faster training install NVIDIA's apex library with the --cuda_ext and --deprecated_fused_adam options

## Installation

#### 1. Half precision computation

NaturalCC supports half precision training.

- If your ``Pytorch.__version__ < 1.6.0`` and ```nvcc -V``` is runnable, install [apex](https://github.com/NVIDIA/apex).
- Otherwise, use Pytorch to build half precision module ```torch.cuda.amp```. This will be supported in the future.

#### 2. Install other prerequisite libraries

```shell
git clone https://github.com/xcodemind/naturalcc
cd naturalcc
pip install -r requirements.txt
```

#### 3. Build or install NaturalCC

Export your NaturalCC cache directory (data and models will be saved in this directory) to user
variables(```~/.bashrc``` or  ```~/.zshrc```).

```shell
echo "export NCC=[directory to save NaturalCC data/models ]" >> ~/.bashrc
```

> Note: PyCharm may not get environment variables and thus we recommend you to register your NCC variable at ncc/\_\_init\_\_.py*

Compile Cython files to accelerate programs and register NaturalCC into your pip list

```shell
# compile for debug
# python setup.py build_ext --inplace
# install 
pip install --editable ./
```

Since NCC is build via Cython, your GCC/G++ version should be greater than 4.9. If you have the root permission, update
GCC/G++; otherwise, install GCC/G++ with conda.

```shell
# install GCC/G++ with conda
conda install -c anaconda gxx_linux-64
conda install -c conda-forge gcc_linux-64
cd ~/anaconda/envs/XXX/bin
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-g++ g++
# check
conda deactivate
conda activate XXX
>> type "gcc/g++ -v" in terminals
```

## Examples

> All the running commands here should be executed in the root of project folder (the path of your `naturalcc`). For example, in my environment I will stay at `/data/wanyao/Dropbox/ghproj-v100/naturalcc`.

## Pre-trained models and logs

### [Code Summarization](run/summarization)

|                 | BLEU-4 | METEOR | ROUGE-L | Cost    | Logs    |
|-----------------|--------|--------|---------|---------|---------|
| Seq2Seq+Attn    | 25.57  | 14.40  | 39.41   | 0.09s/b | [click here](run/summarization/seq2seq/config/python_wan/python.log) |
| Tree2Seq+Attn   | 23.35  | 12.59  | 36.49   | 0.48s/b | [click here](run/summarization/tree2seq/config/python_wan/python.log) |
| Transformer     | 30.64  | 17.65  | 44.59   | 0.26s/b | [click here](run/summarization/transformer/config/python_wan/python.log) |
| Transformer+RPE | 31.57  | 17.74  | 45.18   | 0.27s/b | [click here](run/summarization/neural_transformer/relative/python_wan/python.log) |
| PLBART          | 32.71  | 18.13  | 46.05   | 0.80s/b | [TBC]() |

### [Code Retrieval](run/retrieval)

|          | Go    | Java  | JS    | PHP   | Python | Ruby  | Cost    | Logs    |
|----------|-------|-------|-------|-------|--------|-------|---------|---------|
| NBOW     | 66.59 | 59.92 | 47.15 | 54.75 | 63.33  | 42.86 | 0.16s/b | [click here](run/retrieval/nbow/config/csn/all.log) |
| ConV1d   | 70.87 | 60.49 | 38.81 | 61.92 | 67.29  | 36.53 | 0.30s/b | [click here](run/retrieval/conv1d/config/csn/all.log) |
| BiRNN    | 65.80 | 48.60 | 23.23 | 51.36 | 48.28  | 19.35 | 0.74s/b | [click here](run/retrieval/birnn/config/csn/all.log) |
| SelfAttn | 78.45 | 66.55 | 50.38 | 65.78 | 79.09  | 47.96 | 0.25s/b | [click here](run/retrieval/selfattn/config/csn/all.log) |

### [Code Completion](run/completion)

|           | Attr  | Num   | Name   | Param | Tokens | Cost    | Logs    |
|-----------|-------|-------|--------|-------|--------|---------|---------|
| LSTM      | 51.67 | 47.45 | 46.52  | 66.06 | 73.73  | 0.31s/b | [click here](run/completion/seqrnn/config/raw_py150/python.log) |
| GTP-2     | 70.37 | 62.20 | 63.84  | 73.54 | 82.17  | 0.43s/b | [click here](run/completion/gpt2/config/raw_py150/python.log) |
| TravTrans | 72.08 | 68.55 | 76.33  | 71.08 | 83.17  | 0.43s/b | [click here](run/completion/trav_trans/config/py150/python.log) |

### Type Inference

|             | Acc@1 (All types) | Acc@5 (All types) | Acc@1 (Any types) | Acc@5 (Any types) | Cost    | Logs    |
|-------------|-------------------|-------------------|-------------------|-------------------|---------|---------|
| DeepTyper   | 0.52              | 0.67              | 0.43              | 0.67              | 0.42s/b | [TBC]() |
| Transformer | 0.32              | 0.64              | 0.37              | 0.75              | 0.85s/b | [TBC]() |

## License and Acknowledgement

NaturalCC is [MIT-licensed](https://github.com/CGCL-codes/naturalcc/blob/master/LICENSE.txt). The license applies to the
pre-trained models as well. This project is also highly inspired by [Fairseq](https://github.com/pytorch/fairseq)
and [AllenNLP](https://allennlp.org).

## Citation

Please cite as:

```
under reviewing
```