# NaturalCC *v0.4.0*
NaturalCC is a sequence modeling toolkit that allows researchers and developers to train custom models for many software engineering tasks, e.g., code summarization, code retrieval and code clone detection. 
Our vision is to bridge the gap between programming language and natural language via some machine learning techniques.

**This repo. is ongoing, and we are willing to invite you to join the repo and to develop it together.
If you met bugs or problems during usage. Please, feel free to contact us.
Or if you want to merge your work into this repo, pull your request please.**


The project is inspired by [fairseq](https://github.com/pytorch/fairseq).

<p align="center">
    <img src="https://img.shields.io/badge/version-0.4.0-green" alt="Version">
</p>
<hr>


## Features
- mixed precision training
- multi-gpus training
- raw/bin data reading/writing

## Code Tasks
- [Code Summarization](run/summarization)
    - [Seq2Seq](run/summarization/seq2seq/README.md) [\[pdf\]](https://arxiv.org/pdf/1409.3215.pdf)
    - [vanilla Transformer](run/summarization//README.md) [\[pdf\]](https://arxiv.org/pdf/1706.03762.pdf)
    - [NeuralTransformer](run/summarization/neural_transformer/README.md) [\[pdf\]](https://arxiv.org/pdf/2005.00653.pdf)
- [Code Retrieval](run/retrieval)
    - [NBOW](run/retrieval/nbow/README.md)
- [Code Prediction](run/completion)
    - [NBOW](run/completion/seqrnn/README.md)
- [Type Inference](run/type_prediction)

TBC...

## Dataset
Currently, we have processed the following datasets:

- [Python_wan](dataset/python_wan/README.md)
- [CSN](dataset/csn/README.md)
- [CSN(feng)](dataset/csn_feng/README.md)
- [Py150](dataset/py150/README.md)

TBC...

## TBC:
Please wait.




## Requirements 
- PyTorch version >= 1.4.0
- Python version >= 3.6
- For training new models, you'll also need an NVIDIA GPU and NCCL
- For faster training install NVIDIA's apex library with the --cuda_ext and --deprecated_fused_adam options

## Installation
#### 1) Install [apex](https://github.com/NVIDIA/apex)
 to support half precision training.

```shell script
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```

#### 2) Install other prerequisites libraries
```shell script
git clone https://github.com/xcodemind/naturalcc
cd naturalcc
pip install -r requirements.txt

# or install with conda 
# conda install --yes --file requirements.txt
```
BTW, [install.md](install.md) supports virtual environment installation in details. 
If you meet problems in installation, you can refer to the file. 


#### 3) Install NCC
```shell script
# build for development 
python setup.py build_ext --inplace

# install 
pip install --editable ./
```


## License
NaturalCC is MIT-licensed. The license applies to the pre-trained models as well.

## Citation
Please cite as:
xxx