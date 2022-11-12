# <center> NaturalCC


NaturalCC is a sequence modeling toolkit that allows researchers and developers to train custom models for many software
engineering tasks, e.g., code summarization, code retrieval, code completion, code clone detection and type inference.
Our vision is to bridge the gap between programming language and natural language through machine learning techniques.

<p align=center>
  <a href="https://xcodemind.github.io/">
    <img src="https://img.shields.io/badge/NaturalCC-0.6.0-green" alt="Version">
  </a>

  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.6-3776AB?logo=python" alt="Python">
  </a>

  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch" alt="pytorch">
  </a>

  <a href="https://github.com/EdisonLeeeee/GraphGallery/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EdisonLeeeee/GraphGallery" alt="license">
  </a>       
</p>
<hr>

## 🔖 News 
- [May 10] We have merged the source code of "What Do They Capture? - A Structural Analysis of Pre-Trained Language Models for Source Code" into NaturalCC.

## ⭐ Features

- A collection of code corpus with data preprocessing
- Performance benchmark
- Mixed precision training
    - Nvidia APEX
    - Automatic Mixed Precision
- Multi-GPU training
- Better logging output
- Various Implementations:
    - tensorflow gradient clipping
    - optimizers or learning schedulers
    - baseline models
    - binary data formats


## 🚀 Installation

### Requirements

- PyTorch version >= 1.6.0
- Python version >= 3.6
- GCC/G++ > 5.0
- For training new models, you'll also need an NVIDIA GPU, NCCL and Cuda Toolkit installed.
- (optional) For faster training, you need to install NVIDIA's ```apex``` library.

[comment]: <> "  with the --cuda_ext and --deprecated_fused_adam options"

#### 1. Install prerequisite libraries

```shell
git clone https://github.com/CGCL-codes/naturalcc && cd naturalcc
pip install -r requirements.txt
```

Once you installed prerequisite libraries, you can check them via
```python -m env_test```

#### 2. Build or install NaturalCC

Export your NaturalCC cache directory (data and models will be saved in this directory) to user
variables(```~/.bashrc``` or  ```~/.zshrc``` in Linux, ```~/.zsh_profile``` or ```~/.bash_profile``` in macOS).

```shell
# Linux
echo "export NCC=<path_to_store ncc_data>" >> ~/.bashrc
# macOS
echo "export NCC=<path_to_store ncc_data>" >> ~/.bash_profile
```

> Note: PyCharm cannot get environment variables and, therefore, we recommend you to register your NCC variable at ```ncc/__init__.py```.

Compile Cython files to accelerate programs and register NaturalCC into your pip list

```shell
# compile for debug
# python setup.py build_ext --inplace
# install 
pip install --editable ./
```

#### 3. Half precision computation (optional)

NaturalCC supports half precision training.

- If your ``Pytorch.__version__ < 1.6.0`` and ```nvcc -V``` is runnable, please install [apex](https://github.com/NVIDIA/apex).
- Otherwise, use Automatic Mixed Precision (AMP). Available Now (set ```amp: 1``` in yaml file, [An example](run/summarization/seq2seq/config/python_wan/python.yml)).


#### 4. Install GCC/G++ with conda (if you do not have permission)

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

## 📚 Dataset

Currently, we have processed the following datasets:

- [Python (Wan et al.)](ncc_dataset/python_wan/README.md)
- [CodeSearchNet (Husain et al.)](ncc_dataset/codesearchnet/README.md)
- [CodeXGlue (Feng et al.)](ncc_dataset/codexglue/code_to_text/README.md)
- Py150 [(official processed)](ncc_dataset/py150/README.md) [(raw)](ncc_dataset/raw_py150/README.md)
- [OpenCL (Grewe et al.)](ncc_dataset/opencl/README.md)
- [Java (Hu et, al.)](ncc_dataset/java_hu/README.md)
- [Stack Overflow](ncc_dataset/stackoverflow/README.md)
- [DeepCS (Gu et al.)](ncc_dataset/deepcs)
- [AVATAR (Ahmad et al.)](ncc_dataset/avatar)
- [StackOverflow (Iyer et al.)](ncc_dataset/stackoverflow)

## 🤖 Implementations

#### Code retrieval (search)
- [NBOW](ncc/models/retrieval/nbow.py)
- [BiRNN](ncc/models/retrieval/birnn.py)
- [1D-CNN](ncc/models/retrieval/birnn.py)
- [SelfAttn](ncc/models/retrieval/self_attn.py)
- [Deepcs](ncc/models/retrieval/deepcs.py)

#### Code completion
- [SeqRNN](ncc/models/completion/seqrnn.py)
- [GPT2](ncc/models/completion/gpt2.py)

#### Heterogeneous mapping
- [static mapping](run/mapping/static_mapping)
- [decision tree](run/mapping/decision_tree)
- [deeptune](run/mapping/deeptune)
- [inst2vec](run/mapping/inst2vec)

#### Code summarization
- [Naive Copy](run/translation/naive_copy)
- [CodeNN](ncc/models/summarization/codenn.py)
- [DeepCom](ncc/models/summarization/deepcom.py)
- [Seq2Seeq + Attention](ncc/models/summarization/seq2seq.py)
- [Nary-](ncc/models/summarization/nary_tree2seq.py)/[ChildSum-](ncc/models/summarization/child_sum_tree2seq.py)Tree2Seq
- [Code2Seq](ncc/models/summarization/code2seq.py)
- [Transformer + (Sinusoidal/Relative/Learned Position Encoding)](ncc/models/transfomer)
- [CodeBERT](run/translation/codebert/model.py)
- [GraphCodeBERT](run/translation/graphcodebert/model.py)
- [PLBART](ncc/models/transfomer)

#### Structural Analysis of Pre-Trained Language Models for Source Code
- [ICSE 2022](examples/structural_analysis/)
  
## 📋 Experiments

### [Code Summarization](run/summarization)
Dataset: [Python (Wan et al.)](ncc_dataset/python_wan/README.md)

|                 | BLEU-4 | METEOR | ROUGE-L | Cost    | Logs    |
|-----------------|--------|--------|---------|---------|---------|
| Seq2Seq+Attn    | 25.57  | 14.40  | 39.41   | 0.09s/b | [click here](run/summarization/seq2seq/config/python_wan/python.log) |
| Tree2Seq+Attn   | 23.35  | 12.59  | 36.49   | 0.48s/b | [click here](run/summarization/tree2seq/config/python_wan/python.log) |
| Transformer     | 30.64  | 17.65  | 44.59   | 0.26s/b | [click here](run/summarization/transformer/config/python_wan/python.log) |
| Transformer+RPE | 31.57  | 17.74  | 45.18   | 0.27s/b | [click here](run/summarization/neural_transformer/relative/python_wan/python.log) |
| PLBART          | 32.71  | 18.13  | 46.05   | 0.80s/b | [TBC]() |


### [Code Retrieval](run/retrieval)
Dataset: [CodeSearchNet (Husain et al.)](ncc_dataset/codesearchnet/README.md)

| MRR      | Go    | Java  | JS    | PHP   | Python | Ruby  | Cost    | Logs    |
|----------|-------|-------|-------|-------|--------|-------|---------|---------|
| NBOW     | 66.59 | 59.92 | 47.15 | 54.75 | 63.33  | 42.86 | 0.16s/b | [click here](run/retrieval/nbow/config/csn/all.log) |
| ConV1d   | 70.87 | 60.49 | 38.81 | 61.92 | 67.29  | 36.53 | 0.30s/b | [click here](run/retrieval/conv1d/config/csn/all.log) |
| BiRNN    | 65.80 | 48.60 | 23.23 | 51.36 | 48.28  | 19.35 | 0.74s/b | [click here](run/retrieval/birnn/config/csn/all.log) |
| SelfAttn | 78.45 | 66.55 | 50.38 | 65.78 | 79.09  | 47.96 | 0.25s/b | [click here](run/retrieval/selfattn/config/csn/all.log) |


### [Code Completion](run/completion)
Dataset: Py150 [(official processed)](ncc_dataset/py150/README.md) [(raw)](ncc_dataset/raw_py150/README.md)

| MRR       | Attr  | Num   | Name   | Param | Tokens | Cost    | Logs    |
|-----------|-------|-------|--------|-------|--------|---------|---------|
| LSTM      | 51.67 | 47.45 | 46.52  | 66.06 | 73.73  | 0.31s/b | [click here](run/completion/seqrnn/config/raw_py150/python.log) |
| GPT-2     | 70.37 | 62.20 | 63.84  | 73.54 | 82.17  | 0.43s/b | [click here](run/completion/gpt2/config/raw_py150/python.log) |
| TravTrans | 72.08 | 68.55 | 76.33  | 71.08 | 83.17  | 0.43s/b | [click here](run/completion/trav_trans/config/py150/python.log) |

### [Type Inference](run/type_prediction)
Dataset: [CodeSearchNet-Java (Husain et al.)](ncc_dataset/codesearchnet/README.md)

|             | Acc@1 (All types) | Acc@5 (All types) | Acc@1 (Any types) | Acc@5 (Any types) | Cost    | Logs    |
|-------------|-------------------|-------------------|-------------------|-------------------|---------|---------|
| DeepTyper   | 0.52              | 0.67              | 0.43              | 0.67              | 0.42s/b | [TBC]() |
| Transformer | 0.32              | 0.64              | 0.37              | 0.75              | 0.85s/b | [TBC]() |

### [Heterogeneous Mapping](run/mapping)
Dataset: [OpenCL (Grewe et al.)](ncc_dataset/opencl/README.md)

| Accuracy        | AMD      | NVIDIA  |
|-----------------|----------|---------|
| Static mapping | 58.82    | 56.91    |
| Decision tree | 70.29    | 74.56    |
| Inst2vec | 82.79    | 81.76    |
| DeepTune | 83.24    | 80.15    |



## 🏫 Examples & Tutorials

> All the running commands here should be executed in the root of project folder (the path of your `naturalcc`). For example, in my environment I will stay at `/data/wanyao/Dropbox/ghproj-v100/naturalcc`.
> 
> We also have more detailed [READMEs](examples) to start your tutorial of NaturalCC.

### Step 1: Download and process a dataset from ```datasets```, and follow the instructions from the README.md file.
```shell
# ref: dataset/python_wan/README.md
# download dataset
bash dataset/python_wan/download.sh
# clean data
python -m dataset.python_wan.clean
# cast data attributes into different files
python -m dataset.python_wan.attributes_cast

# ref: dataset/python_wan/summarization/README.md
# save code tokens and docstirng tokens into MMAP format
python -m dataset.python_wan.summarization.preprocess
```

### Step 2 (optional): Register your self-defined models
- If you want to create a new model, please add your model at ```ncc/models``` and ```ncc/modules```.

- If your training policy are more complex than we thought, you should update your criterions and training procedure at ```ncc/criterions``` and ```ncc/trainers```, respectively.
  <br>

  Do not forget to update your self defined module at ```ncc/XX/__init__.py```.

### Step 3: Training and inference.
- Select a task and a model from [task list](run/) and follow the instructions in its README.md to start your learning.
```shell
# ref: run/summarization/transformer/README.md
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.transformer.train -f config/python_wan/python > run/summarization/transformer/config/python_wan/python.log 2>&1 &
# inference
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.transformer.eval -f config/python_wan/python -o run/summarization/transformer/config/python_wan/python.txt
```

# ❓ FAQ
Please fell free to contact me if you have any troubles.

## 😘 License and Acknowledgement

NaturalCC is [MIT-licensed](https://github.com/CGCL-codes/naturalcc/blob/master/LICENSE.txt). The license applies to the
pre-trained models as well. This project is also highly inspired by [Fairseq](https://github.com/pytorch/fairseq)
and [AllenNLP](https://allennlp.org).

## 🔗 Related Links
[Paper](https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf) <br>
[NaturalCC-demo](http://121.43.116.104:3000/) <br>
About us: [XCodeMind](https://xcodemind.github.io/team) <br>


## ❤️ Citation

Please cite as:

```
@inproceedings{wan2022naturalcc,
              author    = {Yao Wan and
                           Yang He and
                           Zhangqian Bi and
                           Jianguo Zhang and
                           Yulei Sui and
                           Hongyu Zhang and
                           Kazuma Hashimoto and
                           Hai Jin and
                           Guandong Xu and
                           Caiming Xiong and
                           Philip S. Yu},
              title     = {NaturalCC: An Open-Source Toolkit for Code Intelligence},
              booktitle   = {Proceedings of 44th International Conference on Software Engineering, Companion Volume},
              publisher = ACM,
              year      = {2022}
            }
```
