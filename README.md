<p align="center">
  <br>
  <img src="docs/naturalcc_logo.png" width="400">
  <br>
</p>
<div align="center">
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

<a href="https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf">Paper</a>,
<a href="http://121.43.116.104:3000">Demo</a>,
<a href="https://xcodemind.github.io/team">About us-XCodeMind</a>

# NaturalCC - Natural Code Comprehension

</div>

## 📖 Introduction

NaturalCC is a sequence modeling toolkit designed to bridge the gap between programming and natural languages through advanced machine learning techniques. It allows researchers and developers to train custom models for a variety of software engineering tasks, e.g., code generation, code completion, code summarization, code retrieval, code clone detection, and type inference.

Key features of NaturalCC include:

- **Modular and Extensible Framework:** Built on the robust Fairseq's registry mechanism, allowing for easy adaptation and extension to diverse software engineering tasks.
- **Datasets and Preprocessing Tools+:** Offers access to a variety of clean, preprocessed benchmarks such as Human-Eval, CodeSearchNet, Python-Doc, and Py150. Comes equipped with scripts for feature extraction using compiler tools like LLVM.
- **Support for Large Code Models:** Incorporates state-of-the-art large code models like Code Llama, CodeT5, CodeGen, and StarCoder.
- **Benchmarking and Evaluation**: Benchmarks multiple downstream tasks (including code generation and code completion), with evaluation capabilities on well-known benchmarks using popular metrics like pass@k.
- **Optimized for Efficiency:** Employs the `NCCL` library and `torch.distributed` for high-efficiency model training across multiple GPUs. Supports both full-precision (`FP32`) and half-precision (`FP16`) computations to accelerate training and inference processes.
- **Enhanced Logging for Improved Debugging:** Advanced logging features to provide clear, detailed feedback during model training and operation, aiding in debugging and performance optimization.

## ✨ Latest News

- **[Nov 25, 2023]** **NaturalCC 2.0 Released!** Now compatible with [Transformers](https://github.com/huggingface/transformers) and supporting popular large code models like Code Llama, CodeT5, CodeGen, and StarCoder from [Hugging Face](https://huggingface.co). Access the previous version in the [ncc1]() branch.
- **[Apr 19, 2023]** Integrated the source code of "You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search" into NaturalCC.
- **[Jan 25, 2022]** Our paper introducing the NaturalCC toolkit was accepted at the ICSE 2022 Demo Track.
- **[May 10, 2022]** Merged the source code of "What Do They Capture? - A Structural Analysis of Pre-Trained Language Models for Source Code" into NaturalCC.

## 🛠️ Installation Guide

To get started with NaturalCC, ensure your system meets the following requirements:

- GCC/G++ version 5.0 or higher
- NVIDIA GPU, NCCL, and the Cuda Toolkit for training new models (optional but recommended)
- NVIDIA's apex library for faster training (optional)

Follow these steps to set up the environment.

1. (Optional) Creating conda environment
    ```shell
    conda create -n naturalcc python=3.6
    conda activate naturalcc
    ```

2. Building NaturalCC from source
    ```shell
    git clone https://github.com/CGCL-codes/naturalcc && cd naturalcc
    pip install -r requirements.txt
    pip install --editable ./
    ```
3. Installing Additional Dependencies
    ```shell
    pip install -q -U git+https://github.com/huggingface/transformers.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    ```

4. HuggingFace Token for Certain Models

    For models like [StarCoder](https://github.com/bigcode-project/starcoder), a HuggingFace token is required. Log in to HuggingFace using:
    ```
    huggingface-cli login
    ```

## 🚀 Quick Start

### Example 1: Code Generation


### Example 2: Code Summarization

1. Download and process a dataset from ```datasets```, and follow the instructions from the README.md file.
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

2. Register your self-defined models
    - If you want to create a new model, please add your model at ```ncc/models``` and ```ncc/modules```.

    - If your training policy are more complex than we thought, you should update your criterions and training procedure at ```ncc/criterions``` and ```ncc/trainers```, respectively.
      <br>

      Do not forget to update your self defined module at ```ncc/XX/__init__.py```.

3. Training and inference.
    - Select a task and a model from [task list](run/) and follow the instructions in its README.md to start your learning.
    ```shell
    # ref: run/summarization/transformer/README.md
    # train
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.transformer.train -f config/python_wan/python > run/summarization/transformer/config/python_wan/python.log 2>&1 &
    # inference
    CUDA_VISIBLE_DEVICES=0 python -m run.summarization.transformer.eval -f config/python_wan/python -o run/summarization/transformer/config/python_wan/python.txt
    ```

> We also have more detailed [READMEs](examples) to start your tutorial of NaturalCC.

## 📚 Dataset

NaturalCC supports a diverse range of datasets, catering to various aspects of code analysis and processing. These datasets include:

- [Python (Wan et al.)](preprocessing/python_wan/README.md)
- [CodeSearchNet (Husain et al.)](preprocessing/codesearchnet/README.md)
- [CodeXGlue (Feng et al.)](preprocessing/codexglue/code_to_text/README.md)
- [Py150 (official processed)](preprocessing/py150/README.md) [(raw)](preprocessing/raw_py150/README.md)
- [OpenCL (Grewe et al.)](preprocessing/opencl/README.md)
- [Java (Hu et, al.)](preprocessing/java_hu/README.md)
- [Stack Overflow](preprocessing/stackoverflow/README.md)
- [DeepCS (Gu et al.)](preprocessing/deepcs)
- [AVATAR (Ahmad et al.)](preprocessing/avatar)
- [StackOverflow (Iyer et al.)](preprocessing/stackoverflow)

## 🤖 Implementations

NaturalCC includes a wide range of implementations for different tasks in code analysis. These are categorized as follows:

#### Code retrieval (search)

- [NBOW](ncc/models/retrieval/nbow.py): Neural Bag-of-Words model for code retrieval.
- [BiRNN](ncc/models/retrieval/birnn.py): Bidirectional Recurrent Neural Network implementation.
- [1D-CNN](ncc/models/retrieval/birnn.py): One-dimensional Convolutional Neural Network.
- [SelfAttn](ncc/models/retrieval/self_attn.py): Self-Attention model for code retrieval.
- [Deepcs](ncc/models/retrieval/deepcs.py): Deep Code Search model.

#### Code completion

- [SeqRNN](ncc/models/completion/seqrnn.py): Sequential RNN for code completion.
- [GPT2](ncc/models/completion/gpt2.py): GPT-2 model adapted for code completion.

#### Heterogeneous mapping

- [static mapping](run/mapping/static_mapping): A static approach to code-feature mapping.
- [decision tree](run/mapping/decision_tree): Decision tree-based mapping method.
- [deeptune](run/mapping/deeptune): Deep learning-based code-feature mapping.
- [inst2vec](run/mapping/inst2vec): Instruction-to-Vector mapping model.

#### Code summarization

- [Naive Copy](run/translation/naive_copy): A simple copy-based summarization approach.
- [CodeNN](ncc/models/summarization/codenn.py): Neural Network model for code summarization.
- [DeepCom](ncc/models/summarization/deepcom.py): Deep Comment Generation model.
- [Seq2Seeq + Attention](ncc/models/summarization/seq2seq.py): Sequence-to-Sequence model with attention mechanism.
- [Nary-](ncc/models/summarization/nary_tree2seq.py)/[ChildSum-](ncc/models/summarization/child_sum_tree2seq.py)Tree2Seq: Nary-Tree2Seq and ChildSum-Tree2Seq models.
- [Code2Seq](ncc/models/summarization/code2seq.py): Code-to-Sequence model.
- [Transformer + (Sinusoidal/Relative/Learned Position Encoding)](ncc/models/transfomer): Transformers with various position encoding techniques.
- [CodeBERT](run/translation/codebert/model.py): CodeBERT model for summarization.
- [GraphCodeBERT](run/translation/graphcodebert/model.py): GraphCodeBERT model implementation.
- [PLBART](ncc/models/transfomer): Pre-trained Language model for BART architecture.

#### Structural Analysis of Pre-Trained Language Models for Source Code

- [ICSE 2022](examples/structural_analysis/)

#### Poisoning Vulnerabilities in Neural Code Search

- [FSE 2022](examples/code-backdoor/)

## 📋 Experiments

### [Code Summarization](run/summarization)

Dataset: [Python (Wan et al.)](preprocessing/python_wan/README.md)

|                 | BLEU-4 | METEOR | ROUGE-L | Cost    | Logs    |
|-----------------|--------|--------|---------|---------|---------|
| Seq2Seq+Attn    | 25.57  | 14.40  | 39.41   | 0.09s/b | [click here](run/summarization/seq2seq/config/python_wan/python.log) |
| Tree2Seq+Attn   | 23.35  | 12.59  | 36.49   | 0.48s/b | [click here](run/summarization/tree2seq/config/python_wan/python.log) |
| Transformer     | 30.64  | 17.65  | 44.59   | 0.26s/b | [click here](run/summarization/transformer/config/python_wan/python.log) |
| Transformer+RPE | 31.57  | 17.74  | 45.18   | 0.27s/b | [click here](run/summarization/neural_transformer/relative/python_wan/python.log) |
| PLBART          | 32.71  | 18.13  | 46.05   | 0.80s/b | [TBC]() |

### [Code Retrieval](run/retrieval)

Dataset: [CodeSearchNet (Husain et al.)](preprocessing/codesearchnet/README.md)

| MRR      | Go    | Java  | JS    | PHP   | Python | Ruby  | Cost    | Logs    |
|----------|-------|-------|-------|-------|--------|-------|---------|---------|
| NBOW     | 66.59 | 59.92 | 47.15 | 54.75 | 63.33  | 42.86 | 0.16s/b | [click here](run/retrieval/nbow/config/csn/all.log) |
| ConV1d   | 70.87 | 60.49 | 38.81 | 61.92 | 67.29  | 36.53 | 0.30s/b | [click here](run/retrieval/conv1d/config/csn/all.log) |
| BiRNN    | 65.80 | 48.60 | 23.23 | 51.36 | 48.28  | 19.35 | 0.74s/b | [click here](run/retrieval/birnn/config/csn/all.log) |
| SelfAttn | 78.45 | 66.55 | 50.38 | 65.78 | 79.09  | 47.96 | 0.25s/b | [click here](run/retrieval/selfattn/config/csn/all.log) |

### [Code Completion](run/completion)

Dataset: Py150 [(official processed)](preprocessing/py150/README.md) [(raw)](preprocessing/raw_py150/README.md)

| MRR       | Attr  | Num   | Name   | Param | Tokens | Cost    | Logs    |
|-----------|-------|-------|--------|-------|--------|---------|---------|
| LSTM      | 51.67 | 47.45 | 46.52  | 66.06 | 73.73  | 0.31s/b | [click here](run/completion/seqrnn/config/raw_py150/python.log) |
| GPT-2     | 70.37 | 62.20 | 63.84  | 73.54 | 82.17  | 0.43s/b | [click here](run/completion/gpt2/config/raw_py150/python.log) |
| TravTrans | 72.08 | 68.55 | 76.33  | 71.08 | 83.17  | 0.43s/b | [click here](run/completion/trav_trans/config/py150/python.log) |

### [Type Inference](run/type_prediction)

Dataset: [CodeSearchNet-Java (Husain et al.)](preprocessing/codesearchnet/README.md)

|             | Acc@1 (All types) | Acc@5 (All types) | Acc@1 (Any types) | Acc@5 (Any types) | Cost    | Logs    |
|-------------|-------------------|-------------------|-------------------|-------------------|---------|---------|
| DeepTyper   | 0.52              | 0.67              | 0.43              | 0.67              | 0.42s/b | [TBC]() |
| Transformer | 0.32              | 0.64              | 0.37              | 0.75              | 0.85s/b | [TBC]() |

### [Heterogeneous Mapping](run/mapping)

Dataset: [OpenCL (Grewe et al.)](preprocessing/opencl/README.md)

| Accuracy        | AMD      | NVIDIA  |
|-----------------|----------|---------|
| Static mapping | 58.82    | 56.91    |
| Decision tree | 70.29    | 74.56    |
| Inst2vec | 82.79    | 81.76    |
| DeepTune | 83.24    | 80.15    |

## 🤝 About Contributing

We warmly welcome contributions to NaturalCC! Your involvement is essential for keeping NaturalCC innovative and accessible. 

We're grateful to all our amazing contributors who have made this project what it is today!

<a href="https://github.com/CGCL-codes/naturalcc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CGCL-codes/naturalcc&r="  width="800px"/>
</a>

## 💡 FAQ

If you have any questions or encounter issues, please feel free to reach out. For quick queries, you can also check our `Issues` page for common questions and solutions.

## 😘 License and Acknowledgement

**License:** NaturalCC is open-sourced under the [MIT-licensed](https://github.com/CGCL-codes/naturalcc/blob/master/LICENSE.txt). This permissive license applies not only to the toolkit itself but also to the pre-trained models provided within.

**Acknowledgements:** We extend our heartfelt gratitude to the broader open-source community, particularly drawing inspiration from projects like [Fairseq](https://github.com/pytorch/fairseq) for their advanced sequence-to-sequence models, and [AllenNLP](https://allennlp.org) for their robust NLP components. Their groundbreaking work has been instrumental in shaping the development of NaturalCC.

## 📄 Citation

We're thrilled that you're interested in using NaturalCC for your research or applications! Citing our work helps us to grow and continue improving this toolkit. You can find more in-depth details about NaturalCC in our [paper](https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf).

If you use NaturalCC in your research, please consider citing our paper. Below is the BibTex format for citation:

```
@inproceedings{wan2022naturalcc,
  title={NaturalCC: An Open-Source Toolkit for Code Intelligence},
  author={Yao Wan and Yang He and Zhangqian Bi and Jianguo Zhang and Yulei Sui and Hongyu Zhang and Kazuma Hashimoto and Hai Jin and Guandong Xu and Caiming Xiong and Philip S. Yu},
  booktitle={Proceedings of 44th International Conference on Software Engineering, Companion Volume},
  publisher=ACM,
  year={2022}
}
```
