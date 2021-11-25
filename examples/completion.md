# Code Completion

This README contains instructions for code completion models.



## Datasets & Pre-Processing

### code completion
| Dataset | Description | Download & Pre-Processing | Processing for Modalities |
|:-------:|:-----------:|:-------------------------:|:------------:|
| CodeSearchNet(Feng)  |  CodeBERT <br> ([Feng et al., 2020](https://arxiv.org/abs/2002.08155v1))   |  [README.md](dataset/codesearchnet_feng/README.md)     |  [Code Tokens](dataset/codesearchnet_feng/summarization/README.md)  |
| Py150  |  Py150 & Py150(raw)   |  [README.md](dataset/py150/README.md) <br>  [README.md(raw)](dataset/raw_py150/README.md)    |  [Py150](dataset/raw_py150/README.md) <br> [Py150(raw)](dataset/raw_py150/README.md) <br>  |

## Training & Inference

### [code completion](run/completion)
Example usage (GPT2)
Follow the instruction in [README.md](run/completion/deeptune/README.md)
```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/python > run/completion/gpt2/config/csn_feng/python.log 2>&1 &
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/python
```

## Models
- [SeqRNN](ncc/models/completion/seqrnn.py)
- [GPT2](ncc/models/completion/gpt2.py)