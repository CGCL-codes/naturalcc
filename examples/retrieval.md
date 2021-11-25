# Code Retrieval (Search)

This README contains instructions for code retrieval (search) models.



## Datasets & Pre-Processing

### code retrieval
| Dataset | Description | Download & Pre-Processing | Processing for Modalities |
|:-------:|:-----------:|:-------------------------:|:------------:|
| CodeSearchNet  |  4 baselines <br> ([Husain et al., 2020](https://arxiv.org/abs/1909.09436))   |  [README.md](dataset/codesearchnet/README.md)     |  [Code Tokens](dataset/codesearchnet/retrieval/README.md) <br>  |
| DeepCS  |  Deep Code Search <br> ([Gu et al., 2017](https://dl.acm.org/doi/10.1145/3180155.3180167))   |  [README.md](dataset/deepcs/README.md)     |  [Code Tokens](dataset/deepcs/README.md) <br>  |
|  AVATAR   |  Multiple candaidates  <br> ([Ahmad et al., 2021](https://arxiv.org/abs/2108.11590))  |    TBC       |   TBC    |

## Training & Inference

### [code retrieval](run/summarization)
Example usage (SelfAttn)
Follow the instruction in [README.md](run/retrieval/selfattn/README.md)
```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn/all > run/retrieval/selfattn/config/csn/all.log 2>&1 &
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn/all
```

## Models
- [NBOW](ncc/models/retrieval/nbow.py)
- [BiRNN](ncc/models/retrieval/birnn.py)
- [1D-CNN](ncc/models/retrieval/birnn.py)
- [SelfAttn](ncc/models/retrieval/self_attn.py)
- [Deepcs](ncc/models/retrieval/deepcs.py)