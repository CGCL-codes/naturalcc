# Code Summarization & Translation

This README contains instructions for code summarization models as well as code translation models.



## Datasets & Pre-Processing

### code summarization
| Dataset | Description | Download & Pre-Processing | Processing for Modalities |
|:-------:|:-----------:|:-------------------------:|:------------:|
| CodeSearchNet(Feng)  |  CodeBERT <br> ([Feng et al., 2020](https://arxiv.org/abs/2002.08155v1))   |  [README.md](dataset/codesearchnet_feng/README.md)     |  [Code Tokens](dataset/codesearchnet_feng/summarization/README.md) <br> [BPE Code Tokens](dataset/codesearchnet_feng/summarization_bpe/README.md) <br> [AST](dataset/codesearchnet_feng/sum_bin_ast/README.md) <br>  |
|  CodeXGLUE-summarization-CSN   |   CodeXGLUE <br> ([Lu et al., 2020](https://arxiv.org/abs/2102.04664))  |    [README.md](dataset/codexglue/code_to_text/README.md)       |   [BPE Code Tokens](dataset/codexglue/code_to_text/summarization/README.md) <br>    |
|  Python-code-docstring   |   Python-code-docstring <br> ([Yao et al., 2018](http://wanyao.me/pubs/2018-ASE-code-summarization.pdf))  |    [README.md](dataset/python_wan/README.md)       |   [Code Tokens](dataset/python_wan/summarization/README.md) <br>    |
|  StackOverflow   |   Python/C\#/SQL <br> ([Iyer et al., 2014](http://sandcat.cs.washington.edu/papers/Iyer-acl-2016.pdf))  |    [README.md](dataset/stackoverflow/README.md)       |   [Code Tokens](dataset/stackoverflow/README.md) <br>    |
|  Java   |   Deepcom <br> ([Hu et al., 2017](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5295&context=sis_research))  |    [README.md](dataset/deepcom/readme.md)       |   [Code Tokens](dataset/deepcom/readme.md) <br>    |
|  Java   |   Transformer + Relative Encoding <br> ([Ahmad et al., 2020](https://aclanthology.org/2020.acl-main.449.pdf))  |    [README.md](dataset/java_hu/README.md)       |   [Code Tokens](dataset/java_hu/README.md) <br>    |


### code translation
| Dataset | Description | Download & Pre-Processing | Processing for Modalities |
|:-------:|:-----------:|:-------------------------:|:------------:|
|  AVATAR   |  Multiple candaidates  <br> ([Ahmad et al., 2021](https://arxiv.org/abs/2108.11590))  |    [README.md](dataset/avatar/translation/README.md)       |   [BPE Code Tokens](dataset/avatar/translation/README.md) <br> [CodeBRT](dataset/avatar/translation/README.md) <br> [GraphCodeBERT](dataset/avatar/translation/README.md) <br>    |
|  CodeXGLUE-translation-CodeTrans   |  CodeXGLUE <br> ([Lu et al., 2020](https://arxiv.org/abs/2102.04664))  |    [README.md](dataset/codexglue/code_to_code/translation/README.md)       |   [Code Tokens](dataset/codexglue/code_to_code/translation/README.md) <br>   |




## Training & Inference

### [code summarization](run/summarization)
Example usage (Transformer + Relative Encoding)
Follow the instruction in [README.md](run/summarization/neural_transformer/README.md)
```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f vanilla/python_wan/python > run/summarization/neural_transformer/vanilla/python_wan/python.log 2>&1 &
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f vanilla/python_wan/python
```

### [code translation](run/translation)
Example usage (PLBART)
Follow the instruction in [README.md](run/translation/plbart/README.md)
```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/codetrans/java-csharp > run/translation/plbart/config/codetrans/java-csharp.log 2>&1 &
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/codetrans/java-csharp -o run/translation/plbart/config/codetrans/java-csharp.pred
```

## Models
- [Naive Copy](run/translation/naive_copy)
- [CodeNN](ncc/models/summarization/codenn.py)
- [DeepCom](ncc/models/summarization/deepcom.py)
- [Seq2Seeq + Attention](ncc/models/summarization/seq2seq.py)
- [Nary-](ncc/models/summarization/nary_tree2seq.py)/[ChildSum-](ncc/models/summarization/child_sum_tree2seq.py)Tree2Seq
- [Code2Seq](ncc/models/summarization/code2seq.py)
- [Transformer + (Sinusoidal/Relative/Learned)](ncc/models/transfomer)
- [CodeBERT](run/translation/codebert/model.py)
- [GraphCodeBERT](run/translation/graphcodebert/model.py)
- [PLBART](ncc/models/transfomer)