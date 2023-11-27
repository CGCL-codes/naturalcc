# graphcodebert in Code Translation Task

```shell
topk5
node13
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG java --TGT_LANG python --topk 5 --dataset avatar > run/translation/graphcodebert/config/avatar/topk5/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG java --TGT_LANG python --topk 5 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk5/java-python.pred

node13
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG python --TGT_LANG java --topk 5 --dataset avatar > run/translation/graphcodebert/config/avatar/topk5/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG python --TGT_LANG java --topk 5 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk5/python-java.pred

topk3
node15
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG java --TGT_LANG python --topk 3 --dataset avatar > run/translation/graphcodebert/config/avatar/topk3/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG java --TGT_LANG python --topk 3 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk3/java-python.pred

# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG python --TGT_LANG java --topk 3 --dataset avatar > run/translation/graphcodebert/config/avatar/topk3/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG python --TGT_LANG java --topk 3 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk3/python-java.pred

topk1
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG java --TGT_LANG python --topk 1 --dataset avatar > run/translation/graphcodebert/config/avatar/topk1/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG java --TGT_LANG python --topk 1 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk1/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG python --TGT_LANG java --topk 1 --dataset avatar > run/translation/graphcodebert/config/avatar/topk1/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG python --TGT_LANG java --topk 1 --dataset avatar -o run/translation/graphcodebert/config/avatar/topk1/python-java.pred
```

```shell
# java -> csharp
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG java --TGT_LANG csharp --dataset codetrans > run/translation/graphcodebert/config/codetrans/java-csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG java --TGT_LANG csharp --dataset codetrans -o run/translation/graphcodebert/config/codetrans/java-csharp.pred
# csharp -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.graphcodebert.train --SRC_LANG csharp --TGT_LANG java --dataset codetrans > run/translation/graphcodebert/config/codetrans/csharp-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.graphcodebert.eval --SRC_LANG csharp --TGT_LANG java --dataset codetrans -o run/translation/graphcodebert/config/codetrans/csharp-java.pred
```