# transformer in Code Translation Task

```shell
# top5
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top5/java-python > run/translation/transformer/config/avatar/top5/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top5/java-python -o run/translation/transformer/config/avatar/top5/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top5/python-java > run/translation/transformer/config/avatar/top5/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top5/python-java -o run/translation/transformer/config/avatar/top5/python-java.pred

# top3
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top3/java-python > run/translation/transformer/config/avatar/top3/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top3/java-python -o run/translation/transformer/config/avatar/top3/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top3/python-java > run/translation/transformer/config/avatar/top3/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top3/python-java -o run/translation/transformer/config/avatar/top3/python-java.pred

# top1
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top1/java-python > run/translation/transformer/config/avatar/top1/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top1/java-python -o run/translation/transformer/config/avatar/top1/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/avatar/top1/python-java > run/translation/transformer/config/avatar/top1/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/avatar/top1/python-java -o run/translation/transformer/config/avatar/top1/python-java.pred
```

```shell
# java -> csharp
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/codetrans/java-csharp > run/translation/transformer/config/codetrans/java-csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/codetrans/java-csharp -o run/translation/transformer/config/codetrans/java-csharp.pred

# csharp -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.transformer.train -f config/codetrans/csharp-java > run/translation/transformer/config/codetrans/csharp-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.transformer.eval -f config/codetrans/csharp-java -o run/translation/transformer/config/codetrans/csharp-java.pred
```