# PLBART in Code Translation Task

## AVATAR

```shell
topk5
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk5/o2o/java-python > run/translation/plbart/config/avatar/topk5/o2o/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk5/o2o/java-python -o run/translation/plbart/config/avatar/topk5/o2o/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk5/o2o/python-java > run/translation/plbart/config/avatar/topk5/o2o/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk5/o2o/python-java -o run/translation/plbart/config/avatar/topk5/o2o/python-java.pred

python -m run.translation.plbart.config.avatar.topk5.eval

topk3
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk3/o2o/java-python > run/translation/plbart/config/avatar/topk3/o2o/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk3/o2o/java-python -o run/translation/plbart/config/avatar/topk3/o2o/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk3/o2o/python-java > run/translation/plbart/config/avatar/topk3/o2o/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk3/o2o/python-java -o run/translation/plbart/config/avatar/topk3/o2o/python-java.pred

python -m run.translation.plbart.config.avatar.topk3.eval

topk1
# java -> python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk1/o2o/java-python > run/translation/plbart/config/avatar/topk1/o2o/java-python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk1/o2o/java-python -o run/translation/plbart/config/avatar/topk1/o2o/java-python.pred
# python -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/avatar/topk1/o2o/python-java > run/translation/plbart/config/avatar/topk1/o2o/python-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/avatar/topk1/o2o/python-java -o run/translation/plbart/config/avatar/topk1/o2o/python-java.pred

python -m run.translation.plbart.config.avatar.topk1.eval
```

## CodeXGlue

```shell
# java -> csharp
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/codetrans/java-csharp > run/translation/plbart/config/codetrans/java-csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/codetrans/java-csharp -o run/translation/plbart/config/codetrans/java-csharp.pred

# csharp -> java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.translation.plbart.train -f config/codetrans/csharp-java > run/translation/plbart/config/codetrans/csharp-java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.translation.plbart.eval -f config/codetrans/csharp-java -o run/translation/plbart/config/codetrans/csharp-java.pred
```