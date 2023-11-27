# DeepCS for code retrieval task

```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.deepcs.train -f config/java > run/retrieval/deepcs/config/java.log 2>&1 &
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.deepcs.eval -f config/java
```