# NeuralTransformer for code completion task

running with float32
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.completion.seqrnn.train -f config/py150 > run/completion/seqrnn/config/seqrnn.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.train -f config/py150
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.eval -f config/py150
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.predictor
```

running with float16
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.completion.seqrnn.train -f config/py150.fp16 > run/completion/seqrnn/config/seqrnn.fp16.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.train -f config/py150.fp16
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.eval -f config/py150.fp16
```