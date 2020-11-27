# NeuralTransformer for code summarization task

running with float32
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f config/python_wan > run/summarization/neural_transformer/config/neural_transformer.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.train -f config/python_wan
# eval, ours BLEU: 28.69     ROUGE-L: 40.70  METEOR: 15.64
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f config/python_wan
```

running with float16
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f config/python_wan.fp16 > run/summarization/neural_transformer/config/neural_transformer.fp16.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.train -f config/python_wan.fp16
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f config/python_wan.fp16
```