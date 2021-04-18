# NeuralTransformer for code summarization task

## 1) vanilla Transformer (Sin Positional Encoding)

running with float32

```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f vanilla/python_wan/python > run/summarization/neural_transformer/vanilla/python_wan/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f vanilla/python_wan/python
```

## 2) RPE Transformer (Relative Positional Encoding)

running with float32

```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/python_wan/python > run/summarization/neural_transformer/relative/python_wan/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f relative/python_wan/python -o run/summarization/neural_transformer/relative/python_wan/python.txt

# CSN(feng)
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/ruby > run/summarization/neural_transformer/relative/csn_feng/ruby.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/go > run/summarization/neural_transformer/relative/csn_feng/go.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/php > run/summarization/neural_transformer/relative/csn_feng/php.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/python > run/summarization/neural_transformer/relative/csn_feng/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/java > run/summarization/neural_transformer/relative/csn_feng/java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f relative/csn_feng/javascript > run/summarization/neural_transformer/relative/csn_feng/javascript.log 2>&1 &

# evaluation
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/ruby
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/go
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/php
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/python
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/java
python -m run.summarization.neural_transformer.eval -f relative/csn_feng/javascript
```

## 3) learned Transformer (Learned Positional Encoding)

running with float32

```shell
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.neural_transformer.train -f learned/python_wan/python > run/summarization/neural_transformer/learned/python_wan/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.neural_transformer.train -f learned/python_wan/python > run/summarization/neural_transformer/learned/python_wan/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.neural_transformer.eval -f learned/python_wan/python
```
