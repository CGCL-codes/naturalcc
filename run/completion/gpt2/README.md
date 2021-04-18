# NeuralTransformer for code completion task

Raw-Py150 dataset

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/raw_py150/python > run/completion/gpt2/config/raw_py150/python.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/raw_py150/python
#[all] tokens, accuracy: 0.756293, MRR: 0.821723
#[attr] tokens, accuracy: 0.627287, MRR: 0.703742
#[num] tokens, accuracy: 0.504472, MRR: 0.621987
#[name] tokens, accuracy: 0.548266, MRR: 0.638393
#[param] tokens, accuracy: 0.697878, MRR: 0.735352
```

<br>

CodeXGlue(code-to-text, csn_feng) dataset

| single | python   | ruby     | go       | php      | java     | javascript |
|-----|----------|----------|----------|----------|----------|------------|
| acc | 0.632133 | 0.427932 | 0.709576 | 0.736956 | 0.694996 | 0.611418   |
| mrr | 0.726944 | 0.546986 | 0.784151 | 0.814174 | 0.777815 | 0.709278   |
| train_size | 251820 | 24927 | 167288 | 241241 | 164923 | 58025  |

| multi_task | python   | ruby     | go       | php      | java     | javascript |
|-----|----------|----------|----------|----------|----------|------------|
| acc | 0.638678 | 0.531883 | 0.731481 | 0.747818 | 0.71073 | 0.659851   |
| mrr | 0.733002 | 0.64119 | 0.802455 | 0.823614 | 0.792368 | 0.750832   |

```shell script
# python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/python > run/completion/gpt2/config/csn_feng/python.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/python
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/python


# ruby
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/ruby > run/completion/gpt2/config/csn_feng/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/ruby
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/ruby


# go
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/go > run/completion/gpt2/config/csn_feng/go.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/go
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/go


# php
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/php > run/completion/gpt2/config/csn_feng/php.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/php
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/php


# java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/java > run/completion/gpt2/config/csn_feng/java.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/java
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/java


# javascript
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/javascript > run/completion/gpt2/config/csn_feng/javascript.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/javascript
# generate for kd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.gen_out -f config/csn_feng/javascript


# all
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/all > run/completion/gpt2/config/csn_feng/all.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/all


# predictor
```