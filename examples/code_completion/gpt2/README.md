### Step 1: Download the raw_py150 dataset
```
bash download.sh
```

### Step 2: Flatten the dataset
```
python -m preprocessing.raw_py150.attributes_cast --raw_dataset_dir /home/wanyao/raw_py150/raw --attributes_dir /home/wanyao/raw_py150/attributes --cores 4
```

### Step 3: Binarize the dataset
```
ncc-preprocess --source-lang code_tokens --trainpref /home/wanyao/raw_py150/attributes/train --testpref /home/wanyao/raw_py150/attributes/test --only-source --destdir /mnt/silver/yanrunbang/naturalcc/data-bin/raw_py150 --workers 4
```

### Step 4: Postprocess the dataset for code compeltion task
```
python -m preprocessing.raw_py150.postprocess --dataset-dir /mnt/silver/yanrunbang/naturalcc/data-bin/raw_py150 --language code_tokens
```


### Step 5: Train your model
```
ncc-train --task code_completion \
  /mnt/silver/yanrunbang/naturalcc/data-bin/raw_py150 \
  --save-dir checkpoints/transformer_raw_py150 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000 \
  --disable-validation
```





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

# sentence length: 256

| single | python   | ruby     | go       | php      | java     | javascript |
|-----|----------|----------|----------|----------|----------|------------|
| acc | 0.6287 | 0.448355 | 0.708251 | 0.73665 | 0.696034 | 0.617656   |
| mrr | 0.724431 | 0.563861 | 0.7831 | 0.814299 | 0.778502 | 0.71414   |
| train_size | 251820 | 24927 | 167288 | 241241 | 164923 | 58025  |

| multi_task | python   | ruby     | go       | php      | java     | javascript |
|-----|----------|----------|----------|----------|----------|------------|
| acc | 0.634384 | 0.519304 | 0.726514 | 0.74291 | 0.705808 | 0.649352   |
| mrr | 0.729622 | 0.632337 | 0.798286 | 0.819752 | 0.788361 | 0.742858   |

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

# javascript_php_python
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/javascript_php_python > run/completion/gpt2/config/csn_feng/javascript_php_python.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/javascript_php_python

# ruby_js
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/ruby_js > run/completion/gpt2/config/csn_feng/ruby_js.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/ruby_js

# ruby_js_java
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/ruby_js_java > run/completion/gpt2/config/csn_feng/ruby_js_java.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/ruby_js_java

# ruby_js_java_go
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/ruby_js_java_go > run/completion/gpt2/config/csn_feng/ruby_js_java_go.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/ruby_js_java_go

# ruby_js_java_go_php
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.gpt2.train -f config/csn_feng/ruby_js_java_go_php > run/completion/gpt2/config/csn_feng/ruby_js_java_go_php.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.gpt2.eval -f config/csn_feng/ruby_js_java_go_php

```