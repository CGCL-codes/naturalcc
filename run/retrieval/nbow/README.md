# NBOW for code retrieval task

## CodeSearchNet
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn/all > run/retrieval/nbow/config/csn/all.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn/ruby > run/retrieval/nbow/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn/all -o run/retrieval/nbow/config/csn/all.txt
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn/ruby
```

## CodeSearchNet_feng
```shell script
# multi-task
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/all > run/retrieval/nbow/config/csn_feng/all.log 2>&1 &
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/all -o run/retrieval/nbow/config/csn_feng/all.txt

# single-task
# ruby
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/ruby > run/retrieval/nbow/config/csn_feng/ruby.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/ruby -o run/retrieval/nbow/config/csn_feng/ruby.txt
# python
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/python > run/retrieval/nbow/config/csn_feng/python.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/python -o run/retrieval/nbow/config/csn_feng/python.txt
# php
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/php > run/retrieval/nbow/config/csn_feng/php.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/php -o run/retrieval/nbow/config/csn_feng/php.txt
# java
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/java > run/retrieval/nbow/config/csn_feng/java.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/java -o run/retrieval/nbow/config/csn_feng/java.txt
# javascript
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/javascript > run/retrieval/nbow/config/csn_feng/javascript.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/javascript -o run/retrieval/nbow/config/csn_feng/javascript.txt
# go
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn_feng/go > run/retrieval/nbow/config/csn_feng/go.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn_feng/go -o run/retrieval/nbow/config/csn_feng/go.txt
```