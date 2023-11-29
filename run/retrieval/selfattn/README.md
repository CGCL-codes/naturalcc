# SelfAttn for code retrieval task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn/all > run/retrieval/selfattn/config/csn/all.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn/ruby > run/retrieval/selfattn/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn/all
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn/ruby
```

## CodeSearchNet_feng
```shell script
# multi-task
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/all > run/retrieval/selfattn/config/csn_feng/all.log 2>&1 &
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/all

# single-task
# ruby
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/ruby > run/retrieval/selfattn/config/csn_feng/ruby.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/ruby
# python
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/python > run/retrieval/selfattn/config/csn_feng/python.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/python
# php
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/php > run/retrieval/selfattn/config/csn_feng/php.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/php
# java
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/java > run/retrieval/selfattn/config/csn_feng/java.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/java
# javascript
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/javascript > run/retrieval/selfattn/config/csn_feng/javascript.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/javascript
# go
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn_feng/go > run/retrieval/selfattn/config/csn_feng/go.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn_feng/go
```