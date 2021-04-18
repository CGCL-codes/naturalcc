# NBOW for code retrieval task

```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn/all > run/retrieval/nbow/config/csn/all.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn/ruby > run/retrieval/nbow/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn/all -o run/retrieval/nbow/config/csn/all.txt
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.retrieval.nbow.eval -f config/csn/ruby
```