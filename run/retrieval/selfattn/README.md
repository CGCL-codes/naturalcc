# SelfAttn for code retrieval task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn/all > run/retrieval/selfattn/config/csn/all.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.selfattn.train -f config/csn/ruby > run/retrieval/selfattn/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn/all -o run/retrieval/selfattn/config/csn/all.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.selfattn.eval -f config/csn/ruby
```