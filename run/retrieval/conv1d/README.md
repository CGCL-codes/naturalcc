# ResNet + ConV1d for code retrieval task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.conv1d.train -f config/csn/all > run/retrieval/conv1d/config/csn/all.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.nbow.train -f config/csn/ruby > run/retrieval/nbow/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.conv1d.eval -f config/csn/all -o run/retrieval/conv1d/config/csn/all.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.conv1d.eval -f config/csn/ruby
```