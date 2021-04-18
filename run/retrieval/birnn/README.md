# BiRNN for code retrieval task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/all > run/retrieval/birnn/config/csn/all.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/ruby > run/retrieval/birnn/config/csn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.birnn.eval -f config/csn/all -o run/retrieval/birnn/config/csn/all.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.retrieval.birnn.eval -f config/csn/all
```