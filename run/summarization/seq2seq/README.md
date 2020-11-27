# Seq2Seq for code summarization task

running with float32
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/python_wan > run/summarization/seq2seq/config/seq2seq.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan
# eval
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.eval -f config/python_wan
```
running with float16
```shell script
# train
CUDA_VISIBALE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/python_wan.fp16 > run/summarization/seq2seq/config/seq2seq.fp16.log 2>&1 &
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/python_wan.fp16
# eval, ours: BLEU: 20.88     ROUGE-L: 36.68  METEOR: 11.94
CUDA_VISIBALE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.eval -f config/python_wan.fp16
```