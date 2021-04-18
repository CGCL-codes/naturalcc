# Seq2Seq for code summarization task


CSN(feng)
```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/ruby > run/summarization/seq2seq/config/csn_feng/ruby.log 2>&1 &
CUDA_VISIBLE_DEVICES=1,2,3 python -m run.summarization.seq2seq.train -f config/csn_feng/ruby
CUDA_VISIBLE_DEVICES=1,2,3 python -m run.summarization.seq2seq.train -f config/csn_feng/javascript
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/go > run/summarization/seq2seq/config/csn_feng/go.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/php > run/summarization/seq2seq/config/csn_feng/php.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/python > run/summarization/seq2seq/config/csn_feng/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/java > run/summarization/seq2seq/config/csn_feng/java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/csn_feng/javascript > run/summarization/seq2seq/config/csn_feng/javascript.log 2>&1 &

# evaluation
python -m run.summarization.seq2seq.eval -f config/csn_feng/ruby
python -m run.summarization.seq2seq.eval -f config/csn_feng/go
python -m run.summarization.seq2seq.eval -f config/csn_feng/php
python -m run.summarization.seq2seq.eval -f config/csn_feng/python
python -m run.summarization.seq2seq.eval -f config/csn_feng/java
python -m run.summarization.seq2seq.eval -f config/csn_feng/javascript
```

Python(wan) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/python_wan/python > run/summarization/seq2seq/config/python_wan/python.log 2>&1 &
# evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.eval -f config/python_wan/python -o run/summarization/seq2seq/config/python_wan/python.txt
```

Java(hu) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/java_hu/java > run/summarization/seq2seq/config/java_hu/java.log 2>&1 &
# evaluation
python -m run.summarization.seq2seq.eval -f config/java_hu/java
```


StackOverflow training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/stack_overflow/csharp > run/summarization/seq2seq/config/stack_overflow/csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/stack_overflow/sql > run/summarization/seq2seq/config/stack_overflow/sql.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/stack_overflow/python > run/summarization/seq2seq/config/stack_overflow/python.log 2>&1 &
# evaluation
python -m run.summarization.seq2seq.eval -f config/stack_overflow/csharp
python -m run.summarization.seq2seq.eval -f config/stack_overflow/sql
python -m run.summarization.seq2seq.eval -f config/stack_overflow/python
```


portion dataset training
```shell script
#BLEU: 20.88     ROUGE-L: 36.68  METEOR: 11.94
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p100 > run/summarization/seq2seq/config/portion/seq2seq.p100.log 2>&1 &
#BLEU: 20.55     ROUGE-L: 36.09  METEOR: 11.64
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p90 > run/summarization/seq2seq/config/portion/seq2seq.p90.log 2>&1 &
#BLEU: 19.32     ROUGE-L: 34.38  METEOR: 10.63
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p80 > run/summarization/seq2seq/config/portion/seq2seq.p80.log 2>&1 &
#BLEU: 19.26     ROUGE-L: 34.21  METEOR: 10.62
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p70 > run/summarization/seq2seq/config/portion/seq2seq.p70.log 2>&1 &
#BLEU: 18.64     ROUGE-L: 33.31  METEOR: 10.14
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p60 > run/summarization/seq2seq/config/portion/seq2seq.p60.log 2>&1 &
#BLEU: 17.91     ROUGE-L: 32.08  METEOR: 9.55
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p50 > run/summarization/seq2seq/config/portion/seq2seq.p50.log 2>&1 &
#BLEU: 17.26     ROUGE-L: 30.64  METEOR: 8.81
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p40 > run/summarization/seq2seq/config/portion/seq2seq.p40.log 2>&1 &
#BLEU: 16.49     ROUGE-L: 29.15  METEOR: 8.03
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p30 > run/summarization/seq2seq/config/portion/seq2seq.p30.log 2>&1 &
#BLEU: 15.55     ROUGE-L: 26.97  METEOR: 7.00
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p20 > run/summarization/seq2seq/config/portion/seq2seq.p20.log 2>&1 &
#BLEU: 15.50     ROUGE-L: 26.57  METEOR: 4.68
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p10 > run/summarization/seq2seq/config/portion/seq2seq.p10.log 2>&1 &
#BLEU: 15.52     ROUGE-L: 25.89  METEOR: 4.79
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p5 > run/summarization/seq2seq/config/portion/seq2seq.p5.log 2>&1 &
#BLEU: 15.22     ROUGE-L: 24.31  METEOR: 4.34
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.seq2seq.train -f config/portion/python_wan.p1 > run/summarization/seq2seq/config/portion/seq2seq.p1.log 2>&1 &
#BLEU: 3.61      ROUGE-L: 0.02   METEOR: 1.70
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.seq2seq.train -f config/portion/python_wan.p0 # zero-shot learning, save init model as checkpoint_last.pt
```