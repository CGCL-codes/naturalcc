# SeqRNN for code completion task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.seqrnn.train -f config/raw_py150/python > run/completion/seqrnn/config/raw_py150/seqrnn.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.eval -f config/raw_py150/python
#[all] tokens, accuracy: 0.654876, MRR: 0.737272
#[attr] tokens, accuracy: 0.429728, MRR: 0.516714
#[num] tokens, accuracy: 0.35076, MRR: 0.474535
#[name] tokens, accuracy: 0.365651, MRR: 0.465247
#[param] tokens, accuracy: 0.62045, MRR: 0.660562
```


## SeqRNN 
1 LSTM layer, sharing embedding weight with out linear layer
```shell
# seq1lrnn
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.seqrnn.train -f config/csn_feng/seq1lrnn/ruby > run/completion/seqrnn/config/csn_feng/seq1lrnn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.eval -f config/csn_feng/seq1lrnn/ruby
# accuracy: 0.40518, MRR: 0.522495
```

## SeqRNN 
3 BiLSTM layer, sharing embedding weight with out linear layer
```shell
# seq3lrnn
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.seqrnn.train -f config/csn_feng/seq3lrnn/ruby > run/completion/seqrnn/config/csn_feng/seq3lrnn/ruby.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.seqrnn.eval -f config/csn_feng/seq3lrnn/ruby
# accuracy: 0.422654, MRR: 0.534116
```