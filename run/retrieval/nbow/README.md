# Seq2Seq for code retrieval task

running with float32
```shell script
# dataset generation
python dataset/csn/flatten.py
python dataset/csn/retrieval/csn/preprocess.py
# train
python ./run/retrieval/nbow/train.py
# eval

```