#Command:
```
# generate ruby path dataset

# generate ruby path/docstring_tokens mmap dataset

run code2seq model
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.code2seq.train -f config/python_wan/python > run/summarization/code2seq/config/python_wan/python.log 2>&1 &

# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.summarization.code2seq.eval -f config/python_wan/python
```

 
