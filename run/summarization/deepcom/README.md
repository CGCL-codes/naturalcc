# deepcom for code summarization task


CSN(feng)
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/ruby > run/summarization/deepcom/config/csn_feng/ruby.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/go > run/summarization/deepcom/config/csn_feng/go.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/php > run/summarization/deepcom/config/csn_feng/php.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/python > run/summarization/deepcom/config/csn_feng/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/java > run/summarization/deepcom/config/csn_feng/java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/csn_feng/javascript > run/summarization/deepcom/config/csn_feng/javascript.log 2>&1 &

# evaluation
python -m run.summarization.deepcom.eval -f config/csn_feng/ruby
python -m run.summarization.deepcom.eval -f config/csn_feng/go
python -m run.summarization.deepcom.eval -f config/csn_feng/php
python -m run.summarization.deepcom.eval -f config/csn_feng/python
python -m run.summarization.deepcom.eval -f config/csn_feng/java
python -m run.summarization.deepcom.eval -f config/csn_feng/javascript
```

Python(wan) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/python_wan/python > run/summarization/deepcom/config/python_wan/python.log 2>&1 &
# evaluation
python -m run.summarization.deepcom.eval -f config/python_wan/python
```

Java(hu) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/java_hu/java > run/summarization/deepcom/config/java_hu/java.log 2>&1 &
# evaluation
python -m run.summarization.deepcom.eval -f config/java_hu/java
```


StackOverflow training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/stack_overflow/csharp > run/summarization/deepcom/config/stack_overflow/csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/stack_overflow/sql > run/summarization/deepcom/config/stack_overflow/sql.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.deepcom.train -f config/stack_overflow/python > run/summarization/deepcom/config/stack_overflow/python.log 2>&1 &
# evaluation
python -m run.summarization.deepcom.eval -f config/stack_overflow/csharp
python -m run.summarization.deepcom.eval -f config/stack_overflow/sql
python -m run.summarization.deepcom.eval -f config/stack_overflow/python
```


