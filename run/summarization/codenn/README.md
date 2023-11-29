# codenn for code summarization task


CSN(feng)
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/ruby > run/summarization/codenn/config/csn_feng/ruby.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/go > run/summarization/codenn/config/csn_feng/go.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/php > run/summarization/codenn/config/csn_feng/php.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/python > run/summarization/codenn/config/csn_feng/python.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/java > run/summarization/codenn/config/csn_feng/java.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/csn_feng/javascript > run/summarization/codenn/config/csn_feng/javascript.log 2>&1 &

# evaluation
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/ruby
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/go
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/php
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/python
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/java
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/csn_feng/javascript
```

Python(wan) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/python_wan/python > run/summarization/codenn/config/python_wan/python.log 2>&1 &
# evaluation
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/python_wan/python
```

Java(hu) training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=1 nohup python -m run.summarization.codenn.train -f config/java_hu/java > run/summarization/codenn/config/java_hu/java.log 2>&1 &
# evaluation
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/java_hu/java
```


StackOverflow training commands
```shell script
# train
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/stack_overflow/csharp > run/summarization/codenn/config/stack_overflow/csharp.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/stack_overflow/sql > run/summarization/codenn/config/stack_overflow/sql.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m run.summarization.codenn.train -f config/stack_overflow/python > run/summarization/codenn/config/stack_overflow/python.log 2>&1 &
# evaluation
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/stack_overflow/csharp
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/stack_overflow/sql
CUDA_VISIBLE_DEVICES=0 python -m run.summarization.codenn.eval -f config/stack_overflow/python
```


