# TravTrans for code completion task

```shell script
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.completion.trav_trans.train -f config/py150/python > run/completion/trav_trans/config/py150/python.log 2>&1 &
# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.completion.trav_trans.eval -f config/py150/python
#[all] tokens, accuracy: 0.76038, MRR: 0.831657
#[attr] tokens, accuracy: 0.648244, MRR: 0.720834
#[num] tokens, accuracy: 0.568942, MRR: 0.685514
#[name] tokens, accuracy: 0.697836, MRR: 0.763258
#[param] tokens, accuracy: 0.671314, MRR: 0.710821
```
