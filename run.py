import os

commands = \
    """
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.tree2seq.train -f config/csn_feng/java.fp16 > run/summarization/tree2seq/config/csn_feng/java.fp16.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.tree2seq.train -f config/csn_feng/go.fp16 > run/summarization/tree2seq/config/csn_feng/go.fp16.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.tree2seq.train -f config/csn_feng/php.fp16 > run/summarization/tree2seq/config/csn_feng/php.fp16.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.tree2seq.train -f config/csn_feng/python.fp16 > run/summarization/tree2seq/config/csn_feng/python.fp16.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.summarization.tree2seq.train -f config/csn_feng/javascript.fp16 > run/summarization/tree2seq/config/csn_feng/javascript.fp16.log 2>&1
    """.strip()

for cmd in commands.split('\n'):
    cmd = cmd.strip()
    print(cmd)
    os.system(cmd)
