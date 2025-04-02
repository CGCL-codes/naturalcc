export CUDA_VISIBLE_DEVICES=3

MODEL=shadow_9

LANG=python                       # set python for py150
DATADIR=/pathto/data/membership_inference/${MODEL}/code-gpt
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=/pathto/data/membership_inference/${MODEL}/code-gpt/ranks
PRETRAINDIR=/pathto/data/membership_inference/${MODEL}/code-gpt/model/checkpoint-best    # directory of your saved model
LOGFILE=${MODEL}_model_rank.log

nohup python -u rank.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_batch_size=16 \
        --logging_steps=100 \
        --seed=42 > load9.log 2>&1 &