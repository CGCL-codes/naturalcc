export CUDA_VISIBLE_DEVICES=2
MODEL=shadow_7

LANG=python                       # set python for py150
DATADIR=/pathto/data/membership_inference/${MODEL}/code-gpt
LITFILE=../dataset/py150/literals.json
OUTPUTDIR=/pathto/data/membership_inference/${MODEL}/code-gpt/model
PRETRAINDIR=microsoft/CodeGPT-small-py        # microsoft/CodeGPT-small-py for py150
LOGFILE=${MODEL}_model_train.log

nohup python run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --do_train \
        --learning_rate=8e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=8 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=5 \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=42 \
        --overwrite_output_dir \
        --not_pretrain > tmp7.log 2>&1 &