export HTTPS_PROXY="127.0.0.1:17890"
export HTTP_PROXY="127.0.0.1:17890"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

source activate UICoder

DATASET_NAME="vu_ws"

python ../scripts/train/train.py \
    --stage=0 \
    --model_name_or_path="/data02/users/lz/code/UICoder/checkpoints/stage0/l2048_p1024_vu_3m*3/checkpoint-90000" \
    --batch_size=1 \
    --output_path="/data02/users/lz/code/UICoder/checkpoints/" \
    --eval_size=4096 \
    --num_train_epochs=3 \
    --learning_rate=2e-5 \
    --name="$DATASET_NAME" \
    --path="/data02/users/lz/code/UICoder/datasets/WebSight-format-parquet/arrow" \
    --max_patches=1024 \
    --max_length=1024 \
    --max_num=3000000 \
    --preprocess=True \
    --make_patches_while_training=True \
    --save_steps=1000
