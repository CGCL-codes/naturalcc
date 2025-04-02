export HTTPS_PROXY="127.0.0.1:17890"
export HTTP_PROXY="127.0.0.1:17890"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

source activate UICoder

DATASET_NAME="ws"

python scripts/train/train.py \
    --stage=2 \
    --model_name_or_path="/data02/models/pix2struct-large" \
    --batch_size=12 \
    --output_path="/data02/users/lz/code/UICoder/checkpoints/" \
    --eval_size=4096 \
    --num_train_epochs=1 \
    --learning_rate=5e-5 \
    --name="$DATASET_NAME" \
    --path="/data02/users/lz/code/UICoder/datasets/WebSight-format-parquet/arrow" \
    --max_patches=512 \
    --max_length=256 \
    --max_num=300000 \
    --preprocess=True \
    --make_patches_while_training=True