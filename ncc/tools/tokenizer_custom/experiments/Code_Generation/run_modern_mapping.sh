#!/bin/bash
export WANDB_DISABLED=true
# 设置环境变量
export TRANSFORMERS_NO_TQDM=0   # ✅ 为1禁用 tqdm 控制符
# export TQDM_ASCII=True
# 定义变量，方便管理

# 使用时，取消您想运行的实验块的注释即可
# bash run_modern_mapping.sh
# export CUDA_VISIBLE_DEVICES="0"
# modify_type=exchange

export CUDA_VISIBLE_DEVICES="1"
modify_type=overuse-allaffix

# export CUDA_VISIBLE_DEVICES="2"
# modify_type=overuse-noaffix

# export CUDA_VISIBLE_DEVICES="3"
# modify_type=nochange

NUM_GPUS=1
MODEL_PATH="/home/wanyao/wangchen/models/Qwen/Qwen2.5-Coder-1.5B"
DATA_DIR="../../dataset/concode"
ATTR="qwen2.5_batchsize24_accum2_epoch40_lr5e-5" # 简化了文件名
# ATTR="qwen2.5_b16_e30_lr5e-5_custom_save" # 建议修改名称以作区分
mapping_file=./mapping/${modify_type}.json
OUTPUT_DIR="./saved_models_mapping_RQ3/${modify_type}_${ATTR}/"

# 基础 master_port
BASE_PORT=29500
# 取第一个 GPU id
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
# master_port = BASE_PORT + FIRST_GPU
MASTER_PORT=$((BASE_PORT + FIRST_GPU))
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] FIRST_GPU=$FIRST_GPU"
echo "[INFO] MASTER_PORT=$MASTER_PORT"


# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 使用 torchrun 启动分布式训练
# 注意：确保每一行的 `\` 后面都没有任何空格
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT run_modern_mapping.py \
    --mapping_file "$mapping_file" \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "${DATA_DIR}/train.json" \
    --validation_file "${DATA_DIR}/dev.json" \
    --test_file "${DATA_DIR}/test.json" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    --weight_decay 0.01 \
    --seed 2 \
    --fp16 \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --save_strategy "no" \
    --max_source_length 256 \
    --max_target_length 256 \
    --generation_num_beams 5 \
    > >(tee "$OUTPUT_DIR/train_stdout.log") \
    2> >(tee "$OUTPUT_DIR/train_stderr.log" >&2)
    # > >(tee "$OUTPUT_DIR/train.log") 
    # > "$OUTPUT_DIR/train.log"
    # > >(tee "$OUTPUT_DIR/train.log") 
    # >：重定向 stdout。
    # >(tee "$OUTPUT_DIR/train.log")：把 stdout 写入 log 文件，同时 stdout 也显示在屏幕上。
    # 2>&1 | tee "$OUTPUT_DIR/train.log"
    # 把 stdout + stderr 合并，通过 pipe 写入 tee
    # 2> >(tee "$OUTPUT_DIR/train.log")
    

    # --logging_steps 100 \
    # --save_strategy steps \
    # --save_steps 500 \
    # --eval_strategy steps \
    # --eval_steps 500 \
    # --save_strategy epoch \
    # --save_total_limit 3 \
    # --load_best_model_at_end True \
    # --metric_for_best_model "eval_loss" \
    # --save_safetensors False \
# 因为您通过 --save_strategy "no" 禁用了 Trainer 的自动保存功能，所以 Trainer 根本就不会有机会去读取和使用 --save_safetensors 这个参数。这个参数所依附的功能本身已经被您关闭了。
# 现在是谁在保存模型？
# 在您当前的脚本中，保存模型的工作完全由您添加的 CustomSaveCallback 回调函数来完成。这个回调函数在每次评估后，通过直接调用 model.save_pretrained(...) 或 model.model.save_pretrained(...) 来手动执行保存。
