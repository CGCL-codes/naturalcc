# `run_modern_mapping.sh` 脚本说明

该脚本用于 **现代化训练任务**，支持自定义 tokenizer 映射 (mapping_file)、分布式训练、多卡训练以及混合精度加速。

---

## 1. 脚本功能

- **设置环境变量**：禁用 WandB、控制 tqdm 显示  
- **配置 GPU 设备** (CUDA_VISIBLE_DEVICES)  
- **定义训练参数**：batch size、学习率、训练轮数等  
- **自动创建输出目录** (OUTPUT_DIR)  
- **使用 torchrun 启动分布式训练**  
- **训练日志同时输出到屏幕和文件** (stdout / stderr)  
- **自定义模型保存逻辑**，由 CustomSaveCallback 回调完成  

---

## 2. 环境变量与 GPU 配置

```bash
export WANDB_DISABLED=true        # 禁用 WandB
export TRANSFORMERS_NO_TQDM=0     # tqdm 显示控制（1 为禁用）
export CUDA_VISIBLE_DEVICES="1"   # 指定使用 GPU
modify_type=overuse-allaffix       # 映射类型，可替换
NUM_GPUS=1                         # 使用 GPU 数量
```

### 自动计算 master_port

```bash
BASE_PORT=29500
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
MASTER_PORT=$((BASE_PORT + FIRST_GPU))
```

- **MASTER_PORT**：用于分布式训练通信  
- **FIRST_GPU**：取 CUDA_VISIBLE_DEVICES 中的第一个 GPU ID  

---

## 3. 训练参数配置

```bash
MODEL_PATH="/home/wanyao/wangchen/models/Qwen/Qwen2.5-Coder-1.5B"
DATA_DIR="../../dataset/concode"
ATTR="qwen2.5_batchsize24_accum2_epoch40_lr5e-5"
mapping_file=./mapping/${modify_type}.json
OUTPUT_DIR="./saved_models_mapping_RQ3/${modify_type}_${ATTR}/"
```

- **MODEL_PATH**：基础模型路径  
- **DATA_DIR**：训练/验证/测试数据路径  
- **mapping_file**：tokenizer 映射文件  
- **OUTPUT_DIR**：保存训练结果的输出目录  

---

## 4. 启动训练命令

```bash
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
```

### 说明

- **torchrun**：启动分布式训练  
- **--fp16**：启用混合精度  
- **--save_strategy "no"**：禁用 Trainer 自动保存，由 CustomSaveCallback 自行保存模型  
- **日志输出**：同时显示在屏幕和保存到文件 (tee)  

---

## 5. 日志与输出

- **标准输出**：$OUTPUT_DIR/train_stdout.log  
- **错误输出**：$OUTPUT_DIR/train_stderr.log  
- **模型保存**：由 CustomSaveCallback 回调函数控制，保存路径在 $OUTPUT_DIR  

---

## 6. 可修改参数

### GPU 选择

```bash
export CUDA_VISIBLE_DEVICES="0"
modify_type=exchange
```

### 映射类型

```bash
modify_type=overuse-allaffix
modify_type=overuse-noaffix
modify_type=nochange
```

### 训练参数

```
--per_device_train_batch_size
--learning_rate
--num_train_epochs
--gradient_accumulation_steps
```

### 日志与保存策略

```
--logging_strategy
--eval_strategy
--save_strategy
```
