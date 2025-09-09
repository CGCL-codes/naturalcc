model=$1
batch_size=$2

accelerate launch  main.py \
    --model $model \
    --tasks humaneval \
    --batch_size $batch_size \
    --max_length_generation 512 \
    --precision fp16 \
    --allow_code_execution \
    --metric_output_path $model/humaneval_evaluation_results.json \
    --save_generations --save_generations_path $model/humaneval_generations.json \
    --max_memory_per_gpu auto \
    --do_sample True \
    --temperature 0.2 \
    --top_p 0.95 \
    --n_samples 50 \
    --seed 42
