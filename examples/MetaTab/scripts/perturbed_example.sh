# This script runs dp on subsampled wtq datasets using gpt-3.5
# - tables are perturbed by both transposition and row shuffling
# - resorting stage in NORM is enabled

CUDA_VISIBLE_DEVICES=0 python run_cot.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider openai --dataset wtq --sub_sample True \
    --perturbation transpose_shuffle --norm True --disable_resort False --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 10 --temperature 0.8 \
    --log_dir output/wtq_dp_transpose_shuffle --cache_dir cache/gpt-3.5
