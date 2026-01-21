# This script runs dp on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_qwencoder_cot.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset wtq --sub_sample False \
    --perturbation none --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 10 --temperature 0.8 \
    --log_dir output/wtq_dp --cache_dir cache/gpt-3.5
