# This script runs dp on subsampled wtq datasets using vicuna-13b-v1.5-16k
# - tables are not perturbed
# - resorting stage in NORM is disabled

CUDA_VISIBLE_DEVICES=0 python run_cot.py \
    --model None --long_model lmsys/vicuna-13b-v1.5-16k \
    --provider vllm --dataset wtq  --sub_sample True \
    --perturbation none --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 10 --temperature 0.8 \
    --log_dir output/wtq_dp_vicuna --cache_dir cache/vicuna