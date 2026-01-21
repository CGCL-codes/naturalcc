# This script runs pyagent on all wtq datasets using gpt-3.5
# - tables are not perturbed
# - resorting stage in NORM is disabled
# - each query will be executed 5 times to do self-consistency

#CUDA_VISIBLE_DEVICES=0 python run_qwencoder_agent.py \
#    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
#    --provider llama3 --dataset wtq --sub_sample False \
#    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
#    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
#    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5

#4090
#CUDA_VISIBLE_DEVICES=1 python run_tablegpt_agent.py \
#    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
#    --provider llama3 --dataset wtq --sub_sample False \
#    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
#    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
#    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5

CUDA_VISIBLE_DEVICES=1 python run_tablegpt_agent_c_cut.py \
    --model gpt-3.5-turbo-0613 --long_model gpt-3.5-turbo-16k-0613 \
    --provider llama3 --dataset tabfact --sub_sample False \
    --perturbation none --use_full_table True --norm False --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 5 --temperature 0.8 \
    --log_dir output/tabfact_agent --cache_dir cache/gpt-3.5

