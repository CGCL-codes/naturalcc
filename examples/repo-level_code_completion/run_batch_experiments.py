#!/usr/bin/env python3
import subprocess
import itertools
import argparse

# 定义模型路径字典
models = {
    "starcoder2-3b": "/mnt/silver/tanlei/hf-models/starcoder2-3b",
    "starcoder2-7b": "/mnt/silver/tanlei/hf-models/starcoder2-7b",
    "CodeLlama-7b-Instruct": "/mnt/silver/tanlei/hf-models/CodeLlama-7b-Instruct-hf",
    "Qwen2.5-Coder-0.5B-Instruct": "/mnt/silver/tanlei/hf-models/Qwen2.5-Coder-0.5B-Instruct",
    "Qwen2.5-Coder-3B-Instruct": "/mnt/silver/tanlei/hf-models/Qwen2.5-Coder-3B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "/mnt/gold/wangchenlong/HF_models/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242",
}

# 固定参数（不可修改）
FIXED_LANGUAGE = "python"
FIXED_PROMPT_FILE = "/home/wanyao/talentan/naturalcc/examples/repo-level_code_completion/merged_line_completion.jsonl"
FIXED_DTYPE = "bf16"
FIXED_OUTPUT_DIR = "/home/wanyao/talentan/naturalcc/examples/repo-level_code_completion"

# 定义所有支持的 task 名称
ALL_TASKS = [
    "repofuse-main",
    # "repofuse-similar_type",
    # "repofuse-crossfile_type",
    # "repofuse-seq_length",
    # "repofuse-ratio",
]

def run_experiment(task, model_name, max_seq_length, ranking_strategy, gen_length, crossfile_type, reranker, batch_size, ratio):
    """
    根据传入的参数构造命令，并执行实验
    """
    inf_seq_length = int(max_seq_length * ratio)
    cfc_seq_length = max_seq_length - inf_seq_length

    cmd = (
        f"accelerate launch --config_file /home/wanyao/talentan/naturalcc/examples/repo-level_code_completion/config.yaml /home/wanyao/talentan/RepoFuse/eval/eval.py "
        f"--language {FIXED_LANGUAGE} "
        f"--task {task} "
        f"--model_name_or_path {models[model_name]} "
        f"--model_type codelm_cfc "
        f"--prompt_file {FIXED_PROMPT_FILE} "
        f"--max_seq_length {max_seq_length} "
        f"--gen_length {gen_length} "
        f"--inf_seq_length {inf_seq_length} "
        f"--cfc_seq_length {cfc_seq_length} "
        f"--crossfile_type {crossfile_type} "
        f"--ranking_strategy {ranking_strategy} "
        f"--reranker {reranker} "
        f"--batch_size {batch_size} "
        f"--dtype {FIXED_DTYPE} "
        f"--output_dir {FIXED_OUTPUT_DIR}"
    )
    
    print("运行命令：", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("命令执行失败:", cmd)
    else:
        print("命令执行成功")

def main(task):
    if task == "repofuse-main":
        # model_name_or_path_list = ["starcoder2-3b", "starcoder2-7b", "CodeLlama-7b-Instruct", "Qwen2.5-Coder-0.5B-Instruct", "Qwen2.5-Coder-3B-Instruct", "Qwen2.5-Coder-7B-Instruct"]
        model_name_or_path_list = ["Qwen2.5-Coder-7B-Instruct"]
        max_seq_length_list = [4096]
        ranking_strategy_list = ['BM25']
        gen_list = [50]
        crossfile_list = ["S_R"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 1
        ratio_list = [0.5]
    elif task == "repofuse-similar_type":
        model_name_or_path_list = ["CodeLlama-7b-Instruct"]
        max_seq_length_list = [4096]
        ranking_strategy_list = ["Graph"]
        gen_list = [50]
        crossfile_list = ["Similar"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 1
        ratio_list = [0.5]
    elif task == "repofuse-crossfile_type":
        model_name_or_path_list = ["Qwen2.5-Coder-7B-Instruct"]
        max_seq_length_list = [4096]
        ranking_strategy_list = ['BM25']
        gen_list = [50]
        crossfile_list = ["S_R"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 1
        ratio_list = [0.4]
    elif task == "repofuse-seq_length":
        model_name_or_path_list = ["starcoder2-7b", "CodeLlama-7b-Instruct", "Qwen2.5-Coder-7B-Instruct"]
        max_seq_length_list = [16384]
        ranking_strategy_list = ['Graph']
        gen_list = [50]
        crossfile_list = ["S_R"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 1
        ratio_list = [0.5]
    elif task == "repofuse-ratio":
        model_name_or_path_list = ["starcoder2-7b", "CodeLlama-7b-Instruct", "Qwen2.5-Coder-7B-Instruct"]
        max_seq_length_list = [4096]
        ranking_strategy_list = ['Graph']
        gen_list = [50]
        crossfile_list = ["S_R"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 1
        ratio_list = [0.1, 0.3, 0.7, 0.9]
    elif task == "test":
        model_name_or_path_list = ["Qwen2.5-Coder-0.5B-Instruct"]
        max_seq_length_list = [4096]
        ranking_strategy_list = ["Jaccard"]
        gen_list = [50]
        crossfile_list = ["S_R"]
        RERANKER = "UnixCoder"
        BATCH_SIZE = 10
        ratio_list = [0.5]


    # 遍历所有组合进行实验
    for model_name, max_seq_length, ranking_strategy, gen_length, crossfile_type, ratio in itertools.product(
            model_name_or_path_list, max_seq_length_list, ranking_strategy_list, gen_list, crossfile_list, ratio_list):
        run_experiment(task, model_name, max_seq_length, ranking_strategy, gen_length, crossfile_type, RERANKER, BATCH_SIZE, ratio)

if __name__ == "__main__":
    # import time
    # delay = 10 * 60 * 60  # 16小时 = 16 * 60分钟 * 60秒
    # print(f"程序将在 {delay} 秒（即{delay/3600}个小时）后开始执行任务...")
    # time.sleep(delay)  # 程序等待16个小时

    parser = argparse.ArgumentParser(description='运行实验脚本')
    parser.add_argument('--task', type=str, help='指定任务类型，如repofuse_main, repofuse-similar_type等')
    args = parser.parse_args()
    if args.task == "all":
        for t in ALL_TASKS:
            print(f"开始执行 task: {t}")
            main(t)
    else:
        main(args.task)
