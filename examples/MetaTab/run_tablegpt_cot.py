import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model

from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe

from run_helper import load_dataset, get_cot_prompt, query, check_transpose, check_sort, read_json_file
import torch
import torch.distributed as dist
from llama import Llama
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("/home/ligen/lg")

def setup_distributed():
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # 默认端口
    os.environ["WORLD_SIZE"] = "1"  # 总进程数
    os.environ["RANK"] = "0"  # 当前进程的 rank
    os.environ["LOCAL_RANK"] = "0"  # 当前节点的 rank

    # 初始化分布式后端
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

def cleanup_distributed():
    # 清理分布式环境
    dist.destroy_process_group()


import re


def extract_last_sentence(text):
    sentences = re.split(r'[。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        return sentences[-1]
    return text
def main(
        model:Optional[str] = "gpt-3.5-turbo-0613", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo-16k-0613", # long model of the agent (only used for long prompt)
        #provider: str = "openai", # openai, huggingface, vllm
        provider: str = "llama3", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq or tabfact
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        sub_sample: bool = True, # whether to only run on the subset sampled data points
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 1e6, # stop at the i-th data point
        self_consistency:int = 10, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/wtq_cot", # directory to store the logs
        cache_dir: str = "cache", # directory to store the cache (normalization results)
        ckpt_dir: str = "./Meta-Llama-3-8B-Instruct",
        tokenizer_path: str = "./Meta-Llama-3-8B-Instruct/tokenizer.model",
        temperature_llama: float = 0.6,
        top_p: float = 0.9,
        #max_seq_len: int = 128,
        #max_gen_len: int = 64,
        max_seq_len: int = 5120,
        max_gen_len: int = 640,
        max_batch_size: int = 4
):

    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset and cot prompt ####
    data = load_dataset(dataset)
    cot_prompt = get_cot_prompt(dataset)

    #### load the model ####

    if model:
        model = Model(model, provider=provider)
    if long_model:
        long_model = Model(long_model, provider=provider)
    
    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))
    
    #### prepare the iterator ####
    global_i = 0
    break_flag = False
    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])

    pbar = tqdm(total=stop_at if stop_at < total else total)
    
    #### start the loop ####
    for table_idx, d in enumerate(data):
        if break_flag:
            break
        
        index_list = d['sampled_indices'] if sub_sample else range(len(d["questions"]))
        
        # if the table is empty, skip
        if len(index_list) == 0:
            continue
            
        # load table infos
        table_id = d["table_id"]
        title = d["title"]

        if perturbation == "none":
            table = construct_markdown_table(**d["table"])
        elif perturbation == "transpose":
            table = construct_markdown_table(**d["transposed_table"])
        elif perturbation == "shuffle":
            table = construct_markdown_table(**d["row_shuffled_table"])
        elif perturbation == "transpose_shuffle":
            table = construct_markdown_table(**d["row_shuffled_transposed_table"])
            
        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []


        # reset the table
        table = df.to_markdown()

        model_name = "tablegpt"
        #model_name = "QwenCoder"

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for idx in index_list:

            if global_i < resume:
                global_i += 1
                pbar.update(1)
                continue
            elif global_i >= stop_at:
                break_flag = True
                break

            question = d["questions"][idx]
            answer = d["answers"][idx]

            question_id = d["ids"][idx]

            prompt = cot_prompt.replace("[TABLE]", table)\
                .replace("[QUESTION]", question)\
                .replace("[TITLE]", title)\
                .strip()
            #print(len(prompt))
            if len(prompt)> 10000:
                continue

            #exit(1)
            #text, response = query(model, long_model, prompt, temperature, self_consistency)
            import time
            import signal


            # 定义一个超时异常
            class TimeoutError(Exception):
                pass

            # 超时处理函数
            def handler(signum, frame):
                raise TimeoutError("Model generation timed out")

            # 设置超时时间（例如10秒）
            timeout = 50

            # 注册信号处理函数
            signal.signal(signal.SIGALRM, handler)

            # 构造消息
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 准备模型输入
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # 启动超时计时器
            signal.alarm(timeout)

            try:
                # 尝试生成响应
                generated_ids = model.generate(**model_inputs, max_new_tokens=5120)
                signal.alarm(0)  # 取消超时
            except TimeoutError:
                # 如果超时，跳过生成
                print("Model generation timed out, skipping...")
                response = "Model generation timed out, no response."
                continue
            else:
                # 处理生成的ID
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


            #print(response)
            # 手动清理特殊标记
            text = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

            # 提取最后一句
            text = [extract_last_sentence(text)]
            #print(text)



            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            with open(log_path, "w") as f:
                f.write("===================Title===================\n")
                f.write(title + "\n")
                f.write("===================Table===================\n")
                f.write(table + "\n")
                f.write("===================Question===================\n")
                f.write(question + "\n")
                f.write("===================Text===================\n")
                f.write(text if isinstance(text, str) else "\n".join(text))
                f.write("\n")
                f.write("===================Answer===================\n")
                f.write(",".join(answer) if isinstance(answer, list) else str(answer))
                f.write("\n")

            res = {
                "idx": global_i,
                "answer": answer,
                "text": text,
                "transpose": transpose_flag,
                "resort": resort_list,
                "question_id": question_id,
                "table_id": table_id,
                "title": title,
                "table": table,
                "question": question,
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
                json.dump(res, f)
                f.write("\n")

            global_i += 1
            pbar.update(1)

            #cleanup_distributed()


if __name__ == "__main__":
    """
    import os
    import torch.distributed as dist

    # 设置所需的环境变量
    os.environ['RANK'] = '0'
    # 设置为单进程
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # 初始化分布式进程组
    dist.init_process_group(backend='gloo', init_method="env://?use_libuv=False")
    print("Distributed process group initialized successfully.")
    """

    Fire(main)