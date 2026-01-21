import os
import json
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes, convert_cells_to_numbers
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, check_transpose, check_sort, read_json_file
from prompt.wtq.agent import agent_prefix, agent_prefix_with_omitted_rows_guideline

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.execute import markdown_to_df, parse_code_from_string, python_repl_ast, print_partial_markdown


def is_terminal(text: str) -> bool:
    return "Final Answer: " in text or "answer_directly" in text or "PROMPT TOO LONG, WE CAN NOT QUERY THE API" in text
def main(
        model:Optional[str] = "gpt-3.5-turbo-0613", # base model of the agent (for short prompt to save money)
        long_model:Optional[str] = "gpt-3.5-turbo-16k-0613", # long model of the agent (only used for long prompt)
        provider: str = "vllm", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq, tabfact
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        use_full_table: bool = True, # whether to use the full table or only the partial table
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        sub_sample: bool = True, # whether to only run on the subset sampled data points
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 1e6, # stop at the i-th data point
        self_consistency:int = 1, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/tabfact_agent", # directory to store the logs
        cache_dir: str = "cache", # directory to store the cache (normalization results)
):
    
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset ####
    data = load_dataset(dataset)

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

    # read the results from output/wtq_cot_wo_norm
    #with open("output/wtq_agent_wo_norm/result.jsonl", "r") as f:
    #    temp = [json.loads(line) for line in f.readlines()]

    model_name = "tablegpt"
    # model_name = "QwenCoder"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        elif perturbation == "column_shuffle":
            table = construct_markdown_table(**d["column_shuffled_table"])
        elif perturbation == "all_shuffle":
           table = construct_markdown_table(**d["row_column_shuffled_table"])
        elif perturbation == "transpose_shuffle":
            table = construct_markdown_table(**d["row_shuffled_transposed_table"])

        df = markdown_to_df(table)

        # transpose and sort if necessary
        transpose_flag = False
        resort_list = []
        
        if norm:
            transpose_flag = check_transpose(model, long_model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir)
            
            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)
            
            if not disable_resort:
                resort_list = check_sort(model, long_model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir)
                df = sort_dataframe(df, resort_list)

        df = convert_cells_to_numbers(df)

        # reset the table

        def split_dataframe_safely(df):
            """
            Safely split dataframe vertically into two parts considering:
            - Column count and structure
            - Always keep first column in both parts
            - Handle edge cases
            Returns two dataframe copies
            """

            if len(df.columns) == 0:
                return df.copy(), df.copy()

            # 如果只有1列，两个部分都返回完整的DataFrame
            if len(df.columns) == 1:
                return df.copy(), df.copy()

            # 如果只有2列，两个部分都包含第一列和第二列
            if len(df.columns) == 2:
                return df.copy(), df.iloc[:, [0, 1]].copy()

            # 计算分割点（从第二列开始分割，因为第一列两个部分都要保留）
            split_point = len(df.columns) // 2

            # 第一部分：第一列 + 前半部分的其他列
            part1_columns = [0] + list(range(1, split_point))
            part1 = df.iloc[:, part1_columns].copy()

            # 第二部分：第一列 + 后半部分的其他列
            part2_columns = [0] + list(range(split_point, len(df.columns)))
            part2 = df.iloc[:, part2_columns].copy()

            return part1, part2

        # 使用示例：
        # part1, part2 = split_dataframe_safely(df)

        def process_table_qa(df, question, title, model, tokenizer, log_path=None, use_full_table=True):
            """Process QA for a single table part"""

            if use_full_table:
                table = df.to_markdown()
                prompt = agent_prefix.replace("[TABLE]", table)
            else:
                table = print_partial_markdown(df)
                prompt = agent_prefix_with_omitted_rows_guideline.replace("[TABLE]", table)

            prompt = prompt.replace("[TITLE]", title).replace("[QUESTION]", question).strip()

            if log_path:
                with open(log_path, "a") as f:
                    f.write("=" * 50 + "\n")
                    f.write(prompt + "\n")

            response_text = ""
            memory = {}

            for i in range(5):  # max_depth
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**model_inputs, max_new_tokens=5120)
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                response_text += response
                if is_terminal(response):
                    break

                code = parse_code_from_response(response)
                observation, memory = python_repl_ast(
                    code,
                    custom_locals={"df": df},
                    custom_globals=globals(),
                    memory=memory
                )

                response_text += f"\nObservation: {observation}"
                prompt += response + f"\nObservation: {observation}"

            if log_path:
                with open(log_path, "a") as f:
                    f.write(response_text + "\n")

            return response_text


        # Main processing loop
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

            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # Split table safely


            try:

                df_part1, df_part2 = split_dataframe_safely(df)

                if len(df_part1) == 0 or len(df_part2) == 0:
                    raise ValueError("Empty table part after split")
            except Exception as e:
                df_part1 = df.copy()
                df_part2 = df.copy()

            # Process both parts
            texts_part1, texts_part2 = [], []

            for _ in range(self_consistency):
                # Process first part
                response_part1 = process_table_qa(
                    df_part1, question, title, model, tokenizer,
                    log_path, use_full_table
                )
                texts_part1.append(response_part1)

                # Process second part
                response_part2 = process_table_qa(
                    df_part2, question, title, model, tokenizer,
                    log_path, use_full_table
                )
                texts_part2.append(response_part2)

            # Save results
            res = {
                "idx": global_i,
                "answer": answer,
                "text_part1": texts_part1 if self_consistency > 1 else texts_part1[0],
                "text_part2": texts_part2 if self_consistency > 1 else texts_part2[0],
                "table_shape": {"original": df.shape, "part1": df_part1.shape, "part2": df_part2.shape},
                "question_id": question_id,
                "table_id": table_id,
                "title": title,
                "question": question,
                "table_part1": df_part1.to_dict(),
                "table_part2": df_part2.to_dict()
            }

            with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
                json.dump(res, f)
                f.write("\n")

            global_i += 1
            pbar.update(1)


if __name__ == "__main__":
    Fire(main)