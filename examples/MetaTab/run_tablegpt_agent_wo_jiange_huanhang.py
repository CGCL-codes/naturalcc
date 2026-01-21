import os
import json
import re
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes, convert_cells_to_numbers
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, check_transpose, check_sort, read_json_file


from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.execute import markdown_to_df, parse_code_from_string, python_repl_ast, print_partial_markdown


def is_terminal(text: str) -> bool:
    return "Final Answer: " in text or "answer_directly" in text or "PROMPT TOO LONG, WE CAN NOT QUERY THE API" in text


def clean_markdown_table(table: str) -> str:
    """
    处理 Markdown 表格：
    1. 第一行去掉 | 但保留所有空格（包括首尾）
    2. 第二行（分隔线）直接删除
    3. 后续行去掉 | 但保留所有空格（包括首尾）
    """
    lines = table.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        if i == 0:  # 第一行：去掉 | 但保留所有空格
            cleaned_line = line.replace('|', ' ')
            cleaned_lines.append(cleaned_line)
        elif i == 1:  # 第二行：直接跳过（删除分隔线）
            continue
        else:  # 后续行：去掉 | 但保留所有空格
            cleaned_line = line.replace('|', ' ')
            if cleaned_line.strip():  # 避免空行（但保留空格）
                cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)

def clean_huanhang_markdown_table(table_str):
    # 先替换真正的换行符（但保留 "\n" 文本）
    lines = table_str.split('\n')  # 按行分割
    cleaned_lines = []
    for line in lines:
        # 替换掉可能导致换行的空格（可选）
        #line = line.strip()  # 移除首尾空格
        cleaned_lines.append(line)
    # 重新拼接成单行（用空格代替换行）
    return ' '.join(cleaned_lines)

    return '\n'.join(cleaned_lines)
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
        elif perturbation == "anonymized":
            if d["anonymized_tables"]:
                table = construct_markdown_table(**d["anonymized_tables"])
            else:
                continue

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
        table = df.to_markdown()



        for idx in index_list:

            if global_i < resume:
                global_i += 1
                pbar.update(1)
                continue

            elif global_i >= stop_at:

                break_flag = True
                break

           # if not transpose_flag:
                # reuse the temp
           #     print(f"Skip {global_i}", flush=True)
           #     with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
           #         json.dump(temp[global_i], f)
           #         f.write("\n")

           #     global_i += 1
           #     pbar.update(1)

           #     continue

            question = d["questions"][idx]

            if perturbation == "anonymized":
                answer = d["anonymized_answers"][idx]
            else:
                answer = d["answers"][idx]

            question_id = d["ids"][idx]


            log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
            # create the file
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            texts = []

            for _ in range(self_consistency):

                # create the table agent
                """
                agent = TableAgent(
                    table=df,
                    prompt_type=dataset,
                    model=model,
                    long_model=long_model,
                    temperature=temperature,
                    log_dir=log_path,
                    use_full_table=use_full_table,
                )

                text, response = agent.run(question=question, title=title)
                texts.append(text)
                """
                from prompt.wtq.agent import agent_prefix, agent_prefix_with_omitted_rows_guideline
                # Prepare the prompt (replacing TableAgent initialization)
                if use_full_table:
                    table = df.to_markdown()


                    table = clean_markdown_table(table)
                    table = clean_huanhang_markdown_table(table)


                    prompt = agent_prefix.replace("[TABLE]", table)
                else:
                    table = print_partial_markdown(df)
                    prompt = agent_prefix_with_omitted_rows_guideline.replace("[TABLE]", table)

                # Add title and question to prompt
                prompt = prompt.replace("[TITLE]", title).replace("[QUESTION]", question).strip()
                #print(prompt)
                #exit(1)
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write("=" * 50 + "\n")
                        f.write(prompt + "\n")


                response_text = ""
                response_list = []
                new_line = "\n"
                memory = {}

                for i in range(5):  # You'll need to define max_depth somewhere
                    # Replace agent.query() with direct model interaction

                    messages = [
                        #{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    generated_ids = model.generate(**model_inputs, max_new_tokens=5120)

                    generated_ids = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    #print(response)

                    text = response  # The generated response becomes our text

                    # Handle new lines
                    if i == 0:
                        if "\n\n" in text:
                            new_line = "\n\n"
                    else:
                        text = new_line + text

                    response_text += text
                    response_list.append(response)



                    # Check if terminal (you'll need to define is_terminal function)
                    if is_terminal(text):
                        break

                    # Parse code from response
                    if "Action Input:" in text:
                        code = parse_code_from_string(text.split("Action Input:")[-1].strip("\n").strip())
                    elif "Action:" in text:
                        code = parse_code_from_string(text.split("Action:")[-1].strip("\n").strip())
                    else:
                        code = parse_code_from_string(text)

                    # Execute code
                    observation, memory = python_repl_ast(
                        code,
                        custom_locals={"df": df},  # Using df directly now
                        custom_globals=globals(),
                        memory=memory
                    )

                    if isinstance(observation, str) and observation == "":
                        observation = "success!"

                    if "\n" in str(observation):
                        observation = "\n" + str(observation)

                    response_text += f"Observation: {observation}"
                    prompt += text + f"Observation: {observation}"


                # Log final response
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(response_text + "\n")

                texts.append(response_text)


            res = {
                "idx": global_i,
                "answer": answer,
                "text": texts if self_consistency > 1 else texts[0],
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


if __name__ == "__main__":
    Fire(main)