import os
import json
from agent.model import Model
from utils.data import print_partial_markdown
from utils.eval import parse_header_checking_result, parse_header_sorting_result


def load_dataset(dataset_name=None, dataset_file=None):
    """
    Load the dataset based on the dataset name, either from dataset name or dataset file.

    Args:
    - dataset_name (str): The name of the dataset.
    - dataset_file (str): The path to the dataset file.

    Returns:
    - dict: The dataset.
    """
    if dataset_name in ["wtq", "wikitablequestion"]:
        with open("assets/data/wtq.json", "r") as f:
            data = json.load(f)
    
    elif dataset_name in ["tabfact", "tabularfact"]:
        with open("assets/data/tabfact.json", "r") as f:
            data = json.load(f)

    else:
        # Load the dataset from the file
        if dataset_file is None:
            raise ValueError(f"Dataset {dataset_name} is not supported, please provide a dataset file.")
        
        with open(dataset_file, "r") as f:
            data = json.load(f)
    return data 

def get_cot_prompt(dataset_name):
    """
    Load the COT prompt based on the dataset name.
    
    Args:
    - dataset_name (str): The name of the dataset.
    
    Returns:
    - str: The COT prompt.
    """
    if dataset_name in ["wtq", "wikitablequestion"]:
        from prompt.wtq.cot import cot_prompt
        return cot_prompt

    elif dataset_name in ["tabfact", "tabularfact"]:
        from prompt.tabfact.cot import cot_prompt
        return cot_prompt
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    

def query(model, long_model, prompt, temperature, self_consistency):
    """
    Execute a query on the model and handle prompt length for choosing the appropriate model.

    Args:
    - model: The primary model for querying.
    - long_model: The long version of the model for longer prompts.
    - prompt (str): The prompt to query.
    - temperature (float): The temperature setting for the query.
    - self_consistency (int): The number of outputs to generate.

    Returns:
    - Tuple: (text, response)
    """
    
    prompt_length = len(long_model.tokenizer.encode(prompt))

    if isinstance(model, Model):
        if prompt_length <= 3328:
            return model.query(prompt=prompt, temperature=temperature, max_tokens=4000 - prompt_length, n=self_consistency)
        elif prompt_length <= 14592:
            print(f"Prompt length -- {prompt_length} is too long, we use the 16k version.")
            return long_model.query(prompt=prompt, temperature=temperature, max_tokens=15360 - prompt_length, n=self_consistency)
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}
    else:
        # no short version of the model provided, which means we use the long version for all prompts
        if prompt_length <= 14592:
            return long_model.query(prompt=prompt, temperature=temperature, max_tokens=15360 - prompt_length, n=self_consistency)
        else:
            if self_consistency == 1:
                return f"Prompt length -- {prompt_length} is too long", {prompt_length: prompt_length}
            else:
                return ["Prompt length -- {prompt_length} is too long"] * self_consistency, {prompt_length: prompt_length}


def check_transpose(model: Model, long_model: Model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir):
    """
    Check if the table needs transposing, using cache if available.

    Args:
    - model, long_model (Model): The models used for querying.
    - table (str): The markdown representation of the table.
    - title (str): The title of the table.
    - table_id (str): The ID of the table.
    - perturbation (str): The perturbation applied to the table.
    - transpose_cache (dict): Cache for transpose information.
    - norm_cache (bool): Flag to determine if normalization caching is enabled.
    - cache_dir (str): Directory for caching.

    Returns:
    - bool: Whether the table needs transposing.
    """
    from prompt.general.transpose_check import header_check_prompt

    # Check cache first
    if table_id in transpose_cache and perturbation in transpose_cache[table_id]:
        return transpose_cache[table_id][perturbation]

    # Construct and send the query
    first_row = ", ".join([cell.strip() for cell in table.split("\n")[0].split("|")[1:-1]])
    first_column = ", ".join([row.split("|")[1].strip() for row in table.split("\n")]).strip()
    transpose_check_prompt = header_check_prompt.replace("[TABLE]", table)\
        .replace("[FIRST_ROW]", first_row)\
        .replace("[FIRST_COLUMN]", first_column)\
        .replace("[TITLE]", title)\
        .strip()
    print(transpose_check_prompt)
    exit(1)

    text, _ = query(model, long_model, transpose_check_prompt, temperature=0, self_consistency=1)

    transpose_flag = parse_header_checking_result(text)

    # Update cache if necessary
    if norm_cache:
        if table_id not in transpose_cache:
            transpose_cache[table_id] = {}
        transpose_cache[table_id] = {perturbation: transpose_flag}
        with open(os.path.join(cache_dir, "transpose.json"), "w") as f:
            json.dump(transpose_cache, f, indent=4)

    return transpose_flag

def check_sort(model: Model, long_model: Model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir):
    """
    Check if the table needs sorting, using cache if available.

    Args:
    - model, long_model: The models used for querying.
    - df (DataFrame): The DataFrame representation of the table.
    - title (str): The title of the table.
    - table_id (str): The ID of the table.
    - perturbation (str): The perturbation applied to the table.
    - resort_cache (dict): Cache for sorting information.
    - norm_cache (bool): Flag to determine if normalization caching is enabled.
    - cache_dir (str): Directory for caching.

    Returns:
    - List: The list of columns for sorting.
    """
    from prompt.general.resort_check import sort_prompt

    # Check cache first
    if table_id in resort_cache and perturbation in resort_cache[table_id]:
        return resort_cache[table_id][perturbation]

    # Construct and send the query
    partial_table = print_partial_markdown(df)
    heading_list = [cell.strip() for cell in partial_table.split("\n")[0].split("|")[1:-1]]
    headings = "; ".join(heading_list)

    resort_check_prompt = sort_prompt.replace("[TABLE]", partial_table)\
        .replace("[HEADINGS]", headings)\
        .replace("[TITLE]", title)\
        .strip()

    text, _ = query(model, long_model, resort_check_prompt, temperature=0, self_consistency=1)

    resort_list = parse_header_sorting_result(text)

    # Update cache if necessary
    if norm_cache:
        os.makedirs(cache_dir, exist_ok=True)
        resort_cache[table_id] = {perturbation: resort_list}
        with open(os.path.join(cache_dir, "resort.json"), "w") as f:
            json.dump(resort_cache, f, indent=4)

    return resort_list


def read_json_file(file_path):
    """
    Read a JSON file.

    Args:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The JSON file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except:
        return {}
    
    return data
