import torch
from Finetune_Hallu_Value import FineTuneModel
from tqdm import tqdm

import json
import itertools


QWEN7B = "HALLU_VALUE_BASE_MODEL_PATH"
MODEL_NAME = "BASE_MODEL_PATH" 

INPUT_FILE = "CODE"
OUTPUT_FILE = "SUMMARY"

BEGIN_LINE = 0 
END_LINE = None  
data = []

device = torch.device("cuda:0") 
device_map = {"": "cuda:0"}    

with open(INPUT_FILE,"r") as input_file:
    for cnt,line in enumerate(itertools.islice(input_file,BEGIN_LINE,END_LINE)):
        p = json.loads(line)
        data.append({"index":cnt + BEGIN_LINE, "code":p["code"]})


from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, AutoModel,StoppingCriteriaList

class ForcePrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_ids, original_input_length):
        self.prefix_ids = prefix_ids
        self.original_input_length = original_input_length
        self.prefix_len = len(prefix_ids)
    
    def __call__(self, input_ids, scores):
        current_length = input_ids.shape[1] - self.original_input_length
        if current_length < self.prefix_len:
            forced_token_id = self.prefix_ids[current_length]
            scores[:, :] = -float('inf')
            scores[:, forced_token_id] = 0
        return scores
    
from transformers import StoppingCriteria


class StopAfterPrefixOnDot(StoppingCriteria):
    def __init__(self, start_len, tokenizer_ref):
        self.start_len = start_len
        self.tokenizer_ref = tokenizer_ref
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.start_len:
            return False

        newly_generated_count = input_ids.shape[1] - self.start_len
        if newly_generated_count < 5:
            return False

        last_token_id1 = input_ids[0, -2].item()
        last_token_id2 = input_ids[0, -1].item()
        last_token = self.tokenizer_ref.decode([last_token_id1, last_token_id2])
        

        is_dot = (". " in last_token) or (".\n" in last_token) or (":\n" in last_token)
        
        if is_dot:
            return True
            
        return False

model7B = AutoModelForCausalLM.from_pretrained(
    QWEN7B,
    torch_dtype="auto",
    device_map=device_map
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map=device_map
)

tokenizer7B = AutoTokenizer.from_pretrained(QWEN7B, device_map = device_map)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map = device_map)

tokenizer7B.padding_side = "left"


if tokenizer7B.pad_token is None:
    tokenizer7B.pad_token = tokenizer7B.eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id
if model7B.config.pad_token_id is None:
    model7B.config.pad_token_id = tokenizer7B.eos_token_id
    
value_model = FineTuneModel(model7B).to(device)

value_model.load_state_dict(torch.load("HALLU-VALUE-WEIGHT",weights_only=True, map_location = device))
value_model.eval()

dot_token_id = tokenizer.convert_tokens_to_ids('.')


prefix = "As a Java Expert, provide a detailed summary of the following Java code. Please structure your summary into the following sections:\n1. Inputs and outputs of the method\n2. Business purpose\n3. Detailed functional summary of the method (You have to explain the code line by line).\n\n"

step_size = 1
max_length = 1024
MAX_SENTENCE = 100

import re
import copy

def gather_stop_ids(tok):
    candidates = ["<|eot_id|>", "<|im_end|>", "</s>"] 
    ids = []
    for s in candidates:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid != tok.unk_token_id and tid != -1:
                ids.append(tid)
        except Exception:
            pass
    base = getattr(tok, "eos_token_id", None)
    if isinstance(base, int):
        ids.append(base)
    return list(dict.fromkeys(ids))

STOP_IDS = gather_stop_ids(tokenizer)

def trim_right_regex(s):
    last_index = -1
    for j in reversed(range(len(s))):
        if s[j] == '.':
            last_index = j
            break

    if last_index != len(s) - 1 and (s[last_index + 1] == ' ' or s[last_index + 1] == '\n' or s[last_index + 1] == '\t'):
        last_index += 1
    return s[:last_index + 1]

for i in tqdm(range(len(data))):
    print(f"now we are at index {i}", flush=True)
    prompt = prefix + data[i]["code"]
    code_enc = tokenizer7B(
            data[i]["code"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    code_input_ids = code_enc["input_ids"].to(device)
    code_attention_mask = code_enc["attention_mask"].to(device)
    messages = [
        {"role": "system", "content": "Assume you are an expert in understanding JAVA code."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    final_response = ""

    current_input_ids = model_inputs.input_ids
    past_key_values = None
    cnt = 0
    original_input_length = model_inputs.input_ids.shape[1]

    for cnt in range(MAX_SENTENCE):
        prefix_ids = tokenizer.encode(final_response, add_special_tokens=False)

        start_index_for_slicing = current_input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([
            StopAfterPrefixOnDot(start_len=start_index_for_slicing, tokenizer_ref=tokenizer)
        ])

        generated_outputs = []
        max_new_tokens = max_length + len(prefix_ids) 
        generated_ids_tmp = [[None] * 6 for _ in range(step_size)]
        response = [[""] * 6 for _ in range(step_size)]
        cand_value = [[0] * 6 for _ in range(step_size)]
        
        max_new_tokens = 256

        for k in range(step_size):
            with torch.no_grad():
                generated_ids_tmp[k][0] = model.generate( 
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=None,
                    top_p = None,
                    top_k = None
                )

                generated_ids_tmp[k][1] = model.generate( 
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=0.1,
                    top_p = 0.9,
                )

                generated_ids_tmp[k][2] = model.generate( 
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=0.3,
                    top_p = 0.9,
                )

                generated_ids_tmp[k][3] = model.generate(
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=0.5,
                    top_p = 0.9,
                )

                generated_ids_tmp[k][4] = model.generate(
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=0.7,
                    top_p = 0.9,
                )

                generated_ids_tmp[k][5] = model.generate(
                    current_input_ids,
                    max_new_tokens=max_new_tokens,
                    past_key_values=copy.deepcopy(past_key_values),
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    temperature=0.9,
                    top_p = 0.9,
                )
        for j in range(6):
            output_object = generated_ids_tmp[k][j]
            newly_generated_ids = output_object.sequences[:, start_index_for_slicing:]
            response[k][j] = final_response + tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)
            cand_final_response = response[k][j]
            response_enc = tokenizer7B(
                cand_final_response,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            cand_input_ids = response_enc["input_ids"].to(device)
            cand_attention_mask = response_enc["attention_mask"].to(device)

            cand_value[k][j] = value_model(
                current_input_ids=cand_input_ids,
                current_attention_mask=cand_attention_mask,
                code_input_ids=code_input_ids,
                code_attention_mask=code_attention_mask
            )

        maxidk = None
        maxidj = None

        selected_response = ""
        maxx = -998244353
        for k in range(step_size):
            for j in range(6):
                if cand_value[k][j] > maxx:
                    maxidk, maxidj = k,j
                    maxx = cand_value[k][j]
                    selected_response = response[k][j]

        best_output = generated_ids_tmp[maxidk][maxidj]
        tail_raw = tokenizer.decode(best_output.sequences[0, start_index_for_slicing:], 
                            skip_special_tokens=False)
        

        if (len(selected_response) - len(final_response) < 3 or "<|start_header_id|>assistant<|end_header_id|>" in tail_raw or "</s>" in tail_raw):
            break
        final_response = selected_response
        

        past_key_values = best_output.past_key_values
        current_input_ids = best_output.sequences

    last_index = -1
    for j in reversed(range(len(final_response))):
        if final_response[j] not in (' ','\t','\n','.'):
            last_index = j
            break

    final_response = final_response[:last_index + 1] + '.'

    data[i]["summary"] = final_response

    with open(OUTPUT_FILE,"a") as output_file:
        output_file.write(json.dumps(data[i]) + "\n")
