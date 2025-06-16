import jsonlines
import json
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
import argparse

prompt_template = """
I possess a triple that includes instruction, output, and complex intermediate thinking steps. However, the intermediate thinking steps are overly elaborate, and I require a relatively concise thinking step and an extremely concise one.
I will provide you with:
{{
"instruction": "{}",
"complex_think_step": "{}",
"output": "{}"
}}
Please output in following json foramt:
{{
"normal_think_step": "?",
"simple_think_step": "?"
}}
"""
#加载模型
model_name = "Qwen/Qwen2.5-72B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def chat(prompt):
    # prompt = "567+2=?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Define arguments
def parse_args():
    parser = argparse.ArgumentParser(description="msg")
    parser.add_argument("--min", type=int)
    parser.add_argument("--max", type=int)
    return parser.parse_args()
args = parse_args()
# 打开JSON文件，你可以根据实际情况调整路径,这里路径仅供参考
json_objects = []
origin_data_path='/home/wanyao/wucai/work/data/code_instructions_filtered_CoT.jsonl'
with jsonlines.open(origin_data_path) as reader:
    i=0
    for data in reader:
        i = i + 1
        if i<args.min or i>args.max:
            continue
        if i%100==0:
            print(i)
        try:
            prompt = prompt_template.format(data['instruction'], data['CoT'], data['output'])
            ans = chat(prompt)
            json_ans = json.loads(re.search(r'\{.*?\}', ans, re.DOTALL).group(), strict=False)
            data['CoT_normal'] = json_ans['normal_think_step']
            data['CoT_simple'] = json_ans['simple_think_step']
            json_objects.append(data)
        except:
            data['CoT_normal'] = "exception!!!CoT_normal"
            data['CoT_simple'] = "exception!!!CoT_simple"
            json_objects.append(data)

with jsonlines.open('./data/code_instructions_filtered_CoT_complex_{}_{}.jsonl'.format(args.min,args.max), 'a') as writer:
    for item in json_objects:
        writer.write(item)

# str = """
# ```json
# {
#   "normal_think_step": "1. Prepare the data:
#    - Load and preprocess the email dataset.
#    - Split the data into training and testing sets.
# 2. Build the neural network model:
#    - Define the input layer with the appropriate shape.
#    - Add hidden layers with activation functions (e.g., ReLU).
#    - Add an output layer with a sigmoid activation function for binary classification.
# 3. Compile the model:
#    - Specify the optimizer (e.g., Adam), loss function (e.g., binary cross-entropy), and metrics (e.g., accuracy).
# 4. Train the model:
#    - Fit the model to the training data with a specified number of epochs and batch size.
# 5. Evaluate the model:
#    - Assess the model's performance on the test data and print the accuracy.",
#   "simple_think_step": "1. Load and split the data.
# 2. Build a neural network with input, hidden, and output layers.
# 3. Compile the model with Adam optimizer and binary cross-entropy loss.
# 4. Train the model on the training data.
# 5. Test the model and print the accuracy."
# }
# ```
# """