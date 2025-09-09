'''Download all the necessary models from HuggingFace'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_model_and_tokenizer(model_name):
    print("Loading model {} ...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    print("Model {} is loaded.".format(model_name))
    return tokenizer, model


if __name__ == '__main__':
    models = [
        'codeparrot/codeparrot-small',
        'codeparrot/codeparrot',
        'Salesforce/codegen-350M-mono',
        'Salesforce/codegen-2B-mono',
        'Qwen/Qwen2.5-Coder-7B',
    ]

    for model_name in models:
        get_model_and_tokenizer(model_name)
