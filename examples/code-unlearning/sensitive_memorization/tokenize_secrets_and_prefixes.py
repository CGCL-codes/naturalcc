import argparse
import os
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize_secret(example):
    content = example['content']
    secrets = eval(example['secrets'])
    example['secret_token_ids'] = []
    for index in range(len(secrets)):
        secret = secrets[index]['value']
        secret_token_ids = tokenizer.encode(secret, return_tensors='pt', max_length=max_secret_len, truncation=True, padding='max_length')
        example['secret_token_ids'].append(secret_token_ids)
    example['secret_token_ids'] = torch.cat(example['secret_token_ids'], dim=0)
    return example


def tokenize_secret_prefix(example):
    content = example['content']
    secrets = eval(example['secrets'])
    example['secret_prefix_token_ids'] = []
    for index in range(len(secrets)):
        secret_prefix = content[:secrets[index]['start']]  # Extract the context leading up to the secret
        # secret_prefix_token_ids = tokenizer.encode(secret_prefix, return_tensors='pt')[..., -1 * max_prefix_len:]
        secret_prefix_token_ids = tokenizer.encode(secret_prefix, return_tensors='pt', max_length=max_prefix_len, truncation=True, padding='max_length')
        example['secret_prefix_token_ids'].append(secret_prefix_token_ids)
    example['secret_prefix_token_ids'] = torch.cat(example['secret_prefix_token_ids'], dim=0)
    return example


def main():
    dataset_path = f"./codeparrot-clean-train-secrets-tokenized-{args.model_name_or_path.split('/')[-1]}"
    if os.path.exists(dataset_path):
        ds_pii = load_from_disk(dataset_path)
    else:
        ds_pii = load_from_disk(f"codeparrot-clean-train-secrets-masked-{args.model_name_or_path.split('/')[-1]}")
        ds_pii = ds_pii.map(tokenize_secret, num_proc=32)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        ds_pii = ds_pii.map(tokenize_secret_prefix, num_proc=16)
        ds_pii.save_to_disk(dataset_path)
    print(ds_pii)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="codeparrot/codeparrot", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    args = parser.parse_args()
    print(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    max_secret_len = 32
    max_prefix_len = 128

    main()
