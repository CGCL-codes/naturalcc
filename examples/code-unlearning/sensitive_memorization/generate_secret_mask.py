import argparse
import os
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def mask_example(example):
    content = example['content'].encode('utf-8', 'ignore').decode('utf-8')
    secrets = eval(example['secrets'])
    sorted_secrets = sorted(secrets, key=lambda secret: secret['start'])
    # print(sorted_secrets)
    
    encoding = tokenizer(content, return_tensors='pt', max_length=max_seq_len, truncation=True, padding='max_length', return_offsets_mapping=True)
    content_token_ids = encoding.input_ids
    offset_mapping = encoding.offset_mapping
    
    secret_mask = torch.zeros_like(content_token_ids, dtype=torch.bool)
    for secret in sorted_secrets:
        secret_start = secret['start']
        secret_end = secret['end']
        offset_mapping_start_check = offset_mapping[..., 0] <= secret_end  # !!! < or <=
        offset_mapping_end_check = offset_mapping[..., 1] >= secret_start  # !!! > or >=
        secret_mask = secret_mask + (offset_mapping_start_check.int() * offset_mapping_end_check.int()).bool()
    secret_token_ids = content_token_ids[secret_mask]
    # print(secret_token_ids)
    # print(tokenizer.convert_ids_to_tokens(secret_token_ids))
    if secret_token_ids.shape[0] == 0:
        example['keep_flag'] = False
    else:
        example['keep_flag'] = True
    
    example['content'] = content
    example['secrets'] = str(sorted_secrets)
    example['content_token_ids'] = content_token_ids
    example['offset_mapping'] = offset_mapping
    example['secret_mask'] = secret_mask

    return example


def filter_example(example):
    content = example['content']
    secrets = eval(example['secrets'])
    
    encoding = tokenizer(content, return_tensors='pt', max_length=max_seq_len, truncation=True, padding='max_length', return_offsets_mapping=True)
    content_token_ids = encoding.input_ids
    offset_mapping = encoding.offset_mapping
    if secrets[0]['start'] > offset_mapping[0, -1, -1]:
        return False
    else:
        return True


def update_example(example):
    content = example['content']
    secrets = eval(example['secrets'])
    
    encoding = tokenizer(content, return_tensors='pt', max_length=max_seq_len, truncation=True, padding='max_length', return_offsets_mapping=True)
    content_token_ids = encoding.input_ids
    offset_mapping = encoding.offset_mapping
    if offset_mapping[0, -1, -1] == 0:
        return example
    elif offset_mapping[0, -1, -1] >= secrets[-1]['start']:
        return example
    else:
        truncation_index = len(secrets)
        for index in range(len(secrets)):
            if secrets[index]['start'] > offset_mapping[0, -1, -1]:
                truncation_index = index
                break
        example['secrets'] = str(secrets[:truncation_index])
        example['number_secrets'] = truncation_index
        return example


def main():
    dataset_path = f"./codeparrot-clean-train-secrets-masked-{args.model_name_or_path.split('/')[-1]}"
    if os.path.exists(dataset_path):
        ds_pii = load_from_disk(dataset_path)
    else:
        ds_pii = load_from_disk('codeparrot-clean-train-secrets-filtered')
        ds_pii = ds_pii.map(mask_example, num_proc=48)
        ds_pii = ds_pii.filter(lambda example: example['keep_flag'], batched=True, batch_size=1000, num_proc=48)
        ds_pii = ds_pii.filter(filter_example, num_proc=32)
        ds_pii = ds_pii.map(update_example, num_proc=32)
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
    
    # max_seq_len = 1024
    max_seq_len = 512

    main()
