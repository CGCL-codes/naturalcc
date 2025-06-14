from ncc.utils.data_util.base_dataset import BaseDataset
import random
from datasets import load_dataset, load_from_disk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

class CodeSearchDataset(BaseDataset):
    def __init__(self, tokenizer, max_length=512):
        super().__init__(tokenizer, max_length)

    def load(self, subset="all"):
        dataset = self.dataset_config["code_search_net"]
        dataset = load_dataset(dataset, name=subset)
        print(dataset.keys())       
        train = dataset["train"]
        valid = dataset["validation"]
        test = dataset["test"]
        
        return train, valid, test

def process_func(examples, tokenizer):
    inputs = [code + " </s> " + doc if doc else code 
              for code, doc in zip(examples["func_code_string"], examples["func_documentation_string"])]
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    return tokenized
def get_tokenized_dataset(dataset, tokenizer, num_proc=16):
    columns_to_remove = ['repository_name', 'func_path_in_repository', 'func_name', 
                        'whole_func_string', 'language', 'func_code_tokens', 
                        'func_documentation_tokens', 'split_name', 'func_code_url']

    tokenized_dataset = dataset.map(
        lambda examples: process_func(examples, tokenizer), 
        batched=True, 
        remove_columns=columns_to_remove,
        num_proc=num_proc  # 使用多线程处理
    )

    return tokenized_dataset

def load_tokenized_dataset(save_path):
    return load_from_disk(save_path)

def save_tokenized_dataset(tokenized_dataset, save_path):
    tokenized_dataset.save_to_disk(save_path)
    print(f"数据集已保存到: {save_path}")
