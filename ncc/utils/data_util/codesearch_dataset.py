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

        train = dataset["train"]
        valid = dataset["valid"]
        test = dataset["test"]
        
        return train, valid, test

def preprocess_function(examples):
    # 拼接代码和文档字符串
    inputs = [code + " </s> " + doc if doc else code 
            for code, doc in zip(examples["func_code_string"], examples["func_documentation_string"])]
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

    # MLM 标签
    tokenized["labels"] = tokenized["input_ids"].copy()

    # RTD 标签：随机替换部分 token
    rtd_labels = []
    modified_input_ids = []
    for input_ids in tokenized["input_ids"]:
        rtd_label = [0] * len(input_ids)  # 初始所有位置为 0
        modified_input_id = input_ids.copy()  # 复制原始 input_ids

        # 随机替换 15% 的 token
        for idx in range(len(input_ids)):
            if random.random() < 0.15 and input_ids[idx] != tokenizer.pad_token_id:
                modified_input_id[idx] = random.randint(0, tokenizer.vocab_size - 1)  # 随机替换
                rtd_label[idx] = 1  # 标记为被替换

        modified_input_ids.append(modified_input_id)
        rtd_labels.append(rtd_label)

    tokenized["input_ids"] = modified_input_ids
    tokenized["rtd_labels"] = rtd_labels

    return tokenized
    
def get_tokenized_dataset(dataset):
    columns_to_remove = ['repository_name', 'func_path_in_repository', 'func_name', 
                        'whole_func_string', 'language', 'func_code_tokens', 
                        'func_documentation_tokens', 'split_name', 'func_code_url']

    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=columns_to_remove
    )

    return tokenized_dataset

def load_tokenized_dataset(save_path):
    return load_from_disk(save_path)

def save_tokenized_dataset(tokenized_dataset, save_path):
    tokenized_dataset.save_to_disk(save_path)
    print(f"数据集已保存到: {save_path}")

class CustomDataCollatorForMLMAndRTD(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        rtd_labels = []
        filtered_examples = []
        for example in examples:
            if "rtd_labels" not in example:
                example["rtd_labels"] = [0] * len(example["input_ids"])

            filtered_example = {
                "input_ids": example["input_ids"],
                "labels": example["labels"]
            }
            rtd_labels.append(torch.tensor(example["rtd_labels"], dtype=torch.long))
            filtered_examples.append(filtered_example)

        batch = super().__call__(filtered_examples)
        padded_rtd_labels = pad_sequence(rtd_labels, batch_first=True, padding_value=0)
        batch["rtd_labels"] = padded_rtd_labels

        return batch