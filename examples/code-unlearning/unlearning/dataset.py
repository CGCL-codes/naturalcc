import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class CodeDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, type_path, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path

        self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
        self.dataset.columns = self.dataset.columns.str.replace('\r', '')
        if self.type_path == 'train':
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.ngpu
            if len(self.dataset) != batch_size:
                raise Exception("Effective batch size should be the same as length of train set.")

    def convert_to_features(self, example_batch):
        doc_id = torch.tensor(example_batch['doc_id'], dtype=torch.int)
        input_, target_ = example_batch['text'], example_batch['text']
        
        source = self.tokenizer(input_, max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_, max_length=self.output_length, add_special_tokens=False, padding='max_length', truncation=True, return_tensors="pt")
        
        return source, targets, doc_id

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source, targets, doc_id = self.convert_to_features(data)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "doc_id": doc_id}

    def __len__(self):
        return len(self.dataset)


class CodeSecretDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, type_path, args):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.type_path = type_path

        self.dataset = load_from_disk(dataset_name)
        self.dataset = self.dataset.add_column('doc_id', list(range(len(self.dataset))))
        if self.type_path == 'train':
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.ngpu
            if len(self.dataset) != batch_size:
                raise Exception("Effective batch size should be the same as length of train set.")

    def __getitem__(self, index):
        data = self.dataset[index]
        input_, target_ = data['content'], data['content']

        # special_tokens = self.tokenizer.special_tokens_map
        # print("Special tokens:", special_tokens)
        
        source = self.tokenizer(input_, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_, max_length=512, add_special_tokens=False, padding='max_length', truncation=True, return_tensors="pt")

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        source_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        
        # secret_spans = []
        # prefix_spans = []
        # current_span = []
        # for i, (token, is_secret) in enumerate(zip(source_ids, torch.BoolTensor(data['secret_mask']).squeeze())):
        #     if is_secret:
        #         if not current_span:
        #             prefix_spans.append(source_ids[max(0, i-1):i])
        #         current_span.append(token)
        #     elif current_span:
        #         secret_spans.append(current_span)
        #         current_span = []
        # if current_span:
        #     secret_spans.append(current_span)
        # for i, (prefix_span, secret_span) in enumerate(zip(prefix_spans, secret_spans)):
        #     print(f"Secret Span {i+1}:")
        #     print(f"  Prefix Token IDs: {prefix_span}")
        #     print(f"  Secret Token IDs: {torch.stack(secret_span)}")
        
        # secret_token_ids = torch.LongTensor(data['secret_token_ids'])
        # print(secret_token_ids)
        # secret_prefix_token_ids = torch.LongTensor(data['secret_prefix_token_ids'])
        # print(secret_prefix_token_ids)
        
        item = {
            'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids, 'target_mask': target_mask,
            'secret_mask': torch.BoolTensor(data['secret_mask']).squeeze(),
            # 'secret_mean_MA': torch.tensor(data['secret_mean_MA']),
            'doc_id': torch.tensor(data['doc_id'])
        }
        # print(item)
        return item

    def __len__(self):
        return len(self.dataset)
