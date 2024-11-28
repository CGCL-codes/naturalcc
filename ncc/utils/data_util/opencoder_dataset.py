import re
from datasets import load_dataset
from ncc.utils.data_util.base_dataset import BaseDataset

class OpenCoderDataset(BaseDataset):
    def __init__(self, tokenizer, max_length=512):
        super().__init__(tokenizer, max_length)

    def load(self, type):
        dataset = self.dataset_config["repobench_"+language]
        dataset = load_dataset(dataset)




