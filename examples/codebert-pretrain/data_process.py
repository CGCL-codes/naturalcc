import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer
from ncc.utils.data_util.codesearch_dataset import CodeSearchDataset, get_tokenized_dataset, save_tokenized_dataset

def main():
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # 加载数据集
    data_class = CodeSearchDataset(tokenizer=tokenizer)
    train, validation, test = data_class.load(subset="all")

    # 预处理数据集
    train_dataset = get_tokenized_dataset(train, tokenizer)
    valid_dataset = get_tokenized_dataset(validation, tokenizer)
    test_dataset = get_tokenized_dataset(test, tokenizer)

    # 保存预处理后的数据集
    save_path = Path("/mnt/silver/tanlei/datasets/codebert")
    save_path.mkdir(parents=True, exist_ok=True)

    save_tokenized_dataset(train_dataset, save_path / "train")
    save_tokenized_dataset(valid_dataset, save_path / "validation")
    save_tokenized_dataset(test_dataset, save_path / "test")

if __name__ == "__main__":
    main()