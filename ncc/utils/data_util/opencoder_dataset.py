# import re
# from datasets import load_dataset
# from ncc.utils.data_util.base_dataset import BaseDataset

# class OpenCoderDataset(BaseDataset):
#     def __init__(self, tokenizer, max_length=512):
#         super().__init__(tokenizer, max_length)

#     def load(self, type):
#         dataset = self.dataset_config["repobench_"+language]
#         dataset = load_dataset(dataset)




# 将下载的数据存储到json文件中，python
import argparse
import json
import os
from typing import Optional
# import argparse:
# 用于解析命令行参数。
# import json:
# 处理和生成 JSON 格式的数据。
# import os:
# 访问系统环境变量和路径相关操作。
# from typing import Optional:
# 用于定义可选参数类型，标明某些函数可以返回 None。
import boto3
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from smart_open import open as sopen
from transformers import AutoTokenizer
# import boto3:
# 用于 AWS SDK，进行 S3 相关操作。
# from datasets import load_dataset:
# 从 Hugging Face Datasets 中加载预定义的 The Stack v2 数据集。
# from dotenv import load_dotenv:
# 读取 .env 文件中的环境变量，如 AWS 和 Hugging Face token。
# from loguru import logger:
# 用于日志记录，记录脚本执行过程中的信息、警告和错误。
# from smart_open import open as sopen:
# 处理 S3 中的文件访问功能。
# from transformers import AutoTokenizer:
# 从 Transformers 库加载预训练的 Llama-3.2 语言模型的 tokenizer，用于对代码进行分词。
# load_dotenv()
# 从 .env 文件加载环境变量，比如 AWS 的密钥和 Hugging Face 的访问令牌。
MIN_LINES = 10  # minimum number of lines in a file
MAX_LINES = 10000  # maximum number of lines in a file 
MAX_TOKENS = 128000  # maximum number of tokens in a file
# 最大允许的代码 tokens 数，超出这个范围的代码将被过滤掉。
LANGUAGES = ["Python"]
# 指定要处理的编程语言。
# TOKENIZER = AutoTokenizer.from_pretrained("/home/wanyao/wangchen/models/llama/Llama-3.1-8B")
# 加载 Llama-3.2 预训练的 tokenizer，用于对代码进行分词以获取 token 数量。

s3 = session.client("s3")

def download_blob(id: str, encoding: str) -> Optional[str]:
    s3_url = f"s3://softwareheritage/content/{id}"

    try:
        with sopen(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            content = fin.read().decode(encoding)
    except Exception as e:
        logger.error(f"Failed downloading from {s3_url} with error: {e}")
        return None

    return content

def load_split(language: str):
    dataset = load_dataset(
        "/home/wanyao/wangchen/datasets/code/bigcode/the-stack-v2-dedup/Python",
        # data_dir=f"data/{language}",
        # token=os.environ["HF_TOKEN"],
        split="train",
    )
    

# 对每一行数据进行映射，将文件的 blob 内容通过 download_blob 函数读取并解码。
# 提取所需字段并格式化。        
    dataset = dataset.map(
        lambda row: {
            "content": download_blob(row["blob_id"], row["src_encoding"]),
            "licenses": row["detected_licenses"],
            "license_type": row["license_type"],
            "host_url": "https://github.com",
            "repo_name": row["repo_name"],
            "file_path": row["path"],  # path within the repo
            "language": row["language"],
            "extension": row["extension"],
            "branch": row["branch_name"],
            "revision_id": row["revision_id"],  # SWH revision (commit) id
            "commit_date": row["committer_date"].isoformat(),
        },
        num_proc=500,
        remove_columns=dataset.column_names
    ).filter(
        lambda row: row["content"] is not None and MIN_LINES <= len(row["content"].splitlines()) <= MAX_LINES
    )
    # .filter:
# 过滤掉代码行数小于 MIN_LINES 或大于 MAX_LINES 的文件。

    # 计算 token 数并过滤:
    # 计算每一行代码的 token 数量，如果少于 MAX_TOKENS，将其加入最终的 tokenised_dataset。
    # tokenised_dataset = []
    # for row in dataset:
    #     num_tokens = len(TOKENIZER.tokenize(row["content"]))
    #     if num_tokens <= MAX_TOKENS:
    #         row["num_tokens"] = num_tokens  # approximate token count with Llama 3.2 tokenizer
    #         tokenised_dataset.append(row)

    # logger.info(f"Loaded {len(tokenised_dataset)} files for {language}")
    # return tokenised_dataset
    # logger.info:
    # 输出日志，显示加载的文件数。
    # return tokenised_dataset:
    # 返回经过筛选、分词处理后的数据集。
    logger.info(f"Loaded {len(dataset)} files for {language}")
    return dataset

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--languages", nargs="+", default="Python")
    # args = parser.parse_args()
    # args.languages=["Python"]
    for language in ["Python"]:
        dataset = load_split(language)
        with open(f"./output/{language}_stack_v2.jsonl", "w") as fout:
            for row in dataset:
                fout.write(json.dumps(row) + "\n")

    logger.info("Done!")
