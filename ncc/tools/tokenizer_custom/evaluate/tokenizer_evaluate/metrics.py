import logging
import os
import shutil
import argparse
import sys
import json
import numpy as np
import json
import logging

from tqdm import tqdm
from collections import OrderedDict
from scipy import stats

from transformers import XLMRobertaTokenizerFast, AutoTokenizer

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    return os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")


# getting tokenizer
def get_tokenizer(tokenizer_path):
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except ValueError:
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, unk_token="<unk>")
    return tokenizer


def save_token_frequency(tokens_with_freq, decoded_tokens_with_freq, out_path, name):
    """Function to save token frequencies and log arguments to a file"""

    # copy current script to the output directory
    # shutil.copyfile(sys.argv[0], os.path.join(out_path, f"{name}_script.py"))
    # save the arguments
    # with open(os.path.join(out_path, f"{name}_args.txt"), "w") as log_file:
    #     log_file.write(" ".join(sys.argv[1:]))
    
    if not os.path.exists(out_path):
            os.makedirs(out_path)

    for save_name, save_object in [
        (f"{name}.json", tokens_with_freq),
        (f"{name}_decoded.json", decoded_tokens_with_freq),
    ]:
        save_path = os.path.join(out_path, save_name)
        with open(save_path, "w", encoding="utf-8") as outfile:
            logging.info(f"Writing frequencies to {save_path}")
            json.dump(
                OrderedDict(save_object),
                outfile,
                indent=2,
                ensure_ascii=False,
            )


def batch(iterator, batch_size):
    """Yield elements from iterator in batches of batch_size."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def compute_frequencies(data_list, tokenizer_path, name="token_frequencies",
                        identifiers_file_path=None, out_dir=None, alpha=None, vocab_size=None, type=None):
    # """Compute token frequencies for a given tokenizer and data."""
    # # languages_str = "-".join(languages)

    # # load the tokenizer
    # if not tokenizer_path:
    #     assert alpha is not None and vocab_size is not None and type is not None and out_dir is not None, (
    #         "If no tokenizer path is provided, alpha, vocab_size, type and out_dir must be provided."
    #     )
    #     tokenizer_path = get_tokenizer_path(out_dir, type, languages_str, alpha, vocab_size)
        
    tokenizer = get_tokenizer(tokenizer_path)

    # open the train data
    batch_size = 10000

    counter = {token_id: 0 for token_id in tokenizer.get_vocab().values()}
    for data_path in data_list:
        logging.info(f"Reading lines from {data_path}")
        with open(data_path, "r") as f:
            # go through the file line by line in batches
            # NOTE: we strip the newline character from the end of each line
            # TODO: maybe we shouldn't do this?
            for line_batch in tqdm(batch(map(lambda s: s.rstrip(), f), batch_size)):
                for tokenized_line in tokenizer(line_batch)["input_ids"]:
                    for id in tokenized_line:
                        counter[id] += 1

    # regularize the counter
    with open(args.data_list[0], 'r') as f:
        lines = f.readlines()
    data_size = len(lines)
    vocab_size = len(tokenizer.get_vocab())
    logger.info("data size: {}".format(data_size))
    logger.info("vocab size: {}".format(vocab_size))
    counter = {k: int(v * vocab_size / data_size) for k, v in counter.items()}

    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    tokens_with_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    decoded_tokens_with_freq = [
        (id_to_token[token_id], freq) for token_id, freq in tokens_with_freq
    ]

    # save_token_frequency(
    #     tokens_with_freq, decoded_tokens_with_freq, tokenizer_path, name
    # )
    save_token_frequency(
        tokens_with_freq, decoded_tokens_with_freq, args.out_dir, args.name
    )

def compute_gaussian(frequencies_path):
    # load data
    with open(frequencies_path) as f:
        frequencies_dict = json.load(f)
    
    data=[frequencies_dict.get(id) for id in frequencies_dict.keys()]
    data.remove(0)
    
    data = np.array(data)
    # 归一化处理
    data = (data - np.mean(data)) / np.std(data)
    
    # 拟合高斯分布
    loc, scale = stats.norm.fit(data)
    
    # 计算拟合结果
    mean = np.mean(data)
    std = np.std(data)
    kurtosis = stats.kurtosis(data)
    
    return mean, std, kurtosis

def compute_interval_distribution(frequencies_path):
    # load data
    with open(frequencies_path) as f:
        frequencies_dict = json.load(f)
    
    frequencies=[frequencies_dict.get(id) for id in frequencies_dict.keys()]
    # data.remove(0)
    
    # 0 1-10 11-100 101-1000 1000-
    counts=[0,0,0,0,0]
    
    for f in frequencies:
        if f==0:
            counts[0]+=1
        elif f>0 and f<=10:
            counts[1]+=1
        elif f>10 and f<=100:
            counts[2]+=1
        elif f>100 and f<=1000:
            counts[3]+=1
        else:
            counts[4]+=1
    proportion=[cnt/sum(counts) for cnt in counts]
    return counts, proportion

# calculate_average_length_of_token_sequence
def compute_average_token_length_of_input(tokenizer_path, file_path):
    count=0
    length=0
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            length += len(tokenizer.tokenize(line))
            count += 1
    return length/count

# compute_subtoken_fertility
def compute_subtoken_fertility(tokenizer_path, file):
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    tokens_length = 0
    text_length = 0
    with open(file, 'r') as f:
        for line in f:
            tokens_length += len(tokenizer.tokenize(line))
            text_length += len(line.split(' '))
    return tokens_length/text_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_list", nargs="+", help="<Required> Set flag", required=True,
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    # parser.add_argument(
    #     "-l",
    #     "--languages",
    #     nargs="+",
    #     required=False,
    #     help="List of languages the tokenizer was trained on.",
    # )
    
    # tokenizer parameters
    parser.add_argument(
        "--tokenizer_path", type=str, required=False, default=None
    )
    parser.add_argument(
        "--identifiers_file_path", type=str, required=False, default="/data/sub3/Doo/datasets/CodeSearchNet/atxtfile/validation_code_identifiers_exclude.txt"
    )
    parser.add_argument(
        "-a", "--alpha", type=str, required=False, help="Balancing coefficient alpha."
    )
    parser.add_argument("-v", "--vocab_size", type=int, required=False)
    parser.add_argument("-t", "--type", type=str, required=False, default="unigram")
    parser.add_argument(
        "-n", "--name", type=str, required=False, default="token_frequencies"
    )
    
    args = parser.parse_args()
    compute_frequencies(**args.__dict__)
    
    save_file=os.path.join(args.out_dir, args.name+'.json')
    
    logger.info("***gaussian distribution***")
    mean, stat, kurtosis = compute_gaussian(save_file)
    logger.info("mean: {}".format(mean))
    logger.info("stat: {}".format(stat))
    logger.info("kurtosis: {}".format(kurtosis))
    
    logger.info("***interval distribution***")
    counts, proportion = compute_interval_distribution(save_file)
    logger.info("counts: {}".format(counts))
    logger.info("proportion: {}".format(proportion))
    
    logger.info("***average length of input token sequences***")
    code_length = compute_average_token_length_of_input(args.tokenizer_path, args.data_list[0])
    logger.info("code length: {}".format(code_length))
    documentation_length = compute_average_token_length_of_input(args.tokenizer_path, args.data_list[1])
    logger.info("documentation length: {}".format(documentation_length))
    
    logger.info("***fertility***")
    fertility = compute_subtoken_fertility(args.tokenizer_path, args.identifiers_file_path)
    logger.info("fertility: {}".format(fertility))
    