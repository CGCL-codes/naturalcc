import os
import json
import time
import glob
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from ncc.utils.code_util.python.repo_level_code_completion.common_similar_context.rerank_utils import lexical_ranking, SemanticReranking
from ncc.utils.code_util.python.repo_level_code_completion.common_similar_context.utils import str2bool, file_distance, tokenize_nltk
from ncc.utils.code_util.python.repo_level_code_completion.common_similar_context.augment_with_cfc import *
from multiprocessing import Pool
from collections import defaultdict

CHUNK_SIZE = 10
SLIDING_WINDOW_SIZE = 5  # non-overlapping chunks if SLIDING_WINDOW_SIZE=CHUNK_SIZE
QUERY_LENGTH = 10  # last N lines from prompt will be query

repository_root = "/home/wanyao/talentan/cceval/data_mini/python/repos"  # get the data from authors

input_files = {
    "python": "/home/wanyao/talentan/cceval/data_mini/python/line_completion.jsonl",
    # "java": "../data/crosscodeeval_data/java/line_completion.jsonl",
    # "typescript": "../data/crosscodeeval_data/typescript/line_completion.jsonl",
    # "csharp": "../data/crosscodeeval_data/csharp/line_completion.jsonl"
}
output_path = "/home/wanyao/talentan/naturalcc/examples/test_repo_context/"
file_ext = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}

if __name__ == "__main__":
    print("Script started execution...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        type=str2bool,
        default=True,
        help="rerank the functions"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="sparse",
        choices=["sparse", "unixcoder", "codebert"],
        help="ranking function"
    )
    parser.add_argument(
        "--ranking_fn",
        type=str,
        default="bm25",
        choices=["tfidf", "bm25", "jaccard_sim", "cosine_sim", "edit_distance"],
        help="ranking function"
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="last_n_lines",
        choices=["last_n_lines", "groundtruth"],
        help="how to form query from prompt"
    )
    parser.add_argument(
        "--crossfile_distance",
        type=int,
        default=100,
        help="max distance to search for crossfile"
    )
    parser.add_argument(
        "--maximum_chunk_to_rerank",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_files",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_file_chunk",
        type=int,
        default=10,
        help="max chunks to return as cfc"
    )
    parser.add_argument(
        "--use_next_chunk_as_cfc",
        type=str2bool,
        default=False,
        help="use next code chunk as context"
    )
    parser.add_argument(
        "--skip_if_no_cfc",
        type=str2bool,
        default=False,
        help="skip adding examples if there is no crossfile context"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["java", "python", "typescript", "csharp"],
        help="language name"
    )
    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"
    if args.use_next_chunk_as_cfc:
        assert args.rerank
        assert args.query_type != "groundtruth"

    tgtfile_suffix = ""
    if args.rerank:
        tgtfile_suffix += f"_{args.ranking_fn}"

    args.num_processes = 60
    if args.ranking_fn == "cosine_sim":
        num_gpus = 7
        args.num_processes = num_gpus
        mp.set_start_method('spawn')
    tqdm_lock = mp.Manager().Lock()

    input_file = input_files[args.language]
    output_path = os.path.dirname(output_path)
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = output_filename + args.output_file_suffix + tgtfile_suffix + ".jsonl"
    output_file = os.path.join(output_path, output_filename)
    print("Parsed arguments:")
    print(f"Language: {args.language}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Number of processes: {args.num_processes}")

    output_examples = attach_data(args, input_file)
    with open(output_file, "w") as fw:
        for ex in output_examples:
            fw.write(json.dumps(ex))
            fw.write("\n")