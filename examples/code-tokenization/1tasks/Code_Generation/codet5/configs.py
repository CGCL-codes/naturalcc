#!/usr/bin/env python3

import argparse
import logging
import multiprocessing
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, required=True, choices=["summarize", "concode", "translate", "refine", "defect", "clone"])
    parser.add_argument("--sub_task", type=str, default="")
    parser.add_argument("--lang", type=str, default="")
    parser.add_argument("--model_type", default="codet5", type=str, choices=["roberta", "bart", "codet5"])
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_filename", default=None, type=str)
    parser.add_argument("--dev_filename", default=None, type=str)
    parser.add_argument("--test_filename", default=None, type=str)
    parser.add_argument("--max_source_length", default=64, type=int)
    parser.add_argument("--max_target_length", default=32, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    if args.task == "summarize":
        args.lang = args.sub_task
    elif args.task in ["refine", "concode", "clone"]:
        args.lang = "java"
    elif args.task == "defect":
        args.lang = "c"
    elif args.task == "translate":
        args.lang = "c_sharp" if args.sub_task == "java-cs" else "java"
    return args


def set_dist(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.cpu_cont = multiprocessing.cpu_count()
    logger.warning("device=%s, n_gpu=%s, distributed=%s", device, args.n_gpu, bool(args.local_rank != -1))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if getattr(args, "n_gpu", 0) > 0:
        torch.cuda.manual_seed_all(args.seed)
