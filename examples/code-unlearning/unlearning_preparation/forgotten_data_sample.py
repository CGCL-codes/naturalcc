import argparse
from datasets import load_from_disk
import torch
import random


def example_filter(example):
    secret_mask = torch.BoolTensor(example['secret_mask'])
    return example['secret_mean_MA'] >= 0.9 and secret_mask.sum() >= 32  # Prioritize high-risk samples for unlearning


def main():
    ds_pii = load_from_disk(f"../sensitive_memorization/codeparrot-clean-train-secrets-probed-{args.model_name_or_path.split('/')[-1]}")
    ds_pii = ds_pii.filter(example_filter, num_proc=16)
    random.seed(42)
    
    indices = list(range(len(ds_pii)))
    random.shuffle(indices)

    for i in range(5):
        sampled_group = ds_pii.select(indices[i * args.k: (i + 1) * args.k])
        print(sampled_group)
        sampled_group.save_to_disk(f"../unlearning/data/{args.model_name_or_path.split('/')[-1]}_secret/{args.model_name_or_path.split('/')[-1]}_forgot_set_{args.k}_{i}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="codeparrot/codeparrot", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument('--k', type=int, default=32,
                        help="The number of forgotten samples.")
    args = parser.parse_args()
    print(args)

    main()
