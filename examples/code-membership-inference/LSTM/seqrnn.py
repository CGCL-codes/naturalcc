from model import LSTMModel
from seq.dataset import Dataset, Vocab
import train

import json
import os
import torch
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train a seqrnn(lstm) model")
    parser.add_argument("--base_dir", "-b", default="/pathto/data/membership_inference/target/lstm")
    parser.add_argument("--model_fp", "-m", default="model", help="Relative filepath to saved model")
    parser.add_argument("--vocab_fp", "-v", default="vocab.pkl", help="Relative filepath to vocab pkl")
    parser.add_argument("--train_fp", default="train.txt", help="Relative filepath to training dataset")
    parser.add_argument("--valid_fp", default="dev.txt", help="Relative filepath to validation dataset")
    parser.add_argument("--test_fp", default="test.txt", help="Relative filepath to test dataset")
    parser.add_argument("--metrics_fp", default="metrics.txt", help="Relative filepath to results")

    parser.add_argument("--batch_size", default=4, help="Batch size")
    parser.add_argument("--epochs", default=20, help="Max epochs")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()
    logging.info("Base dir: {}".format(args.base_dir))
    return args


def main():
    args = parse_args()
    base_dir = args.base_dir
    vocab = Vocab(os.path.join(base_dir, args.vocab_fp))
    pad_idx = vocab.pad_idx
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    n_ctx = 1000

    model = LSTMModel(
        vocab_size=len(vocab),
        n_embd=300,
        loss_fn=loss_fn,
        n_ctx=n_ctx,
    )
    # dataset = Dataset(os.path.join(base_dir, args.train_fp), os.path.join(base_dir, "train_ids.txt"))
    dataset = {}
    dataset['train'] = Dataset(os.path.join(base_dir, args.train_fp))
    dataset['valid'] = Dataset(os.path.join(base_dir, args.valid_fp))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    model.to(device)
    train.train(model, vocab, dataset, os.path.join(base_dir, args.metrics_fp), 
                loss_fn = loss_fn, lr = 1e-3, run_dir = os.path.join(args.base_dir, args.model_fp), batch_size = args.batch_size, max_epochs = args.epochs, device=device)


if __name__ == "__main__":
    main()