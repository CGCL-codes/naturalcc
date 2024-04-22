from model import LSTMModel
from seq.dataset import Dataset, Vocab
import train

import json
import os
import argparse
import logging
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)
logging.getLogger().setLevel(logging.INFO)


def prepare_data(batch, device):
    x = batch["input_seq"].to(device)
    y = batch["target_seq"].to(device)
    ext = batch["extended"]
    # rel = batch["rel_mask"].to(device) if "rel_mask" in batch else None
    # child = batch["child_mask"].to(device) if "child_mask" in batch else None
    paths = batch["root_paths"].to(device) if "root_paths" in batch else None
    return x, y, ext, paths


def greater_than(list, value):
    """Return how many value greater than `value` in the input list

    Args:
        list (torch.Tensor): 
        value (int or torch.Tensor(size == [1])): 
    """
    return torch.sum(list.gt(value)).item()


def post_process(gts, preds, exts, ranks, probability, label, saved_file):
    """Split the prediction by file level, filter the special tokens(<s>, </s>, <pad>, <EOL>...), save the results in a .pth file

    Args:
        args:
        gts: ground truth tokens' index
        preds: prediction tokens' index
        ranks: ground truth tokens' predicted ranks
        probability: ground truth tokens' predicted probability
        label: 1: member, 0: non-member
        saved_file: the saved pth file.
    """
    cnt = 0
    results = []

    cur_gt, cur_pred, cur_rank, cur_probability = gts[0], preds[0], ranks[0], probability[0]
    for i in range(1, len(exts)):
        if exts[i] == 0:
            cnt += 1
            results.append({"gt": torch.tensor(cur_gt), "pred": torch.tensor(cur_pred),
                            "rank": torch.tensor(cur_rank), "prob": torch.tensor(cur_probability),
                            "label": torch.tensor(label)})
            cur_gt, cur_pred, cur_rank, cur_probability = gts[i], preds[i], ranks[i], probability[i]
        else:
            cur_gt.extend(gts[i])
            cur_pred.extend(preds[i])
            cur_rank.extend(ranks[i])
            cur_probability.extend(probability[i])

    results.append({"gt": torch.tensor(cur_gt), "pred": torch.tensor(cur_pred),
                            "rank": torch.tensor(cur_rank), "prob": torch.tensor(cur_probability),
                            "label": torch.tensor(label)})
    
    out_dir = ("/").join(saved_file.split("/")[0:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(results, saved_file)
    return cnt


def compute_accuracy(total_gt, total_pred, unk_idx):
    cnt = 0
    total = 0
    for gt, pred in zip(total_gt, total_pred):
        total += len(gt)
        for i in range(len(gt)):
            if gt[i] != unk_idx and  gt[i] == pred[i]: #ground truth isn't OOV and ground truth == predictions
                cnt += 1
    return cnt / total


def load_rank(model, vocab, dataset, rank_dir, metrics_fp, batch_size, device):
    model.eval()
    accuracy = {}
    for file_type, label in zip(("member", "non_member"), (1, 0)):
        logging.info("load the predicted ranks of the {} dataset".format("member" if label else "non-member"))
        collate_fn = partial(dataset[file_type].collate, vocab=vocab)
        dataloader = torch.utils.data.DataLoader(
            dataset[file_type],
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=16,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            )
        total_pred = []
        total_gt = []
        total_ext = []
        total_ranks = [] # the predicted ranks of ground truth

        total_probability = [] # the predicted probability of the ground truth

        for step, batch in enumerate(tqdm(dataloader, mininterval=30)):
            x, y, ext, paths = prepare_data(batch, device)
            with torch.no_grad():
                y_pred = model(x, y, ext, paths, return_loss = False)
                y_pred.squeeze()
                pred_probs = F.softmax(y_pred, dim=-1)
                pred_ids = y_pred.argmax(-1)
            # here we pad out the indices that have been evaluated before
                for i, ext_i in enumerate(ext):
                    y[i][:ext_i] = vocab.pad_idx
                for i in range(len(ext)):
                    idx = (y[i] != vocab.pad_idx)
                    total_pred.append(pred_ids[i][idx].cpu().tolist())
                    total_gt.append(y[i][idx].cpu().tolist())
                    total_ext.append(ext[i].item())

                    ranks, probs = [], []
                    for gt, prob in zip(y[i][idx], pred_probs[i][idx]):
                        ranks.append(greater_than(prob, prob[gt]) if gt != vocab.unk_idx else len(vocab) + 1) # if the token is the UNK token, set it's rank to be len(vocab) + 1
                        probs.append(prob[gt].item())
                    total_ranks.append(ranks)
                    total_probability.append(probs)
                
        
        assert len(total_gt) == len(total_pred) and len(total_pred) == len(total_ranks) and len(total_ranks) == len(total_probability)
        saved_file = os.path.join(rank_dir, "{}_ranks.pth".format("member" if label == 1 else "non_member"))
        total_samples = post_process(total_gt, total_pred, total_ext, total_ranks, total_probability, label, saved_file)
        
        accuracy[file_type] = compute_accuracy(total_gt, total_pred, vocab.unk_idx) # compute the accuracy
        logging.info("Eval on {}, saved at {}".format(total_samples, saved_file))
    
    accuracy = {name: "{:.6f}".format(num) for name, num in accuracy.items()}
    accuracy_str = json.dumps(accuracy)
    logging.info("accuracy: {}".format(accuracy_str))
    with open(metrics_fp, "w") as fout:
        fout.write(accuracy_str)
    
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a seqrnn(lstm) model")
    parser.add_argument("--base_dir", "-b", default="/pathto/data/membership_inference/target/lstm")
    parser.add_argument("--rank_dir", default="/pathto/data/membership_inference/target/lstm/ranks")
    
    parser.add_argument("--model_fp", "-m", default="model/best_model.pt", help="Relative filepath to saved model")
    parser.add_argument("--vocab_fp", "-v", default="vocab.pkl", help="Relative filepath to vocab pkl")
    parser.add_argument("--member_fp", default="train.txt", help="Relative filepath to training dataset")
    parser.add_argument("--non_member_fp", default="test.txt", help="Relative filepath to test dataset")
    parser.add_argument("--metrics_fp", default="accuracy.txt", help="Relative filepath to results")
    
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", default=4, help="Batch size")
    args = parser.parse_args()
    logging.info("Base dir: {}".format(args.base_dir))
    return args


def main():
    args = parse_args()
    base_dir = args.base_dir
    model_fp = os.path.join(base_dir, args.model_fp)
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
    logging.info('Created {} model!'.format(model_fp))

    new_checkpoint = {}
    checkpoint = torch.load(model_fp, map_location=torch.device('cpu'))
    for name, weights in checkpoint.items():
        name = name.replace('module.', '')
        new_checkpoint[name] = weights
    del checkpoint

    model.load_state_dict(new_checkpoint)
    # model.eval()
    logging.info("Loaded model from:{}".format(model_fp))

    logging.info("Loading ranks parameters %s", args)

    dataset = {}
    dataset['member'] =  Dataset(os.path.join(base_dir, args.member_fp))
    dataset['non_member'] = Dataset(os.path.join(base_dir, args.non_member_fp))
    logging.info(f"member dataset: {os.path.join(base_dir, args.member_fp)}, non_member dataset: {os.path.join(base_dir, args.non_member_fp)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model.to(device)
    load_rank(model, vocab, dataset, args.rank_dir, os.path.join(base_dir, args.metrics_fp),
                batch_size = args.batch_size, device=device)


if __name__ == "__main__":
    main()