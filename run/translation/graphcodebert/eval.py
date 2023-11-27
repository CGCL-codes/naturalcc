# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from ncc import LOGGER
from ncc.eval.summarization.summarization_metrics import eval_accuracies
from ncc.tokenizers.tokenization import SPACE_SPLITTER
from ncc.utils.path_manager import PathManager
from ncc.utils.utils import move_to_cuda
from run.translation.graphcodebert import config
from run.translation.graphcodebert.cross_pair_dataset import CrossPairDataset, collater
from run.translation.graphcodebert.model import (
    GraphCodeBERT,
    load_checkpoint,
)

from run.translation.graphcodebert.bleu import compute_bleu

parser = argparse.ArgumentParser()
parser.add_argument("--SRC_LANG", default='csharp', type=str, help="source language")
parser.add_argument("--TGT_LANG", default='java', type=str, help="target language")
parser.add_argument("--use_best_model", action='store_false', help="best/last model")
parser.add_argument("--topk", default=3, type=int, help="topk")
parser.add_argument("--dataset", default='codetrans', type=str, help="dataset")
parser.add_argument(
    '--out_file', '-o', type=str, help='output generated file',
    default=None,
)
args = parser.parse_args()

if args.dataset == 'avatar':
    from run.translation.graphcodebert.config.avatar import config

    DATA_PATH = config.DATA_PATH[args.topk]

elif args.dataset == 'codetrans':
    from run.translation.graphcodebert.config.codetrans import config

    DATA_PATH = config.DATA_PATH
SRC_LANG, TGT_LANG = args.SRC_LANG, args.TGT_LANG
MODEL_DIR = os.path.join(DATA_PATH, f"{SRC_LANG}-{TGT_LANG}")
PathManager.mkdir(MODEL_DIR)
if args.use_best_model:
    CHECKPOINT = os.path.join(MODEL_DIR, 'best_checkpoint.pt')
else:
    CHECKPOINT = os.path.join(MODEL_DIR, 'last_checkpoint.pt')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        DEVICE_NUM = 1
    else:
        device = torch.device("cpu")
        DEVICE_NUM = 0

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if DEVICE_NUM > 0:
        torch.cuda.manual_seed_all(config.SEED)

    vocab, model = GraphCodeBERT.build_model()
    LOGGER.info(f'eval on {CHECKPOINT}')
    load_checkpoint(filename=CHECKPOINT, model=model)

    test_dataset = CrossPairDataset(
        config, vocab, data_path=DATA_PATH, mode="test", src_lang=SRC_LANG, tgt_lang=TGT_LANG,
        dataset=args.dataset, topk=args.topk,
    )
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), collate_fn=collater,
                                 batch_size=2, num_workers=3)

    model.to(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        predictions = []
        for batch in tqdm(test_dataloader):
            # train
            if DEVICE_NUM > 0:
                batch = move_to_cuda(batch)
            src_tokens, src_masks, src_positions, attn_mask, _, _, indices = batch
            encoder_output = model.encoder_forward(src_tokens, src_masks, src_positions, attn_mask)
            preds = model.beam_decode(encoder_output[0], src_masks, max_length=500)
            for pred_tensor in preds:
                pred_tensor = pred_tensor[0].tolist()
                if model.bos in pred_tensor:
                    pred_tensor = pred_tensor[:pred_tensor.index(model.bos)]
                if model.eos in pred_tensor:
                    pred_tensor = pred_tensor[:pred_tensor.index(model.eos)]
                pred_sentence = vocab.decode(pred_tensor, clean_up_tokenization_spaces=False)
                predictions.append(pred_sentence)

        references = [[c.split() for c in code] if isinstance(code, list) else [code.split()]
                      for code in test_dataset.tgt_data['code'][:len(predictions)]]
        smoothed_bleu, _, _, _, _, _ = compute_bleu(
            reference_corpus=references,
            translation_corpus=[pred.split() for pred in predictions],
            smooth=True
        )
        smoothed_bleu = round(100 * smoothed_bleu, 2)

        references = {idx: code if isinstance(code, list) else [code]
                      for idx, code in enumerate(test_dataset.tgt_data['code'][:len(predictions)])}
        hypotheses = {idx: [pred] for idx, pred in enumerate(predictions)}
        bleu4, rouge_l, meteor = eval_accuracies(hypotheses, references, mode='test', filename=args.out_file)
        bleu4, rouge_l, meteor = map(lambda v: round(v, 2), (bleu4, rouge_l, meteor))
        LOGGER.info(
            f"{test_dataset.src_lang} -> {test_dataset.tgt_lang}, Smoothed BLEU: {smoothed_bleu}, RougeL: {rouge_l}, Meteor: {meteor}, BLEU-4: {bleu4}"
        )
