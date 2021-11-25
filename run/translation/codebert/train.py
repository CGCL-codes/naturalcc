# -*- coding: utf-8 -*-

import argparse
import os
import random

import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)

from ncc import LOGGER
from ncc.data import constants
from ncc.utils.path_manager import PathManager
from ncc.utils.utils import move_to_cuda
from run.translation.bleu import compute_bleu
from run.translation.codebert.cross_pair_dataset import CrossPairDataset, collater
from run.translation.codebert.model import (
    CodeBERT,
    load_checkpoint, save_checkpoint,
)

parser = argparse.ArgumentParser()
parser.add_argument("--SRC_LANG", default='java', type=str, help="source language")
parser.add_argument("--TGT_LANG", default='csharp', type=str, help="target language")
parser.add_argument("--topk", default=1, type=int, help="topk")
parser.add_argument("--gpus", default=1, type=int, help="device num")
parser.add_argument("--dataset", default='codetrans', type=str, help="dataset")
args = parser.parse_args()

if args.dataset == 'avatar':
    from run.translation.codebert.config.avatar import config

    DATA_PATH = config.DATA_PATH[args.topk]
elif args.dataset == 'codetrans':
    from run.translation.codebert.config.codetrans import config

    DATA_PATH = config.DATA_PATH

SRC_LANG, TGT_LANG, DEVICE_NUM = args.SRC_LANG, args.TGT_LANG, args.gpus
MODEL_DIR = os.path.join(DATA_PATH, f"{SRC_LANG}-{TGT_LANG}")
PathManager.mkdir(MODEL_DIR)
BEST_CHECKPOINT = os.path.join(MODEL_DIR, 'best_checkpoint.pt')
LAST_CHECKPOINT = os.path.join(MODEL_DIR, 'last_checkpoint.pt')

if __name__ == '__main__':
    if torch.cuda.is_available() and DEVICE_NUM > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        DEVICE_NUM = 0

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if DEVICE_NUM > 0:
        torch.cuda.manual_seed_all(config.SEED)

    vocab, model = CodeBERT.build_model()

    no_decay = ['bias', 'LayerNorm.weight']
    parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters, lr=config.LR, eps=constants.EPS)
    train_dataset = CrossPairDataset(vocab, data_path=DATA_PATH, mode="train", src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), collate_fn=collater,
                                  batch_size=config.TRAIN_BATCH_SIZE * torch.cuda.device_count(), num_workers=3, )
    valid_dataset = CrossPairDataset(vocab, data_path=DATA_PATH, mode="valid", src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), collate_fn=collater,
                                  batch_size=config.DEV_BATCH_SIZE, num_workers=3, )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dataloader) * config.TRAIN_EPOCHS * 0.1,
        num_training_steps=len(train_dataloader) * config.TRAIN_EPOCHS,
    )

    init_epoch, best_bleu4, best_epoch = 1, 0, 0
    if os.path.exists(LAST_CHECKPOINT):
        init_epoch, best_bleu4 = load_checkpoint(filename=LAST_CHECKPOINT, model=model, optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler)
    else:
        LOGGER.info(f"No {LAST_CHECKPOINT} to initialize model")
    LOGGER.info(
        f"Start training epoch {init_epoch:>3}/{config.TRAIN_EPOCHS:>3}, best bleu4: {best_bleu4:.2f}"
    )
    if DEVICE_NUM > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)


    def CrossEntropyLoss(logits, labels, mask):
        logits_mask = (mask[:, 1:] != 0).bool()
        logits = logits[:, :-1, :].contiguous()[logits_mask]
        labels = labels[:, 1:].contiguous()[logits_mask]
        loss = F.cross_entropy(
            input=logits.view(-1, logits.size(-1)),
            target=labels.view(-1),
            ignore_index=vocab.pad(),
            reduction='sum',
        )
        loss = loss / mask.size(0)
        return loss


    for epoch in range(1, 1 + config.TRAIN_EPOCHS):
        def train():
            loss, num_sample = 0., 0
            model.train()
            if DEVICE_NUM > 0:
                torch.cuda.empty_cache()
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                # train
                if DEVICE_NUM > 0:
                    batch = move_to_cuda(batch)
                with torch.cuda.amp.autocast(enabled=True):
                    src_tokens, src_masks, tgt_tokens, tgt_masks, _ = batch
                    logits = model.forward(src_tokens, src_masks, tgt_tokens, tgt_masks)
                    batch_loss = CrossEntropyLoss(logits, tgt_tokens, mask=tgt_masks)
                if DEVICE_NUM > 1:
                    batch_loss = batch_loss.mean()

                loss += batch_loss.item() * src_tokens.size(0)
                num_sample += src_tokens.size(0)

                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # batch_loss.backward()
                # optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if batch_idx % config.LOG_INTERVAL == 0:
                    LOGGER.info(
                        f"Epoch {epoch:>3}/{config.TRAIN_EPOCHS:>3}, Batch {batch_idx:>3}/{len(train_dataloader):>3}, train loss: {loss / num_sample:.4f}"
                    )

            loss = round(loss / num_sample, 4)
            return loss


        train_loss = train()
        LOGGER.info(
            f"Epoch {epoch:>3}/{config.TRAIN_EPOCHS:>3}, train loss: {train_loss:.4f}"
        )


        def valid(cal_loss=True, cal_bleu=True):
            model.eval()

            if cal_loss:
                if DEVICE_NUM > 0:
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    loss, num_sample = 0., 0
                    for batch in valid_dataloader:
                        # train
                        if DEVICE_NUM > 0:
                            batch = move_to_cuda(batch)
                        with torch.cuda.amp.autocast(enabled=True):
                            src_tokens, src_masks, tgt_tokens, tgt_masks, _ = batch
                            logits = model.forward(src_tokens, src_masks, tgt_tokens, tgt_masks)
                            batch_loss = CrossEntropyLoss(logits, tgt_tokens, mask=tgt_masks)
                        if DEVICE_NUM > 1:
                            batch_loss = batch_loss.mean()

                        loss += batch_loss.item() * src_tokens.size(0)
                        num_sample += src_tokens.size(0)
                loss = round(loss / num_sample, 4)
            else:
                loss = 0

            if cal_bleu:
                if DEVICE_NUM > 0:
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    predictions = []
                    for batch in valid_dataloader:
                        # train
                        if DEVICE_NUM > 0:
                            batch = move_to_cuda(batch)
                        src_tokens, src_masks, _, _, _ = batch
                        if DEVICE_NUM > 1:
                            encoder_output = model.module.encoder_forward(src_tokens, src_masks)
                            preds = model.module.greedy_decode(encoder_output, src_masks)
                        else:
                            encoder_output = model.encoder_forward(src_tokens, src_masks)
                            preds = model.greedy_decode(encoder_output, src_masks)
                        for pred_tensor in preds:
                            pred_tensor = pred_tensor.tolist()
                            if vocab.eos() in pred_tensor:
                                pred_tensor = pred_tensor[:pred_tensor.index(vocab.eos())]
                            pred_sentence = vocab.decode(pred_tensor, clean_up_tokenization_spaces=False)
                            # pred_sentence = SPACE_SPLITTER.sub(" ", pred_sentence)
                            predictions.append(pred_sentence.split())

                    references = [[code.split()] for code in valid_dataset.tgt_code]
                    bleu_score, _, _, _, _, _ = compute_bleu(references, predictions, 4, True)

                    # import sacrebleu
                    # bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='none')
                    bleu4 = round(100 * bleu_score, 2)
            else:
                bleu4 = 0

            return loss, bleu4


        if epoch % config.SAVE_INTERVAL == 0:
            epoch_checkpoint = os.path.join(MODEL_DIR, f'{epoch}.pt')
            save_checkpoint(filename=epoch_checkpoint, model=model,
                            optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, bleu4=best_bleu4)
            LOGGER.info(f"save {epoch_checkpoint}")

        valid_loss, bleu4 = valid(cal_loss=False)
        if best_bleu4 <= bleu4:
            best_bleu4 = bleu4
            best_epoch = epoch
            save_checkpoint(filename=BEST_CHECKPOINT, model=model,
                            optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, bleu4=best_bleu4)
            LOGGER.info(f"update {BEST_CHECKPOINT}")
        LOGGER.info(
            f"Epoch {epoch:>3}/{config.TRAIN_EPOCHS:>3}, valid loss: {valid_loss:.4f}, valid bleu4: {bleu4:.2f}, best bleu4: {best_bleu4:.2f}"
        )

        save_checkpoint(filename=LAST_CHECKPOINT, model=model,
                        optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch, bleu4=best_bleu4)
        LOGGER.info(f"update {LAST_CHECKPOINT}")
        if epoch - best_epoch >= config.EARLY_STOP:
            LOGGER.info(f"early stop because of no improvement in {config.EARLY_STOP} epochs")
            exit()
