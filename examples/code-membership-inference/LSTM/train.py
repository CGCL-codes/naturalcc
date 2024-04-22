#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from functools import partial
from typing import Set

import torch


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


def build_dataloader(dataset, batch_size, collate_fn, train_split=0.90):
    train_len = int(train_split * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, lengths=([train_len, len(dataset) - train_len])
    )
    logging.info("Batch size: {}".format(batch_size))
    logging.info(
        "Train / val split ({}%): {} / {}".format(
            100 * train_split, len(train_dataset), len(val_dataset)
        )
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    logging.info("len(train_dataloader) = {}".format(len(train_dataloader)))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(batch_size / 4),
        collate_fn=collate_fn,
        num_workers=16,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    logging.info("len(val_dataloader) = {}".format(len(val_dataloader)))
    return train_dataloader, val_dataloader


def build_train_dataloader(train_dataset, batch_size, collate_fn):
    logging.info("Batch size: {}".format(batch_size))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    logging.info("len(train_dataloader) = {}".format(len(train_dataloader)))
    return train_dataloader
    

def build_test_dataloader(test_dataset, batch_size, collate_fn):
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=16,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    logging.info("len(test_dataloader) = {}".format(len(test_dataloader)))
    return test_dataloader


def build_metrics(loss_fn, unk_idx_set: Set[int], pad_idx, ids_str):
    from ignite.metrics import Loss, TopKCategoricalAccuracy

    def strip(out, id_str="all"):
        if id_str != "all":
            ids = out[id_str]
            y_pred = out["y_pred"][ids]
            y = out["y"][ids]
        else:
            y_pred = out["y_pred"]
            y = out["y"]
        idx = y != pad_idx
        return y_pred[idx], y[idx]

    def topk_trans(id_str):
        def wrapped(out):
            y_pred, y = strip(out, id_str)
            for idx in unk_idx_set: # non-existing tokens
                y[y == idx] = -2
            return y_pred, y

        return wrapped

    def topk_ex_unk_trans(id_str):
        def wrapped(out):
            y_pred, y = strip(out, id_str)
            idx_tensor = torch.ones(y.shape, dtype=torch.bool)
            for idx in unk_idx_set:
                idx_tensor[y == idx] = False
            return y_pred[idx_tensor], y[idx_tensor]

        return wrapped

    def loss_trans():
        def wrapped(out):
            return strip(out)

        return wrapped

    metrics = {"_loss": Loss(loss_fn, loss_trans())}
    # for id_str in ["attr_ids", "leaf_ids"] + ["all"]:
    for id_str in ["all"]:
        # reporting metrics for attr and leaf only
        metrics["{}_acc".format(id_str)] = TopKCategoricalAccuracy(
            1, topk_trans(id_str)
        )

    return metrics


def build_evaluator(model, metrics, metrics_fp, device, pad_idx):
    from ignite.contrib.handlers import ProgressBar
    from ignite.engine import Engine, Events

    @Engine
    @torch.no_grad()
    def evaluator(engine, batch):
        model.eval()
        x, y, ext, paths = prepare_data(batch, device)
        y_pred = model(x, y, ext, paths)
        # here we pad out the indices that have been evaluated before
        for i, ext_i in enumerate(ext):
            y[i][:ext_i] = pad_idx
        res = {"y_pred": y_pred.view(-1, y_pred.size(-1)), "y": y.view(-1)}
        res.update(batch["ids"])
        return res

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    ProgressBar(bar_format="").attach(evaluator, metric_names=[])

    @evaluator.on(Events.COMPLETED)
    def log_val_metrics(engine):
        metrics = engine.state.metrics
        metrics = {name: "{:.4f}".format(num) for name, num in metrics.items()}
        # mrr
        metrics_str = json.dumps(metrics, indent=2, sort_keys=True)
        logging.info("val metrics: {}".format(metrics_str))
        with open(metrics_fp, "a") as fout:
            fout.write(metrics_str)
            fout.write("\n")

    return evaluator


def build_trainer(
    model,
    loss_fn,
    optimizer,
    train_dataloader,
    val_dataloader,
    run_dir,
    validator,
    device,
    score_fn=lambda engine: engine.state.metrics["all_acc"],
):
    from ignite.contrib.handlers import ProgressBar
    from ignite.engine import Engine, Events
    from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
    from ignite.metrics import RunningAverage

    @Engine
    def trainer(engine, batch):
        model.train()
        x, y, ext, paths = prepare_data(batch, device)
        loss = model(x, y, ext, paths, return_loss=True)
        loss = loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"batchloss": loss.item()}

    # # validation first
    # @trainer.on(Events.STARTED)
    # def validate(engine):
    #     validator.run(val_dataloader)

    RunningAverage(output_transform=lambda out: out["batchloss"]).attach(
        trainer, "batchloss"
    )
    ProgressBar(bar_format="").attach(trainer, metric_names=["batchloss"])

    # store the model before validation
    pre_model_handler = ModelCheckpoint(
        dirname=run_dir,
        filename_prefix="pre",
        n_saved=1,  # save all bests
        save_interval=1,
        require_empty=False,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, pre_model_handler, {"model": model}
    )

    # validation
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        validator.run(val_dataloader)

    # terminate on NaN
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # store the best model
    best_model_handler = ModelCheckpoint(
        dirname=run_dir,
        filename_prefix="best",
        n_saved=1,  # save only one
        score_name="val_acc",
        score_function=score_fn,
        require_empty=False,
    )
    validator.add_event_handler(Events.COMPLETED, best_model_handler, {"model": model})

    # Early stopping
    es_handler = EarlyStopping(patience=5, score_function=score_fn, trainer=trainer)
    validator.add_event_handler(Events.COMPLETED, es_handler)

    return trainer


def train(
    model,
    vocab,
    dataset,
    metrics_fp,
    loss_fn,
    lr,
    run_dir,
    batch_size,
    max_epochs,
    device,
    ids_str=None,
):
    collate_fn = partial(dataset['train'].collate, vocab=vocab)
    train_dataloader = build_train_dataloader(
        dataset['train'], batch_size=batch_size, collate_fn=collate_fn
    )

    collate_fn = partial(dataset['valid'].collate, vocab=vocab)
    val_dataloader = build_test_dataloader(
        dataset['valid'], batch_size=int(batch_size/2), collate_fn=collate_fn
    )
    metrics = build_metrics(loss_fn, (vocab.unk_idx, ), vocab.pad_idx, ids_str)

    # run the trainer and validator
    validator = build_evaluator(
        model=model,
        metrics=metrics,
        metrics_fp=metrics_fp,
        device=device,
        pad_idx=vocab.pad_idx,
    )
    trainer = build_trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        run_dir=run_dir,
        validator=validator,
        device=device,
    )
    trainer.run(train_dataloader, max_epochs=max_epochs)


def eval_model(
    model, vocab, test_dataset, metrics_fp, loss_fn, batch_size, device, ids_str
):
    collate_fn = partial(test_dataset.collate, pad_idx=vocab.pad_idx)
    test_dataloader = build_test_dataloader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn
    )
    metrics = build_metrics(loss_fn, (vocab.unk_idx, ), vocab.pad_idx, ids_str)

    # run the evaluator
    evaluator = build_evaluator(
        model=model,
        metrics=metrics,
        metrics_fp=metrics_fp,
        device=device,
        pad_idx=vocab.pad_idx,
    )
    evaluator.run(test_dataloader)
