import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.opencl import (
    LANGUAGES,
    DATASET_DIR,
)
from ncc.criterions.mapping import DeepTuneLoss
from ncc.data import (
    indexed_dataset,
)
from ncc.data.dictionary import Dictionary
from ncc.data.mapping.language_pair_dataset import (
    LanguagePairDataset,
    collate,
)
from ncc.data.tools import data_utils
from ncc.data.wrappers.truncate_dataset import TruncateDataset
from ncc.eval.mapping import mapping_metrics
from ncc.models.mapping.deeptune import DeepTuneEncoder
from ncc.modules.common.initializers import xavier_normal
from ncc.utils.utils import move_to_cuda


def init(model):
    for p in model.parameters():
        xavier_normal(p)


def load_mmap_dataset(dataset):
    return indexed_dataset.MMapIndexedDataset(dataset)


def cli_main():
    SEED = 204
    BATCH_SIZE = 64
    MAX_SOURCE_POSITIONS = 1024
    EPOCH = 50

    from ncc.utils.set_seed import set_seed
    set_seed(SEED)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = os.environ.get('CUDA_VISIBALE_DEVICES', [0])[0]  # get first device as default
        torch.cuda.set_device(f'cuda:{device}')
    criterion = DeepTuneLoss(task=None, sentence_avg=-1)
    if use_cuda:
        criterion = criterion.cuda()

    data = []
    for i, platform in enumerate(LANGUAGES):
        DATA_DIR = os.path.join(DATASET_DIR, f'mapping/{platform}/data-mmap')

        def get_attr(attr):
            oracle_file = os.path.join(DATA_DIR, f'train.{attr}')
            with open(oracle_file, 'rb') as reader:
                out = pickle.load(reader)
            return np.asarray(out)

        platform_name = mapping_metrics.platform2str(platform)
        benchmarks = get_attr('benchmark')
        runtime_cpus = get_attr('runtime_cpu')
        runtime_gpus = get_attr('runtime_gpu')

        #################### load dataset ####################
        src_dataset = load_mmap_dataset(os.path.join(DATA_DIR, f'train.src_tokens'))
        src_dataset = TruncateDataset(src_dataset, truncation_length=MAX_SOURCE_POSITIONS, truncate_prefix=0)
        tgt_dataset = load_mmap_dataset(os.path.join(DATA_DIR, f'train.oracle'))

        src_dict = Dictionary.load(os.path.join(DATA_DIR, 'src_tokens.dict.jsonl'))
        src_aux = OrderedDict()
        src_aux['transfer'] = get_attr('transfer')
        src_aux['wgsize'] = get_attr('wgsize')

        tgt_dict = Dictionary.load(os.path.join(DATA_DIR, 'oracle.dict.jsonl'))

        dataset = LanguagePairDataset(
            src=src_dataset, src_sizes=src_dataset.sizes, src_dict=src_dict, src_aux=src_aux,
            tgt=tgt_dataset, tgt_sizes=tgt_dataset.sizes, tgt_dict=tgt_dict, tgt_aux=None,
            left_pad_source=True, max_source_positions=MAX_SOURCE_POSITIONS,
        )
        #################### load dataset ####################

        # build toy dataset for 10-fold cross validation
        tgt_data = [tgt_dataset[idx].item() for idx in range(len(tgt_dataset))]
        src_data = [None] * len(tgt_data)

        # 10-fold cross-validation
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        for j, (train_ids, test_ids) in enumerate(kf.split(src_data, tgt_data)):
            # deeptune model
            model = DeepTuneEncoder(dictionary=src_dict, embed_dim=64,
                                    rnn_cell='lstm', rnn_hidden_dim=64, rnn_dropout=0., rnn_num_layers=2,
                                    aux_dim=2, inner_dim=32, out_dim=2)
            if use_cuda:
                model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch_i in range(EPOCH):
                if dataset.shuffle:
                    random.shuffle(train_ids)
                train_batch_sampler = data_utils.batch_by_size(
                    train_ids,
                    num_tokens_fn=lambda *args: -1,
                    max_sentences=BATCH_SIZE,
                )
                train_dataloader = DataLoader(dataset=dataset,
                                              batch_sampler=train_batch_sampler,
                                              collate_fn=collate, )
                with tqdm(total=len(train_dataloader)) as t:
                    for sample_i, sample in enumerate(train_dataloader, start=1):
                        t.set_description(f'Epoch {epoch_i + 1}/{EPOCH} Batch {sample_i}/{len(train_dataloader)}')
                        if use_cuda:
                            sample = move_to_cuda(sample)
                        loss, sample_size, logging_output = criterion(model, sample)
                        loss.div_(sample_size)
                        t.set_postfix(loss=loss.item())
                        t.update()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            # test accuracy
            test_batch_sampler = data_utils.batch_by_size(
                test_ids,
                num_tokens_fn=lambda *args: -1,
                max_sentences=BATCH_SIZE,
            )
            test_dataloader = DataLoader(dataset=dataset,
                                         batch_sampler=test_batch_sampler,
                                         collate_fn=collate, )
            predictions, ground_truth = [], []
            for sample in test_dataloader:
                if use_cuda:
                    sample = move_to_cuda(sample)
                hybrid_out, _ = model(**sample['net_input'])
                predictions.append(hybrid_out.max(dim=-1)[1])
                ground_truth.append(sample['target'].view(-1))
            predictions = torch.cat(predictions)
            ground_truth = torch.cat(ground_truth)

            accuracy = (predictions == ground_truth).tolist()
            # runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA)
            gt_runtimes = (runtime_cpus if platform == "amd" else runtime_gpus)[test_ids]
            pred_runtimes = [
                (runtime_cpus if pred == 0 else runtime_gpus)[idx]
                for idx, pred in zip(test_ids, predictions)
            ]
            speedup = gt_runtimes / pred_runtimes

            # record results
            for benchmark_, o_, p_, accuracy_, p_speedup_ in \
                zip(benchmarks[test_ids], ground_truth, predictions, accuracy, speedup):
                data.append({
                    "Model": model.__class__.__name__,
                    "Platform": platform_name,
                    'Benchmark': mapping_metrics.escape_benchmark_name(benchmark_),
                    'Benchmark Suite': mapping_metrics.escape_suite_name(benchmark_),
                    "Oracle Mapping": o_,
                    "Predicted Mapping": p_,
                    "Accuracy": accuracy_,
                    "Speedup": p_speedup_,
                })
            del model, optimizer
    performance = pd.DataFrame(
        data, index=range(1, len(data) + 1), columns=[
            "Model",
            "Platform",
            "Benchmark",
            "Benchmark Suite",
            "Oracle Mapping",
            "Predicted Mapping",
            "Accuracy",
            "Speedup"
        ])
    benchmark_out = performance.groupby(['Platform', 'Benchmark Suite'])[['Platform', 'Accuracy', 'Speedup']].mean()
    benchmark_out['Accuracy'] = round(benchmark_out['Accuracy'] * 100, 2)
    benchmark_out['Speedup'] = round(benchmark_out['Speedup'], 2)
    print(benchmark_out)
    out = performance.groupby(['Platform'])[['Platform', 'Accuracy', 'Speedup']].mean()
    out['Accuracy'] = round(out['Accuracy'] * 100, 2)
    out['Speedup'] = round(out['Speedup'], 2)
    print(out)


if __name__ == '__main__':
    cli_main()
