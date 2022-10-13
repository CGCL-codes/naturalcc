import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ncc_dataset.opencl import (
    LANGUAGES,
    ATTRIBUTES_DIR,
)
from ncc.eval.mapping import mapping_metrics
from ncc.models.mapping import DecisionTree
from ncc.utils.file_ops import json_io

SEED = 204


def encode_1hot(y: np.array) -> np.array:
    """ 1-hot encode labels """
    labels = np.vstack([np.expand_dims(x, axis=0) for x in y])
    l2 = [x[0] for x in labels]
    l1 = [not x for x in l2]
    return np.array(list(zip(l1, l2)), dtype=np.int32)


def cli_main():
    data = []
    for i, platform in enumerate(LANGUAGES):
        def get_attr(attr):
            oracle_file = os.path.join(ATTRIBUTES_DIR, platform, f'train.{attr}')
            with open(oracle_file, 'r') as reader:
                out = [json_io.json_loads(line) for line in reader]
            return np.asarray(out)

        def get_src_tokens():
            transfer = get_attr("transfer")
            comp = get_attr("comp")
            mem = get_attr("mem")
            coalesced = get_attr("coalesced")
            localmem = get_attr("localmem")
            wgsize = get_attr("wgsize")
            return np.array([
                transfer / (comp + mem),  # F1
                coalesced / mem,  # F2
                localmem / mem * wgsize,  # F3
                comp / mem,  # F4
            ]).T

        platform_name = mapping_metrics.platform2str(platform)
        devices = get_attr('oracle')
        benchmarks = get_attr('benchmark')
        runtime_cpus = get_attr('runtime_cpu')
        runtime_gpus = get_attr('runtime_gpu')

        # optimal mappings
        src_tokens = get_src_tokens()
        ground_truth = np.array([1 if x == "GPU" else 0 for x in devices])

        # 10-fold cross-validation
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        for j, (train_ids, test_ids) in enumerate(kf.split(src_tokens, ground_truth)):
            # decisoin tree model
            model = DecisionTree.build_model()
            # train
            model.fit(src_tokens=src_tokens[train_ids], ground_truth=ground_truth[train_ids])
            # accuracy
            predictions = model(src_tokens=src_tokens[test_ids])
            gt = ground_truth[test_ids]
            correct = (predictions == gt)
            # runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA)
            gt_runtimes = (runtime_cpus if platform == "amd" else runtime_gpus)[test_ids]
            pred_runtimes = [
                (runtime_cpus if pred == 0 else runtime_gpus)[idx]
                for idx, pred in zip(test_ids, predictions)
            ]
            speedup = gt_runtimes / pred_runtimes

            # record results
            for benchmark_, o_, p_, correct_, p_speedup_ in \
                zip(benchmarks[test_ids], ground_truth[test_ids], predictions, correct, speedup):
                data.append({
                    "Model": model.__class__.__name__,
                    "Platform": platform_name,
                    'Benchmark': mapping_metrics.escape_benchmark_name(benchmark_),
                    'Benchmark Suite': mapping_metrics.escape_suite_name(benchmark_),
                    "Oracle Mapping": o_,
                    "Predicted Mapping": p_,
                    "Accuracy": correct_,
                    "Speedup": p_speedup_,
                })
            del model
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
