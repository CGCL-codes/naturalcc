import math
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import torch

from ncc_dataset.opencl_large import (
    ATTRIBUTES_DIR,
)
from ncc.utils.file_ops import json_io
from ncc.eval.mapping import mapping_metrics
from ncc.models.mapping import StaticMapping
from ncc.utils.file_ops import json_io

SEED = 204


def cli_main():
    data = []
    for i, platform in enumerate(['amd', 'nvidia']):
        def get_attr(attr):
            oracle_file = os.path.join(ATTRIBUTES_DIR, f'{platform}.{attr}')
            with open(oracle_file, 'r') as reader:
                out = [json_io.json_loads(line) for line in reader]
            return np.asarray(out)

        platform_name = mapping_metrics.platform2str(platform)
        devices = get_attr('oracle')
        benchmarks = get_attr('benchmark')
        runtime_cpus = get_attr('runtime_cpu')
        runtime_gpus = get_attr('runtime_gpu')

        # staic mapping model
        model = StaticMapping.build_model(devices)

        # optimal mappings
        src_tokens = torch.from_numpy(np.zeros(len(devices)))
        ground_truth = torch.from_numpy(np.array([1 if x == 1 else 0 for x in devices]))

        predictions = model(src_tokens)
        accuracy = (predictions == ground_truth).tolist()
        # runtimes of baseline mapping (CPU on AMD, GPU on NVIDIA)
        gt_runtimes = (runtime_cpus if platform == "amd" else runtime_gpus)
        pred_runtimes = [
            (runtime_cpus if pred == 0 else runtime_gpus)[idx]
            for idx, pred in enumerate(predictions)
        ]
        speedup = gt_runtimes / pred_runtimes
        # record results
        for benchmark_, o_, p_, accuracy_, p_speedup_ in \
            zip(benchmarks, ground_truth, predictions, accuracy, speedup):
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
    print(benchmark_out)
    out = performance.groupby(['Platform'])[['Platform', 'Accuracy', 'Speedup']].mean()
    out['Accuracy'] = round(out['Accuracy'] * 100, 2)
    print(out)


if __name__ == '__main__':
    cli_main()
