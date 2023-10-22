import json
import numpy as np
from scipy.stats import wilcoxon

from tqdm import tqdm

from metrics_evaluation.metrics.ruby.similarity import ruby, string_similarity, tree_similarity, graph_similarity

data = json.load(open("to-grade/conala.json", "r"))

models = ["baseline", "tranx-annot", "best-tranx", "best-tranx-rerank"]

all_rubies = []

for field in models:
    print(f"Results for {field}:")
    rubies = []
    graphs = []
    trees = []
    strings = []
    for d in tqdm(data):
        sample = d[field].replace("`", '"')
        reference = d["snippet"]
        rubies.append(ruby(sample, reference)[0])
        graphs.append(graph_similarity(sample, reference))
        trees.append(tree_similarity(sample, reference))
        strings.append(string_similarity(sample, reference))

    for vals, name in zip(
        [graphs, trees, strings, rubies], ["GRS", "TRS", "STS", "RUBY"]
    ):
        vals = [v if v is not None else 0 for v in vals]
        print(f"{name}: {np.mean(vals):.4f}+-{np.std(vals)}, ({min(vals)}-{max(vals)})")
    print("=" * 20)
    all_rubies.append(rubies)

for i, label in enumerate(models):
    for j, o_label in zip(range(i + 1, len(models)), models[i + 1 :]):
        print("=" * 20)
        print(f"{label} vs {o_label}")
        print(f"{label}: {np.mean(all_rubies[i]):.4f}")
        print(f"{o_label}: {np.mean(all_rubies[j]):.4f}")
        print(f"{wilcoxon(all_rubies[i], all_rubies[j])}")
        print("=" * 20)
