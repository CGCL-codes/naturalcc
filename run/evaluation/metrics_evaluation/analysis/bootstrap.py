import random

from tqdm import tqdm

from metrics_evaluation.metrics.sacrebleu_code.sacrebleu_methods import corpus_bleu
from numpy import mean
import json
from collections import Counter

# Import CrystalBLEU
from metrics_evaluation.metrics.sacrebleu_code.sacrebleu_methods.compat import corpus_bleu as crb
from pygments.lexers.python import PythonLexer
from metrics_evaluation.metrics import tokenize_tranx as tknz
import re
from nltk.util import ngrams


def synthesize_model(
    model_dictionary, base_model, other_models, change_percentage, metrics, improve=True
):
    model_size = len(model_dictionary)
    change_lump = round(change_percentage * model_size / 100)
    new_model = base_model + "_" + str(change_percentage) + "_" + str(int(improve))
    changes_list = []
    for i in range(model_size):
        changed = False
        change = (0, i, "model")
        for model in other_models:
            diff = (
                model_dictionary[i]["grade-" + model]
                - model_dictionary[i]["grade-" + base_model]
            )
            improved_when_needed = (improve is True) and (diff > 0)
            worsened_when_needed = (improve is False) and (diff < 0)
            if improved_when_needed or worsened_when_needed:
                if change[0] < abs(diff):
                    change = (abs(diff), i, model)
                changed = True
        if changed is True:
            changes_list.append(change)
    changes_list.sort(key=lambda tup: tup[0], reverse=True)
    changes_list = changes_list[:change_lump]
    change_lump -= len(changes_list)
    for item in changes_list:
        i = item[1]
        model = item[2]
        model_dictionary[i]["grade-" + new_model] = model_dictionary[i][
            "grade-" + model
        ]
        model_dictionary[i][new_model] = model_dictionary[i][model]
        for metric in metrics:
            model_dictionary[i][metric + "-" + new_model] = model_dictionary[i][
                metric + "-" + model
            ]

    for item in model_dictionary:
        item["grade-" + new_model] = item.get(
            "grade-" + new_model, item["grade-" + base_model]
        )
        item[new_model] = item.get(new_model, item[base_model])
        for metric in metrics:
            item[metric + "-" + new_model] = item.get(
                metric + "-" + new_model, item[metric + "-" + base_model]
            )
    if change_lump > 0:
        print(
            f"Failed to generate enough changed snippets. {change_lump} new snippets are lacking."
        )


def bootstrapped_bleu(model_dictionary, model_name, bootstrap_list):
    hyps = [model_dictionary[i][model_name] for i in bootstrap_list]
    all_snippets = [model_dictionary[i]["snippet"] for i in bootstrap_list]

    is_list = isinstance(all_snippets[0], list)
    if is_list:
        max_refs = max(len(snippets) for snippets in all_snippets)
        refs = [[] for _ in range(max_refs)]
        for snippets in all_snippets:
            for i in range(max_refs):
                if i < len(snippets):
                    refs[i].append(snippets[i])
                else:
                    refs[i].append(snippets[-1])
    else:
        refs = [[]]
        for snippet in all_snippets:
            refs[0].append(snippet)

    value = corpus_bleu(hyps, refs, tokenize="code").score / 100
    return value


def bootstrapped_crb(model_dictionary, model_name, bootstrap_list):
    allrefs = []
    for problem in model_dictionary:
        allrefs.append(problem["snippet"])
    content = [item for sublist in allrefs for item in sublist]
    k = 100
    lexer = PythonLexer()
    tokens = []
    for j in content:
        tokens.extend(
            [
                i
                for i in list(map(lambda x: x[1], lexer.get_tokens(j)))
                if not (
                    re.fullmatch("\s+", i)
                    or re.fullmatch("#.*\n", i)
                    or re.match('""".*"""', i, re.DOTALL)
                )
            ]
        )
    pl_counts = []
    for i in range(1, 5):
        pl_counts.append(dict(Counter(ngrams(tokens, i)).most_common(k)))
    trivially_shared_ngrams = {k: v for d in pl_counts for k, v in d.items()}
    hyp = []
    ref = []
    for i in bootstrap_list:
        hyp.append(tknz(model_dictionary[i][model_name]))
        eref = []
        for item in model_dictionary[i]["snippet"]:
            eref.append(tknz(item))
        ref.append(eref)
    return crb(hyp, ref, ignoring=trivially_shared_ngrams)


def bootstrapped_metric(model_dictionary, model_name, metric, bootstrap_indices):
    scores = []
    grade_name = metric + "-" + model_name
    for i in bootstrap_indices:
        scores.append(model_dictionary[i][grade_name])
    return mean(scores)


def compare_models(model1, model2):
    m1 = 0
    for item1, item2 in zip(model1, model2):
        if item1 > item2:
            m1 += 1
    return m1 / len(model1)


def compare_for_significance(grade_significance, metric_significance):
    if ((0.95 > grade_significance > 0.05) and (metric_significance <= 0.05)) or (
        (0.95 > grade_significance > 0.05) and (metric_significance >= 0.95)
    ):
        return 1
    elif ((0.05 >= grade_significance) and (0.95 > metric_significance > 0.05)) or (
        (0.95 <= grade_significance) and (0.95 > metric_significance > 0.05)
    ):
        return 2
    elif ((0.05 >= grade_significance) and (0.95 <= metric_significance)) or (
        (0.95 <= grade_significance) and (0.05 >= metric_significance)
    ):
        return -1
    else:
        return 0


def metric_bin_score(metric_significance, model1_score, model2_score):
    if 0.95 > metric_significance > 0.05:
        return "NS"
    else:
        return abs(model2_score - model1_score)


def split_into_bins(model_pairs, model_scores, metrics, models):
    splitting = dict()
    for metric in metrics:
        splitting[metric] = []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i > j:
                    model_comparison = []
                    metric_significance = model_pairs[model1][model2][metric]
                    grade_significance = model_pairs[model1][model2]["grade"]
                    model1_score = model_scores[model1][metric]
                    model2_score = model_scores[model2][metric]
                    model_comparison.append(
                        metric_bin_score(
                            metric_significance, model1_score, model2_score
                        )
                    )
                    model_comparison.append(
                        compare_for_significance(
                            grade_significance, metric_significance
                        )
                    )
                    splitting[metric].append(model_comparison)
    return splitting


def bootstrap(model_dictionary, models, metrics, p_value=0.05, bootstrap_sampling=500):
    model_size = len(model_dictionary)
    model_scores = dict()
    model_pairs = dict()
    bootstrap_results = dict()
    for model in models:
        model_scores[model] = dict()
        bootstrap_results[model] = dict()
        for metric in metrics:
            bootstrap_results[model][metric] = []
    for _ in tqdm(range(bootstrap_sampling)):
        bootstrap_list = random.choices([m for m in range(model_size)], k=model_size)
        for m, model in enumerate(models):
            for metric in metrics:
                if metric == "bleu":
                    bootstrap_results[model][metric].append(
                        bootstrapped_bleu(model_dictionary, model, bootstrap_list)
                    )
                else:
                    bootstrap_results[model][metric].append(
                        bootstrapped_metric(
                            model_dictionary, model, metric, bootstrap_list
                        )
                    )
    with open("data/tmp.json", "w") as f:
        json.dump(bootstrap_results, f)
    for model1 in models:
        model_pairs[model1] = dict()
        for model2 in models:
            if model1 != model2:
                model_pairs[model1][model2] = dict()
                for metric in metrics:
                    model_pairs[model1][model2][metric] = compare_models(
                        bootstrap_results[model1][metric],
                        bootstrap_results[model2][metric],
                    )

    for model in models:
        for metric in metrics:
            bootstrap_results[model][metric].sort()
            model_scores[model][metric + "-low"] = bootstrap_results[model][metric][
                round(p_value * bootstrap_sampling / 2)
            ]
            model_scores[model][metric + "-high"] = bootstrap_results[model][
                metric
            ][-round(p_value * bootstrap_sampling / 2)]
            model_scores[model][metric] = bootstrap_results[model][metric][
                round(bootstrap_sampling / 2)
            ]

    return model_pairs, model_scores


def kendall_tau_metric(model_dictionary, models, metric):
    concordant_pairs = 0
    discordant_pairs = 0
    for model in models:
        for i, snippet1 in enumerate(model_dictionary[model]):
            for j, snippet2 in enumerate(model_dictionary[model]):
                if i > j:
                    if snippet1["gold-" + model] == snippet2["gold-" + model]:
                        continue
                    elif (snippet1["gold-" + model] > snippet2["gold-" + model]) and (
                        snippet1["grade-" + model + "-" + metric]
                        > snippet2["grade-" + model + "-" + metric]
                    ):
                        concordant_pairs += 1
                    elif (snippet1["gold-" + model] < snippet2["gold-" + model]) and (
                        snippet1["grade-" + model + "-" + metric]
                        < snippet2["grade-" + model + "-" + metric]
                    ):
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1
    kendall_tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs
    )
    return kendall_tau


def kendall_tau_people(
    model_dictionary, models, grader1, grader2
):  # how should we properly write that?
    concordant_pairs = 0
    discordant_pairs = 0
    for model in models:
        grader_1 = grader1 + "-" + model
        grader_2 = grader2 + "-" + model
        for i, snippet1 in enumerate(model_dictionary[model]):
            for j, snippet2 in enumerate(model_dictionary[model]):
                if i > j:
                    if (
                        (snippet1.get(grader_1) is None)
                        or (snippet2.get(grader_1) is None)
                        or (snippet1.get(grader_1) is None)
                        or (snippet2.get(grader_1) is None)
                    ):
                        continue
                    if (snippet1[grader_2] == snippet2[grader_2]) and (
                        snippet1[grader_1] == snippet2[grader_1]
                    ):
                        concordant_pairs += 1
                    elif (snippet1[grader_2] > snippet2[grader_2]) and (
                        snippet1[grader_1] > snippet2[grader_1]
                    ):
                        concordant_pairs += 1
                    elif (snippet1[grader_2] < snippet2[grader_2]) and (
                        snippet1[grader_1] < snippet2[grader_1]
                    ):
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1
    kendall_tau = (concordant_pairs - discordant_pairs) / (
        concordant_pairs + discordant_pairs
    )
    return kendall_tau


def main():
    model_dictionary = json.load(open("../../data/to-grade/all-singles.json"))
    lst = [i for i in range(len(model_dictionary) - 225)]
    a = bootstrapped_bleu(model_dictionary, "tranx-annot", lst)
    print(a)


if __name__ == "__main__":
    main()
