import editdistance
import numpy as np
from .utils import load_jsonl
from nltk.tokenize import RegexpTokenizer
from typing import FrozenSet
import keyword
import re
import argparse
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
code_tokenizer = RegexpTokenizer(r'\w+')
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def compute_EM(target, prediction, language="python"):
    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    if target_lines_str == prediction_lines_str:
        return 1
    else:
        return 0


def compute_ES(target, prediction, language="python"):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]

    target_str = ''.join(target_lines)
    prediction_str = ''.join(prediction_lines)
    ES_score = 1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))

    return ES_score


def hit(search_cases, hits=None):
    if hits is None:
        hits = [1, 5, 10]
    hit_res = [0.0 for _ in range(0, len(hits))]
    for case in search_cases:
        target_lines = [line.strip() for line in case['groundtruth'].splitlines() if line.strip()]
        target_lines = [line for line in target_lines if not line.startswith('#')]
        target_line = "".join(target_lines)
        hit_pos = np.inf
        for i in range(1, len(case['top_k_context'])+1):
            prediction_lines = [line.strip() for line in case['top_k_context'][-i][0].splitlines() if line.strip()]
            prediction_lines = [line for line in prediction_lines if not line.startswith('#')]
            prediction_line = "".join(prediction_lines)
            if target_line in prediction_line:
                hit_pos = i
                break
        for i in range(0, len(hits)):
            if hits[i] >= hit_pos:
                hit_res[i] += 1.0

    for i in range(0, len(hit_res)):
        hit_res[i] /= len(search_cases)
    return hit_res


def compute_batch_EM(file_path, language="python"):
    data = load_jsonl(file_path)
    em_val = 0
    for case in data:
        pred_str = case['generate_response']
        gt_str = case['groundtruth']
        em_val += compute_EM(gt_str, pred_str, language=language)
    return em_val / len(data)


def compute_batch_ES(file_path, language="python"):
    data = load_jsonl(file_path)
    es_val = 0
    for case in data:
        pred_str = case['generate_response']
        gt_str = case['groundtruth']
        es_val += compute_ES(gt_str, pred_str, language=language)
    return es_val / len(data)


def get_language_keywords() -> FrozenSet[str]:
    return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')


def is_identifier(token, language="python"):
    return True if IDENTIFIER_REGEX.match(token) \
                   and (language is None or token not in get_language_keywords()) else False


def extract_identifiers(source_code, language="python"):
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, language=language)]
    return _ids


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def compute_identifier_match(prediction, target, language="python"):

    comment_prefix = ""
    if language == "python":
        comment_prefix = "#"
    elif language == "java":
        comment_prefix = "//"

    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][:len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)

    pred_ids = extract_identifiers(prediction_lines_str, language=language)
    gt_ids = extract_identifiers(target_lines_str, language=language)
    identifier_em = int(pred_ids == gt_ids)
    id_tp, id_fp, id_fn = compute_id_match(pred_ids, gt_ids)
    id_f1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0
    return identifier_em, id_f1


def compute_batch_identifier_match(file_path, language="python"):
    data = load_jsonl(file_path)
    em_val = 0
    f1_val = 0
    for case in data:
        pred_str = case['generate_response']
        gt_str = case['groundtruth']
        em, f1 = compute_identifier_match(pred_str, gt_str, language=language)
        em_val += em
        f1_val += f1
    return em_val / len(data), f1_val / len(data)


def main():
    parser = argparse.ArgumentParser(description="Metrics")
    parser.add_argument('--file_path', type=str, required=True, help="包含 ground truth 和 predictions 的文件路径")
    parser.add_argument('--metric', type=str, required=True, choices=['EM', 'ES', 'IM'], help="要计算的指标")
    parser.add_argument('--language', type=str, default='python', help="编程语言，默认为 python")
    args = parser.parse_args()

    if args.metric == 'EM':
        em_score = compute_batch_EM(args.file_path, language=args.language)
        print(f"Exact Match (EM) Score: {em_score:.4f}")
    elif args.metric == 'ES':
        es_score = compute_batch_ES(args.file_path, language=args.language)
        print(f"Edit Similarity (ES) Score: {es_score:.4f}")
    elif args.metric == 'IM':
        em_score, f1_score = compute_batch_identifier_match(args.file_path, language=args.language)
        print(f"Identifier Match Exact Match (EM) Score: {em_score:.4f}")
        print(f"Identifier Match F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    main()