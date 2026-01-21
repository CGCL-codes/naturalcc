from collections import Counter
from utils.eval import eval_ex_match, extract_answer
import random
import json
import numpy as np
from tqdm import tqdm
from fire import Fire
from typing import Union, List, Tuple, Dict
import re


def flatten(lst):
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list


def load_single_results(file_path: str):
    """加载单个结果文件"""
    print(f"Loading {file_path}...")

    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            results = [json.loads(line) for line in f.readlines()]
    else:
        with open(f"output/{file_path}/result.jsonl", "r") as f:
            results = [json.loads(line) for line in f.readlines()]

    print(f"Loaded {len(results)} results.")

    # 去重
    results = {result["question_id"]: result for result in results}
    return list(results.values())


def load_dual_results(original_path: str, metamorphic_path: str):
    """加载原始和蜕变两个结果文件"""
    original_results = load_single_results(original_path)
    metamorphic_results = load_single_results(metamorphic_path)

    # 确保两个结果集基于question_id对齐
    orig_dict = {r["question_id"]: r for r in original_results}
    meta_dict = {r["question_id"]: r for r in metamorphic_results}

    # 只保留两个文件都有的question_id
    common_ids = set(orig_dict.keys()) & set(meta_dict.keys())

    aligned_results = []
    for qid in common_ids:
        aligned_results.append({
            'question_id': qid,
            'original': orig_dict[qid],
            'metamorphic': meta_dict[qid]
        })

    print(f"Aligned {len(aligned_results)} common results.")
    return aligned_results


def classify_question(question_text: str, table_columns: List[str] = None) -> List[str]:
    """返回问题所属的所有SQL操作类别"""
    question = question_text.lower()
    categories = set()

    # 检测聚合函数（COUNT/SUM/AVG等）
    aggregation_keywords = [
        r"\b(count\(|sum\(|avg\(|average\(|max\(|min\()",
        r"\b(total\b|how many|number of|average of|sum of)",
        r"\b(most|least)\b.*\b(amount|quantity)\b"
    ]
    if any(re.search(pattern, question) for pattern in aggregation_keywords):
        categories.add("AGGREGATION")

    # 检测排序（ORDER BY）
    if re.search(r"\b(order by|sort by|highest|lowest|top|bottom|ascending|descending)", question):
        categories.add("ORDER_BY")

    # 检测分组（GROUP BY）
    if re.search(r"\b(group by|per|by each|for each)", question):
        categories.add("GROUP_BY")

    # 检测条件过滤（WHERE）
    condition_keywords = r"(>|<|=|!=|>=|<=|where|and|or|not in|excluding)"
    if re.search(condition_keywords, question):
        if table_columns:
            for col in table_columns:
                col = col.lower()
                if (col in question) and re.search(condition_keywords, question):
                    categories.add("WHERE")
                    break
        else:
            categories.add("WHERE")

    # 默认类别（简单查询）
    if not categories:
        categories.add("SELECT")

    return sorted(categories)


def eval_metamorphic_wtq(original_checkpoint: str,
                         metamorphic_checkpoint: str,
                         n_times: int = 100,
                         sub_sample_question_ids: list = None) -> Dict[str, float]:
    """
    评估WTQ数据集的蜕变测试性能

    Args:
        original_checkpoint: 原始结果文件路径
        metamorphic_checkpoint: 蜕变结果文件路径
        n_times: 重复评估次数
        sub_sample_question_ids: 子采样问题ID列表

    Returns:
        包含Precision, Recall, F1等指标的字典
    """
    # 加载对齐的结果
    results = load_dual_results(original_checkpoint, metamorphic_checkpoint)

    if sub_sample_question_ids:
        results = [r for r in results if r['question_id'] in sub_sample_question_ids]

    # 初始化统计
    sql_categories = ["SELECT", "WHERE", "GROUP_BY", "ORDER_BY", "AGGREGATION", "MULTI_OP"]
    category_metrics = {cat: {"precision": [], "recall": [], "f1": []} for cat in sql_categories}

    overall_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    }

    for _ in tqdm(range(n_times), desc="Evaluating Metamorphic Testing"):
        tp = fp = fn = tn = 0
        category_tp = {cat: 0 for cat in sql_categories}
        category_fp = {cat: 0 for cat in sql_categories}
        category_fn = {cat: 0 for cat in sql_categories}
        category_tn = {cat: 0 for cat in sql_categories}

        for result in results:
            orig_data = result['original']

            meta_data = result['metamorphic']

            # 获取真实答案
            true_answer = ", ".join(orig_data["answer"]) if isinstance(orig_data["answer"], list) else orig_data[
                "answer"]

            # 提取预测答案
            orig_preds = flatten([orig_data["text"]]) if isinstance(orig_data["text"], str) else flatten(
                orig_data["text"])
            meta_preds = flatten([meta_data["text"]]) if isinstance(meta_data["text"], str) else flatten(
                meta_data["text"])


            orig_preds = [extract_answer(pred) for pred in orig_preds if pred]
            meta_preds = [extract_answer(pred) for pred in meta_preds if pred]

            if not orig_preds or not meta_preds:
                continue

            # 多数投票
            orig_final_pred, _ = Counter(orig_preds).most_common(1)[0]
            meta_final_pred, _ = Counter(meta_preds).most_common(1)[0]

            # 判断原始答案是否正确
            orig_correct = eval_ex_match(true_answer, orig_final_pred)

            # 检测不一致性
            inconsistency = not eval_ex_match(orig_final_pred, meta_final_pred)

            # 更新混淆矩阵
            if not orig_correct:  # 原始答案有幻觉
                if inconsistency:
                    tp += 1
                else:
                    fn += 1
            else:  # 原始答案正确
                if inconsistency:
                    fp += 1
                else:
                    tn += 1

            # 按类别统计
            table_columns = orig_data.get("table_columns", [])
            categories = classify_question(orig_data["question"], table_columns)

            for cat in categories:
                if not orig_correct:
                    if inconsistency:
                        category_tp[cat] += 1
                    else:
                        category_fn[cat] += 1
                else:
                    if inconsistency:
                        category_fp[cat] += 1
                    else:
                        category_tn[cat] += 1

            if len(categories) > 1:
                if not orig_correct:
                    if inconsistency:
                        category_tp["MULTI_OP"] += 1
                    else:
                        category_fn["MULTI_OP"] += 1
                else:
                    if inconsistency:
                        category_fp["MULTI_OP"] += 1
                    else:
                        category_tn["MULTI_OP"] += 1

        # 计算总体指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        overall_metrics["precision"].append(precision)
        overall_metrics["recall"].append(recall)
        overall_metrics["f1"].append(f1)

        # 计算类别指标
        for cat in sql_categories:
            cat_tp, cat_fp, cat_fn = category_tp[cat], category_fp[cat], category_fn[cat]
            cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
            cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (
                                                                                                  cat_precision + cat_recall) > 0 else 0

            category_metrics[cat]["precision"].append(cat_precision)
            category_metrics[cat]["recall"].append(cat_recall)
            category_metrics[cat]["f1"].append(cat_f1)

    # 计算平均指标
    final_metrics = {
        "overall": {
            "precision": np.mean(overall_metrics["precision"]) * 100,
            "recall": np.mean(overall_metrics["recall"]) * 100,
            "f1": np.mean(overall_metrics["f1"]) * 100,
            "precision_std": np.std(overall_metrics["precision"]) * 100,
            "recall_std": np.std(overall_metrics["recall"]) * 100,
            "f1_std": np.std(overall_metrics["f1"]) * 100,
        },
        "by_category": {}
    }

    # 添加类别指标
    for cat in sql_categories:
        if len(category_metrics[cat]["precision"]) > 0:
            final_metrics["by_category"][cat] = {
                "precision": np.mean(category_metrics[cat]["precision"]) * 100,
                "recall": np.mean(category_metrics[cat]["recall"]) * 100,
                "f1": np.mean(category_metrics[cat]["f1"]) * 100,
                "samples": category_tp[cat] + category_fn[cat]  # 该类别的幻觉样本数
            }

    # 打印结果
    print("\n📊 ========== 蜕变测试评估结果 ==========")
    print(f"总样本数: {len(results)}")
    print(f"Precision: {final_metrics['overall']['precision']:.2f}% ± {final_metrics['overall']['precision_std']:.2f}%")
    print(f"Recall:    {final_metrics['overall']['recall']:.2f}% ± {final_metrics['overall']['recall_std']:.2f}%")
    print(f"F1 Score:  {final_metrics['overall']['f1']:.2f}% ± {final_metrics['overall']['f1_std']:.2f}%")
    """
    print(f"\n🔍 混淆矩阵 (最后一次运行):")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")

    print(f"\n📈 按类别统计:")
    for cat, metrics in final_metrics["by_category"].items():
        if metrics["samples"] > 0:
            print(
                f"{cat.ljust(10)}: P={metrics['precision']:.1f}%, R={metrics['recall']:.1f}%, F1={metrics['f1']:.1f}% ({metrics['samples']} samples)")
    """
    return final_metrics


from collections import Counter
from utils.eval import eval_ex_match
import random
import json
import numpy as np
from tqdm import tqdm
from fire import Fire
from typing import Union, List, Tuple, Dict
import re


def classify_question(question_text: str, table_columns: List[str] = None) -> List[str]:
    """返回问题所属的所有SQL操作类别"""
    question = question_text.lower()
    categories = set()

    # 检测聚合函数（COUNT/SUM/AVG等）
    aggregation_keywords = [
        r"\b(count\(|sum\(|avg\(|average\(|max\(|min\()",
        r"\b(total\b|how many|number of|average of|sum of)",
        r"\b(most|least)\b.*\b(amount|quantity)\b"
    ]
    if any(re.search(pattern, question) for pattern in aggregation_keywords):
        categories.add("AGGREGATION")

    # 检测排序（ORDER BY）
    if re.search(r"\b(order by|sort by|highest|lowest|top|bottom|ascending|descending)", question):
        categories.add("ORDER_BY")

    # 检测分组（GROUP BY）
    if re.search(r"\b(group by|per|by each|for each)", question):
        categories.add("GROUP_BY")

    # 检测条件过滤（WHERE）
    condition_keywords = r"(>|<|=|!=|>=|<=|where|and|or|not in|excluding)"
    if re.search(condition_keywords, question):
        if table_columns:
            for col in table_columns:
                col = col.lower()
                if (col in question) and re.search(condition_keywords, question):
                    categories.add("WHERE")
                    break
        else:
            categories.add("WHERE")

    # 默认类别（简单查询）
    if not categories:
        categories.add("SELECT")

    return sorted(categories)


def extract_answer_cut(
        text: str,
        patterns: list = [r"Final Answer: (.*)", r": (.*)", r"is (.*)"],
        return_match_flag: bool = False,
        require_numeric: bool = True
):
    """
    Extracts the answer from a response text.
    """
    answer = None
    match_flag = False

    for pattern in reversed(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            if require_numeric and not candidate.isdigit():
                continue
            answer = candidate
            match_flag = "final answer" in pattern.lower()
            break

    if return_match_flag:
        return answer, match_flag
    return answer


def flatten(lst):
    """展平嵌套列表"""
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list








def flatten(lst):
    """展平嵌套列表"""
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list


def load_cut_results(checkpoint_path: str, elements_per_checkpoint: int = None):
    """加载cut版本的结果文件"""
    print(f"Loading cut results from {checkpoint_path}...")

    if checkpoint_path.endswith(".jsonl"):
        with open(checkpoint_path, "r") as f:
            results = [json.loads(line) for line in f.readlines()]
    else:
        with open(f"output/{checkpoint_path}/result.jsonl", "r") as f:
            results = [json.loads(line) for line in f.readlines()]

    print(f"Loaded {len(results)} results.")

    # 去重
    results = {result["question_id"]: result for result in results}
    results = list(results.values())

    # 处理text_part1和text_part2字段
    for result in results:
        if isinstance(result.get("text_part1"), str):
            result["text_part1"] = [result["text_part1"]]
        if isinstance(result.get("text_part2"), str):
            result["text_part2"] = [result["text_part2"]]

        # 随机采样
        if elements_per_checkpoint is not None:
            if "text_part1" in result and result["text_part1"]:
                result["text_part1"] = random.sample(result["text_part1"],
                                                     min(elements_per_checkpoint, len(result["text_part1"])))
            if "text_part2" in result and result["text_part2"]:
                result["text_part2"] = random.sample(result["text_part2"],
                                                     min(elements_per_checkpoint, len(result["text_part2"])))

    return results


def process_cut_predictions(result: Dict, separators: List[str] = ["Final answer: "]):
    """处理cut版本的预测结果"""
    # 展平text_part1和text_part2
    if "text_part1" in result:
        result["text_part1"] = flatten(result["text_part1"])
    if "text_part2" in result:
        result["text_part2"] = flatten(result["text_part2"])

    # 提取答案
    preds1 = [extract_answer_cut(text) for text in result.get("text_part1", [])]
    preds2 = [extract_answer_cut(text) for text in result.get("text_part2", [])]

    # 替换None为0
    preds1 = [0 if pred is None else pred for pred in preds1]
    preds2 = [0 if pred is None else pred for pred in preds2]

    # 合并预测结果
    preds = preds1 + preds2
    preds = [pred for pred in preds if pred]

    if not preds:
        return None

    # 处理分隔符
    used_separator = None
    for sep in separators:
        if sep in str(preds[0]):
            used_separator = sep
            break

    if used_separator:
        processed_pred = str(preds[0]).replace(used_separator, "|")
        pred_list = [item.strip() for item in processed_pred.split("|") if item.strip()]
    else:
        pred_list = [str(pred) for pred in preds]

    # 多数投票
    pred_count = Counter(pred_list)
    try:
        final_pred, _ = pred_count.most_common(1)[0]
        return final_pred
    except:
        return None


def eval_metamorphic_wtq_cut(original_path: str,
                             metamorphic_path: str,
                             elements_per_checkpoint: int = None,
                             n_times: int = 100,
                             sub_sample_question_ids: list = None) -> Dict[str, float]:
    """
    评估cut版本的蜕变测试性能，计算Precision, Recall, F1 Score

    Args:
        original_path: 原始结果文件路径
        metamorphic_path: 蜕变结果文件路径
        elements_per_checkpoint: 每个检查点采样数量
        n_times: 重复评估次数
        sub_sample_question_ids: 子采样问题ID列表

    Returns:
        包含Precision, Recall, F1等指标的字典
    """
    print("🚀 Starting Metamorphic Testing for Cut Version...")

    # 加载对齐的结果
    orig_results = load_cut_results(original_path, elements_per_checkpoint)
    meta_results = load_cut_results(metamorphic_path, elements_per_checkpoint)

    # 创建映射
    orig_dict = {r["question_id"]: r for r in orig_results}
    meta_dict = {r["question_id"]: r for r in meta_results}


    # 只保留共同的问题ID
    common_ids = set(orig_dict.keys()) & set(meta_dict.keys())
    if sub_sample_question_ids:
        common_ids = common_ids & set(sub_sample_question_ids)

    print(f"Evaluating {len(common_ids)} common samples...")

    # 初始化统计
    sql_categories = ["SELECT", "WHERE", "GROUP_BY", "ORDER_BY", "AGGREGATION", "MULTI_OP"]

    overall_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": []
    }

    category_metrics = {cat: {"precision": [], "recall": [], "f1": []} for cat in sql_categories}

    for i in tqdm(range(n_times), desc="Metamorphic Evaluation", unit="batch"):
        # 初始化混淆矩阵
        tp = fp = fn = tn = 0
        category_cm = {cat: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cat in sql_categories}

        for qid in common_ids:
            orig_result = orig_dict[qid]
            meta_result = meta_dict[qid]

            # 获取真实答案
            true_answer = ", ".join(orig_result["answer"]) if isinstance(orig_result["answer"], list) else orig_result[
                "answer"]

            # 处理预测答案
            orig_pred = flatten([orig_result["text"]]) if isinstance(orig_result["text"], str) else flatten(
                orig_result["text"])
            orig_pred = [extract_answer_cut(pred) for pred in orig_pred if pred]
            orig_pred, _ = Counter(orig_pred).most_common(1)[0]


            meta_pred = process_cut_predictions(meta_result)


            if orig_pred is None or meta_pred is None:
                continue

            # 判断原始答案是否正确
            orig_correct = eval_ex_match(true_answer, orig_pred)

            # 检测不一致性
            inconsistency = not eval_ex_match(orig_pred, meta_pred)

            # 更新混淆矩阵
            if not orig_correct:  # 原始答案有幻觉
                if inconsistency:
                    tp += 1  # 正确检测到幻觉
                else:
                    fn += 1  # 漏报
            else:  # 原始答案正确
                if inconsistency:
                    fp += 1  # 误报
                else:
                    tn += 1  # 正确通过

            # 按类别统计
            table_columns = orig_result.get("table_columns", [])
            categories = classify_question(orig_result["question"], table_columns)

            for cat in categories:
                if not orig_correct:
                    if inconsistency:
                        category_cm[cat]["tp"] += 1
                    else:
                        category_cm[cat]["fn"] += 1
                else:
                    if inconsistency:
                        category_cm[cat]["fp"] += 1
                    else:
                        category_cm[cat]["tn"] += 1

            if len(categories) > 1:
                if not orig_correct:
                    if inconsistency:
                        category_cm["MULTI_OP"]["tp"] += 1
                    else:
                        category_cm["MULTI_OP"]["fn"] += 1
                else:
                    if inconsistency:
                        category_cm["MULTI_OP"]["fp"] += 1
                    else:
                        category_cm["MULTI_OP"]["tn"] += 1

        # 计算总体指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        overall_metrics["precision"].append(precision)
        overall_metrics["recall"].append(recall)
        overall_metrics["f1"].append(f1)
        overall_metrics["accuracy"].append(accuracy)

        # 计算类别指标
        for cat in sql_categories:
            cat_tp, cat_fp, cat_fn = category_cm[cat]["tp"], category_cm[cat]["fp"], category_cm[cat]["fn"]
            cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
            cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (
                                                                                                  cat_precision + cat_recall) > 0 else 0

            category_metrics[cat]["precision"].append(cat_precision)
            category_metrics[cat]["recall"].append(cat_recall)
            category_metrics[cat]["f1"].append(cat_f1)

    # 计算平均指标
    final_metrics = {
        "overall": {
            "precision": np.mean(overall_metrics["precision"]) * 100,
            "recall": np.mean(overall_metrics["recall"]) * 100,
            "f1": np.mean(overall_metrics["f1"]) * 100,
            "accuracy": np.mean(overall_metrics["accuracy"]) * 100,
            "precision_std": np.std(overall_metrics["precision"]) * 100,
            "recall_std": np.std(overall_metrics["recall"]) * 100,
            "f1_std": np.std(overall_metrics["f1"]) * 100,
            "accuracy_std": np.std(overall_metrics["accuracy"]) * 100,
        },
        "by_category": {},
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    }

    # 添加类别指标
    for cat in sql_categories:
        if len(category_metrics[cat]["precision"]) > 0:
            final_metrics["by_category"][cat] = {
                "precision": np.mean(category_metrics[cat]["precision"]) * 100,
                "recall": np.mean(category_metrics[cat]["recall"]) * 100,
                "f1": np.mean(category_metrics[cat]["f1"]) * 100,
                "samples": category_cm[cat]["tp"] + category_cm[cat]["fn"]  # 该类别的幻觉样本数
            }

    # 打印结果
    print("\n📊 ========== Cut Version Metamorphic Testing Results ==========")
    print(f"总样本数: {len(common_ids)}")
    print(f"Precision: {final_metrics['overall']['precision']:.2f}% ± {final_metrics['overall']['precision_std']:.2f}%")
    print(f"Recall:    {final_metrics['overall']['recall']:.2f}% ± {final_metrics['overall']['recall_std']:.2f}%")
    print(f"F1 Score:  {final_metrics['overall']['f1']:.2f}% ± {final_metrics['overall']['f1_std']:.2f}%")
    print(f"Accuracy:  {final_metrics['overall']['accuracy']:.2f}% ± {final_metrics['overall']['accuracy_std']:.2f}%")
    """
    print(f"\n🔍 Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")

    # 计算额外统计
    hallucination_rate = (tp + fn) / len(common_ids) * 100 if len(common_ids) > 0 else 0
    detection_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    print(f"\n📈 Additional Statistics:")
    print(f"Hallucination Rate: {hallucination_rate:.2f}%")
    print(f"Detection Rate: {detection_rate:.2f}%")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}%")

    print(f"\n🎯 Category-wise Results:")
    for cat, metrics in final_metrics["by_category"].items():
        if metrics["samples"] > 0:
            print(
                f"{cat.ljust(10)}: P={metrics['precision']:.1f}%, R={metrics['recall']:.1f}%, F1={metrics['f1']:.1f}% ({metrics['samples']} samples)")
    """
    return final_metrics



if __name__ == "__main__":
    # 使用示例



    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_agent_adv_test/wtq_agent/result.jsonl",
        n_times=100
    )
    exit(1)

    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_agent_row_shuffle/wtq_agent/result.jsonl",
        n_times=100
    )

    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_agent_column_shuffle/wtq_agent/result.jsonl",
        n_times=100
    )
    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_transpose/wtq_agent/result.jsonl",
        n_times=100
    )

    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_reconstruction/wtq_agent/result.jsonl",
        n_times=100
    )

    metrics = eval_metamorphic_wtq_cut(
        original_path="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_path="./output_tablegpt_agent_cut/wtq_agent/result.jsonl",
        elements_per_checkpoint=5,
        n_times=10
    )
    metrics = eval_metamorphic_wtq_cut(
        original_path="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_path="./output_tablegpt_column_cut/wtq_agent/result.jsonl",
        elements_per_checkpoint=5,
        n_times=10
    )
    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablgpt_Symbolization_pure_numbers_to_words/wtq_agent/result.jsonl",
        n_times=100
    )
    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_Category_Anonymization/wtq_agent/result.jsonl",
        n_times=100
    )
    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_time/wtq_agent/result.jsonl",
        n_times=100
    )
    exit(1)