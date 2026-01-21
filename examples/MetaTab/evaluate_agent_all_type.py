from collections import Counter
from utils.eval import eval_ex_match, extract_answer
import random
import json
import numpy as np
from tqdm import tqdm
from fire import Fire
from typing import Union, List, Tuple
import re

def flatten(lst):
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list

def load_results(checkpoints, elements_per_checkpoint):

    print(f"Loading {checkpoints}...")
    # not a list or a tuple, make it a list
    if not isinstance(checkpoints, list) and not isinstance(checkpoints, tuple):
        # try to split by comma
        if "," in checkpoints:
            checkpoints = checkpoints.split(",")
            # remove the spaces
            checkpoints = [checkpoint.strip() for checkpoint in checkpoints]
        else:
            checkpoints = [checkpoints]

    all_results = []

    # read all checkpoints
    for checkpoint in checkpoints:
        print(f"Loading {checkpoint}...")

        if checkpoint.endswith(".jsonl"):
            with open(checkpoint, "r") as f:
                results = [json.loads(line) for line in f.readlines()]
        else:
            with open(f"output/{checkpoint}/result.jsonl", "r") as f:
                results = [json.loads(line) for line in f.readlines()]

        print(f"Loaded {len(results)} results.")


        # deduplicate the results by id
        results = {result["question_id"]: result for result in results}
        results = list(results.values())

        all_results.append(results)

    # make sure the checkpoints are same length, if not, cut the longer one
    min_len = min([len(results) for results in all_results])
    all_results = [results[:min_len] for results in all_results]

    # the results are now in the form of [[dict, dict, ...], [dict, dict, ...], ...]
    # we want to combine them into one list of dicts by aggregating the dict["text"] field
    combined_results = []
    for i, results in enumerate(all_results):
        if i == 0:
            # if this is the first checkpoint, just add the results
            combined_results = results
            # make the text field a list of list
            for result in combined_results:
                # random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                result["text"] = random.sample(result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else [result["text"]]

        else:
            # if this is not the first checkpoint, add the text field to the existing list
            for j, result in enumerate(results):
                # remember to random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                temp = random.sample(result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else result["text"]

                # add by question id instead of index
                for k, combined_result in enumerate(combined_results):
                    if combined_result["question_id"] == result["question_id"]:
                        combined_results[k]["text"].append(temp)
                        break

    # now we have a list of dicts with the text field being a list of list
    return combined_results


from typing import List, Dict
import re

def classify_question(question_text: str, table_columns: List[str] = None) -> List[str]:
    """返回问题所属的所有SQL操作类别"""
    question = question_text.lower()
    categories = set()

    # 检测聚合函数（COUNT/SUM/AVG等）
    # 改进后的聚合检测（覆盖更多关键词）
    aggregation_keywords = [
        r"\b(count\(|sum\(|avg\(|average\(|max\(|min\()",  # 函数形式
        r"\b(total\b|how many|number of|average of|sum of)",  # 自然语言
        r"\b(most|least)\b.*\b(amount|quantity)\b"  # 隐含聚合
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
        if table_columns:  # 如果有列名，进一步验证列名+条件组合
            for col in table_columns:
                col = col.lower()
                if (col in question) and re.search(condition_keywords, question):
                    categories.add("WHERE")
                    break
        else:  # 无列名时直接匹配条件关键词
            categories.add("WHERE")

    # 默认类别（简单查询）
    if not categories:
        categories.add("SELECT")



    return sorted(categories)  # 返回排序后的类别列表

# 示例调用
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def eval_wtq_with_multilabel(checkpoints: Union[List, Tuple, str],
                             elements_per_checkpoint: Union[None, int, List] = None,
                             n_times: int = 100,
                             sub_sample_question_ids: list = None):
    # 加载数据
    results = load_results(checkpoints, elements_per_checkpoint)

    # 初始化统计字典
    sql_categories = ["SELECT", "WHERE", "GROUP_BY", "ORDER_BY", "AGGREGATION", "MULTI_OP"]
    category_stats = {cat: {"correct": 0, "total": 0} for cat in sql_categories}
    acc_list = []  # 存储每轮的总体准确率

    for _ in tqdm(range(n_times), desc="Evaluating"):
        batch_correct, batch_total = 0, 0

        for result in results:
            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                continue

            # 多标签分类
            table_columns = result.get("table_columns", [])
            categories = classify_question(result["question"], table_columns)

            # 计算当前问题是否正确
            answer = ", ".join(result["answer"])
            preds = flatten([result["text"]]) if isinstance(result["text"], str) else flatten(result["text"])
            preds = [extract_answer(pred) for pred in preds if pred]

            is_correct = False

            if preds:
                final_pred, _ = Counter(preds).most_common(1)[0]
                is_correct = eval_ex_match(answer, final_pred)
                batch_correct += int(is_correct)
            batch_total += 1
            """
            is_correct = False

            if preds:  # 有预测结果
                final_pred, _ = Counter(preds).most_common(1)[0]
                is_correct = eval_ex_match(answer, final_pred)
            else:  # 没有预测结果，算错
                is_correct = False
            batch_correct += int(is_correct)
            batch_total += 1
            """
            # 更新多标签统计
            for cat in categories:
                category_stats[cat]["total"] += 1
                if is_correct:
                    category_stats[cat]["correct"] += 1
            if len(categories) > 1:
                category_stats["MULTI_OP"]["total"] += 1
                if is_correct:
                    category_stats["MULTI_OP"]["correct"] += 1

        acc_list.append(batch_correct / batch_total * 100 if batch_total > 0 else 0)


    # 打印全局统计（保留原有输出）
    print("\n📊 ========== 全局统计 ==========")
    print(f"总样本数: {len(results)}")
    print(f"最小准确率: {min(acc_list):.2f}%")
    print(f"最大准确率: {max(acc_list):.2f}%")
    print(f"平均准确率: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")


    # 打印多标签分类结果
    print("\n🔍 ========== 多标签分类统计 ==========")
    for category in sql_categories:
        stats = category_stats[category]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            print(f"{category.ljust(10)}: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"{category.ljust(10)}: 无样本")

    # 多操作问题占比
    multi_op_ratio = category_stats["MULTI_OP"]["total"] / len(results) * 100
    print(f"\n🔹 多操作问题占比: {multi_op_ratio:.1f}%")

if __name__ == "__main__":
    #Fire(eval_wtq)
    #eval_wtq(checkpoints ="./assets/results/wtq-cot-all/result_5.jsonl",n_times=100)
    eval_wtq_with_multilabel(checkpoints="./output/wtq_agent/result.jsonl", n_times=1
             )