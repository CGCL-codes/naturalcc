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

    # 排序确保顺序固定
    common_ids = sorted(set(orig_dict.keys()) & set(meta_dict.keys()))

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

import re
import re
import re
from typing import List, Union

import re
from typing import List, Union

def extract_action_input_code(data: List[Union[str, List[str]]]) -> List[str]:
    """
    从 ReAct 风格日志中提取所有 Action Input 后面的代码块。
    规则：
    - 以 Action Input 开始，到下一条 Observation 或 Thought/Action 开头之前结束
    - 去掉 ``` 和 python 标签
    """
    code_blocks = []

    # 匹配 Action Input 到 Observation/Thought/Action 的内容
    pattern = re.compile(
        r"Action Input:\s*(.*?)\s*(?=\n(?:Observation:|Thought:|Action:|$))",
        re.DOTALL
    )

    for item in data:
        if isinstance(item, str):
            matches = pattern.findall(item)
            for m in matches:
                # 去掉 ``` 和 python 标签
                cleaned = re.sub(r"```(?:python)?", "", m, flags=re.IGNORECASE).strip()
                if cleaned:
                    code_blocks.append(cleaned)
        elif isinstance(item, list):
            code_blocks.extend(s.strip() for s in item if isinstance(s, str) and s.strip())

    return code_blocks




import ast

class SymbolicState:
    def __init__(self):
        self.vars = {}

    def assign(self, var, expr):
        # 保存符号表达式
        self.vars[var] = expr
        #print(f"[DEBUG] Assign: {var} = {expr}")

    def get(self, var):
        return self.vars.get(var, var)


def clean_code_lines(code):
    """
    去掉注释、空行和 import 语句
    """
    if isinstance(code, str):
        lines = code.strip().split("\n")
    else:
        lines = code

    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("import") and not line.startswith("from"):
            cleaned.append(line)
    return cleaned


def ast_to_symbol(node, state):
    """
    将 AST 节点转换成符号表达式
    """
    if isinstance(node, ast.Name):
        return state.get(node.id)
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.BinOp):
        left = ast_to_symbol(node.left, state)
        right = ast_to_symbol(node.right, state)
        op = type(node.op).__name__
        return f"({left} {op} {right})"
    elif isinstance(node, ast.Compare):
        left = ast_to_symbol(node.left, state)
        comparators = [ast_to_symbol(c, state) for c in node.comparators]
        ops = [ast_to_symbol_op(op) for op in node.ops]
        comparisons = " ".join(f"{op} {c}" for op, c in zip(ops, comparators))
        return f"({left} {comparisons})"
    elif isinstance(node, ast.BoolOp):
        values = [ast_to_symbol(v, state) for v in node.values]
        op = type(node.op).__name__
        return f"({f' {op} '.join(values)})"
    elif isinstance(node, ast.UnaryOp):
        operand = ast_to_symbol(node.operand, state)
        op = type(node.op).__name__
        return f"({op} {operand})"
    elif isinstance(node, ast.Call):
        func = ast_to_symbol(node.func, state)
        args = [ast_to_symbol(a, state) for a in node.args]
        return f"{func}({', '.join(args)})"
    elif isinstance(node, ast.Attribute):
        value = ast_to_symbol(node.value, state)
        return f"{value}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value = ast_to_symbol(node.value, state)
        slice_ = ast_to_symbol(node.slice, state)
        return f"{value}[{slice_}]"
    elif isinstance(node, ast.Index):  # Python <3.9
        return ast_to_symbol(node.value, state)
    elif isinstance(node, ast.Slice):
        lower = ast_to_symbol(node.lower, state) if node.lower else ""
        upper = ast_to_symbol(node.upper, state) if node.upper else ""
        step = ast_to_symbol(node.step, state) if node.step else ""
        return f"{lower}:{upper}:{step}"
    elif isinstance(node, ast.Expr):
        return ast_to_symbol(node.value, state)
    elif isinstance(node, ast.Assign):
        target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else str(node.targets[0])
        value = ast_to_symbol(node.value, state)
        state.assign(target, value)
        return value
    elif isinstance(node, ast.Tuple):
        elts = [ast_to_symbol(e, state) for e in node.elts]
        return f"({', '.join(elts)})"
    elif isinstance(node, ast.List):
        elts = [ast_to_symbol(e, state) for e in node.elts]
        return f"[{', '.join(elts)}]"
    elif isinstance(node, ast.Dict):
        keys = [ast_to_symbol(k, state) for k in node.keys]
        values = [ast_to_symbol(v, state) for v in node.values]
        return f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, values))}}}"
    elif isinstance(node, ast.Lambda):
        args = [arg.arg for arg in node.args.args]
        body = ast_to_symbol(node.body, state)
        return f"lambda {', '.join(args)}: {body}"

    elif isinstance(node, ast.ListComp):
        elt = ast_to_symbol(node.elt, state)
        gens = []
        for g in node.generators:
            target = ast_to_symbol(g.target, state)
            iter_ = ast_to_symbol(g.iter, state)
            ifs = " ".join(f"if {ast_to_symbol(if_cond, state)}" for if_cond in g.ifs)
            gens.append(f"for {target} in {iter_} {ifs}".strip())
        return f"[{elt} {' '.join(gens)}]"

    elif isinstance(node, ast.GeneratorExp):
        elt = ast_to_symbol(node.elt, state)
        gens = []
        for g in node.generators:
            target = ast_to_symbol(g.target, state)
            iter_ = ast_to_symbol(g.iter, state)
            ifs = " ".join(f"if {ast_to_symbol(if_cond, state)}" for if_cond in g.ifs)
            gens.append(f"for {target} in {iter_} {ifs}".strip())
        return f"({elt} {' '.join(gens)})"

    elif isinstance(node, ast.IfExp):
        body = ast_to_symbol(node.body, state)
        test = ast_to_symbol(node.test, state)
        orelse = ast_to_symbol(node.orelse, state)
        return f"({body} if {test} else {orelse})"

    else:
        return ast.dump(node)


def ast_to_symbol_op(op):
    """
    将比较符号转换为字符串
    """
    if isinstance(op, ast.Eq): return "=="
    elif isinstance(op, ast.NotEq): return "!="
    elif isinstance(op, ast.Lt): return "<"
    elif isinstance(op, ast.LtE): return "<="
    elif isinstance(op, ast.Gt): return ">"
    elif isinstance(op, ast.GtE): return ">="
    elif isinstance(op, ast.In): return "in"
    elif isinstance(op, ast.NotIn): return "not in"
    else:
        return type(op).__name__



def symbolic_execute_ast(code_lines):
    """
    使用 AST 做符号执行
    """
    state = SymbolicState()
    last_expr = None

    for line in code_lines:
        try:
            tree = ast.parse(line)
            for node in tree.body:
                last_expr = ast_to_symbol(node, state)
        except Exception as e:
            last_expr = f"<Error: {e}>"

    return last_expr


def symbolic_logic_equivalence_ast(codes):
    """
    使用 AST 符号执行判断逻辑一致性
    """
    symbolic_results = []

    for i, code in enumerate(codes):
        #print(f"[DEBUG] Processing code {i+1}/{len(codes)}")
        cleaned_lines = clean_code_lines(code)
        sym_result = symbolic_execute_ast(cleaned_lines)
        symbolic_results.append(sym_result)
        #print(f"[DEBUG] Symbolic result: {sym_result}")

    all_equal = all(r == symbolic_results[0] for r in symbolic_results)
    #print(f"[DEBUG] All symbolic results equal? {all_equal}")
    return all_equal, symbolic_results


import re


def normalize_symbolic_states(symbolic_states):
    normalized = []

    for code in symbolic_states:
        code = code.strip()

        # 1️⃣ df[df['col']==value]['col2'].values[0]
        m1 = re.match(r"df\[\(df\['(.+?)'\]\s*==\s*'(.+?)'\)\]\['(.+?)'\](?:\.values\[0\])?", code)
        if m1:
            col, value, target_col = m1.groups()
            normalized.append(f"get_value(df, '{col}', '{value}', '{target_col}')")
            continue

        # 2️⃣ df.loc[df['col']==value, 'col2'].values[0]
        m2 = re.match(r"df\.loc\[\(df\['(.+?)'\]\s*==\s*'(.+?)'\),\s*'(.+?)'\](?:\.values\[0\])?", code)
        if m2:
            col, value, target_col = m2.groups()
            normalized.append(f"get_value(df, '{col}', '{value}', '{target_col}')")
            continue

        # 3️⃣ len(df[df['col']==value]) 或 df[df['col']==value].shape[0]
        m3 = re.match(r"(?:len|df\[\(df\['(.+?)'\]\s*==\s*'(.+?)'\)\]\.shape\[0\])", code)
        if m3 and m3.groups()[0] and m3.groups()[1]:
            col, value = m3.groups()
            normalized.append(f"count(df, '{col}', '{value}')")
            continue

        # 没匹配到规则，直接原样保留
        normalized.append(code)

    return normalized

def sort_unique(states):
    counter = Counter(states)
    sorted_states = sorted(counter.items(), key=lambda x: -x[1])
    return [s for s, _ in sorted_states]

import os
save_path = "metamorphic_place_holder.json"
def eval_metamorphic_wtq(original_checkpoint: str,
                         metamorphic_checkpoint: str,
                         n_times: int = 100,
                         sub_sample_question_ids: list = None,
                         save_path: str = "metamorphic_same_code.json") -> Dict[str, float]:
    """
    评估WTQ数据集的蜕变测试性能，并统计 same_code 情况
    TP 仅计入 same_codes 非空，same_codes 为空的算作 FN
    """
    results = load_dual_results(original_checkpoint, metamorphic_checkpoint)

    if sub_sample_question_ids:
        results = [r for r in results if r['question_id'] in sub_sample_question_ids]

    sql_categories = ["SELECT", "WHERE", "GROUP_BY", "ORDER_BY", "AGGREGATION", "MULTI_OP"]
    category_metrics = {cat: {"precision": [], "recall": [], "f1": []} for cat in sql_categories}

    overall_metrics = {"precision": [], "recall": [], "f1": []}

    # ===== NEW: 全局 same_code 计数器 =====
    total_eval_cases = 0
    same_code_cases = 0

    for _ in tqdm(range(n_times), desc="Evaluating Metamorphic Testing"):
        tp = fp = fn = tn = 0

        category_tp = {cat: 0 for cat in sql_categories}
        category_fp = {cat: 0 for cat in sql_categories}
        category_fn = {cat: 0 for cat in sql_categories}
        category_tn = {cat: 0 for cat in sql_categories}

        for result in results:
            orig_data = result['original']
            meta_data = result['metamorphic']

            orig_codes = extract_action_input_code(orig_data['text'])
            meta_codes = extract_action_input_code(meta_data['text'])

            _, orig_symbolic_states = symbolic_logic_equivalence_ast(orig_codes)
            _, meta_symbolic_states = symbolic_logic_equivalence_ast(meta_codes)

            true_answer = ", ".join(orig_data["answer"]) if isinstance(orig_data["answer"], list) else orig_data["answer"]

            orig_preds = flatten([orig_data["text"]]) if isinstance(orig_data["text"], str) else flatten(orig_data["text"])
            meta_preds = flatten([meta_data["text"]]) if isinstance(meta_data["text"], str) else flatten(meta_data["text"])

            orig_preds = [extract_answer(pred) for pred in orig_preds if pred]
            meta_preds = [extract_answer(pred) for pred in meta_preds if pred]

            if not orig_preds or not meta_preds:
                continue

            # ===== NEW: 计入评估样本 =====
            total_eval_cases += 1

            orig_final_pred, _ = Counter(orig_preds).most_common(1)[0]
            meta_final_pred, _ = Counter(meta_preds).most_common(1)[0]

            orig_final_codes = [
                code for code, pred in zip(orig_symbolic_states, orig_preds)
                if pred == orig_final_pred
            ]
            meta_final_codes = [
                code for code, pred in zip(meta_symbolic_states, meta_preds)
                if pred == meta_final_pred
            ]

            orig_correct = eval_ex_match(true_answer, orig_final_pred)
            inconsistency = not eval_ex_match(orig_final_pred, meta_final_pred)

            same_codes = list({
                o_code
                for o_code in orig_final_codes
                for m_code in meta_final_codes
                if o_code == m_code
            })

            # ===== NEW: same_code 统计 =====
            if same_codes:
                same_code_cases += 1

            # ===== 更新混淆矩阵（原逻辑不变）=====
            if same_codes:
                if not orig_correct:
                    if inconsistency:
                        tp += 1
                        if os.path.exists(save_path):
                            with open(save_path, "r", encoding="utf-8") as f:
                                all_data = json.load(f)
                        else:
                            all_data = []

                        all_data.append({
                            "question_id": result.get("question_id"),
                            "question": orig_data['question'],
                            "true_answer": true_answer,
                            "orig_final_pred": orig_final_pred,
                            "meta_final_pred": meta_final_pred,
                            "same_codes": same_codes
                        })

                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(all_data, f, ensure_ascii=False, indent=2)
                    else:
                        fn += 1
                else:
                    if inconsistency:
                        fp += 1
                    else:
                        tn += 1

            # ===== 按类别统计 =====
            table_columns = orig_data.get("table_columns", [])
            categories = classify_question(orig_data["question"], table_columns)

            for cat in categories:
                if not orig_correct:
                    if inconsistency:
                        category_tp[cat] += 1 if same_codes else 0
                        category_fn[cat] += 1 if not same_codes else 0
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
                        category_tp["MULTI_OP"] += 1 if same_codes else 0
                        category_fn["MULTI_OP"] += 1 if not same_codes else 0
                    else:
                        category_fn["MULTI_OP"] += 1
                else:
                    if inconsistency:
                        category_fp["MULTI_OP"] += 1
                    else:
                        category_tn["MULTI_OP"] += 1

        # ===== 计算总体指标 =====
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        overall_metrics["precision"].append(precision)
        overall_metrics["recall"].append(recall)
        overall_metrics["f1"].append(f1)

    # ===== 平均指标 =====
    final_metrics = {
        "overall": {
            "precision": np.mean(overall_metrics["precision"]) * 100,
            "recall": np.mean(overall_metrics["recall"]) * 100,
            "f1": np.mean(overall_metrics["f1"]) * 100,
            "precision_std": np.std(overall_metrics["precision"]) * 100,
            "recall_std": np.std(overall_metrics["recall"]) * 100,
            "f1_std": np.std(overall_metrics["f1"]) * 100,
        },
        "same_code_ratio": {
            "same_code_cases": same_code_cases,
            "total_cases": total_eval_cases,
            "ratio": same_code_cases / total_eval_cases * 100 if total_eval_cases > 0 else 0
        },
        "by_category": {}
    }

    print("\n📊 ========== 蜕变测试评估结果 ==========")
    print(f"总评估样本数: {total_eval_cases}")
    print(f"same_code 样本数: {same_code_cases}")
    print(f"same_code 占比: {final_metrics['same_code_ratio']['ratio']:.2f}%")
    print(f"Precision: {final_metrics['overall']['precision']:.2f}% ± {final_metrics['overall']['precision_std']:.2f}%")
    print(f"Recall:    {final_metrics['overall']['recall']:.2f}% ± {final_metrics['overall']['recall_std']:.2f}%")
    print(f"F1 Score:  {final_metrics['overall']['f1']:.2f}% ± {final_metrics['overall']['f1_std']:.2f}%")

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






if __name__ == "__main__":

    metrics = eval_metamorphic_wtq(
        original_checkpoint="./output_tablegpt_agent_base/wtq_agent/result.jsonl",
        metamorphic_checkpoint="./output_tablegpt_agent_column_shuffle/wtq_agent/result.jsonl",
        n_times=1
    )
# 读取保存的 JSON 文件
if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
else:
    all_data = []

# 统计 true_answer 不是空字符串的条目数
num_error_with_same_code = sum(1 for entry in all_data if entry.get("true_answer"))

print(f"共有 {num_error_with_same_code} 个错误程序找到相同代码段")
# PMR1
# Precision: 48.37% ± 0.00%
# Recall:    33.20% ± 0.00%
# F1 Score:  39.37% ± 0.00%




# PMR1
# #TP 总数: 1155
#   same_codes 非空: 577 (49.96%)
#   same_codes 为空: 578 (50.04%)

#PMR2
# TP 总数: 1067
#   same_codes 非空: 553 (51.83%)
#   same_codes 为空: 514 (48.17%)

#PMR3
# TP 总数: 1294
#   same_codes 非空: 262 (20.25%)
#   same_codes 为空: 1032 (79.75%)

# PMR4
# TP 总数: 1148
#   same_codes 非空: 469 (40.85%)
#   same_codes 为空: 679 (59.15%)
#
# DMR1
# 总数: 23
#   same_codes 非空: 16 (69.57%)
#   same_codes 为空: 7 (30.43%)

# DMR2
# 非空: 2(5.26 %)
# same_codes
# 为空: 36(94.74 %)

#SMR1
# TP 总数: 1027
#   same_codes 非空: 484 (47.13%)
#   same_codes 为空: 543 (52.87%)


#SMR2
# TP 总数: 986
#   same_codes 非空: 581 (58.92%)
#   same_codes 为空: 405 (41.08%)

#SMR3
# TP 总数: 24
#   same_codes 非空: 11 (45.83%)
#   same_codes 为空: 13 (54.17%)
