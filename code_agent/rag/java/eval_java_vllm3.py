#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# =========================
# vLLM 日志/进度条控制（必须在 import vllm 前设置）
# =========================
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import Levenshtein

from vllm import LLM, SamplingParams

# ✅ 多导入 LC_PT_FILE
from utils import DS_FILE, PT_FILE, LC_PT_FILE, EVAL_FILE, RESULT_FILE, MODEL, IMP_FILE


# -------------------------
# IO
# -------------------------
# [maint-note:b0462be5df6f]
# 维护导航：load_config；下列为本函数源码中的接口与路径索引。
# 返回路径：config。
# 直接调用：open。
# 直接调用：yaml.safe_load。
# 资源作用域：open('./config.yaml', 'r')；退出该 with 时调用对应上下文清理。
# [/maint-note:b0462be5df6f]
def load_config():
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


# [maint-note:172aba55e3c3]
# 维护导航：load_jsonl；下列为本函数源码中的接口与路径索引。
# 输入 file_path：类型由调用方与函数体约束。
# 分支条件：not line。
# 遍历边界：line ← f。
# 返回路径：data。
# 直接调用：open。
# 直接调用：line.strip。
# 直接调用：data.append。
# 直接调用：json.loads。
# 资源作用域：open(file_path, 'r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:172aba55e3c3]
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# -------------------------
# Java completion postprocess
# -------------------------
# [maint-note:1d935ccf5fea]
# 维护导航：process_java_completion；下列为本函数源码中的接口与路径索引。
# 输入 completion：类型由调用方与函数体约束。
# 输入 add_log：类型由调用方与函数体约束；缺省表达式 False。
# 分支条件：not completion。
# 分支条件：add_log and len(result) < len(original)。
# 分支条件：escaped。
# 分支条件：ch == '\\' and (in_string or in_char)。
# 分支条件：in_line_comment。
# 分支条件：ch == '\n'。
# 分支条件：in_block_comment。
# 分支条件：ch == '*' and nxt == '/'。
# 分支条件：in_string。
# 分支条件：ch == '"' and (not escaped)。
# 分支条件：in_char。
# 分支条件：ch == "'" and (not escaped)。
# 分支条件：ch == '/' and nxt == '/'。
# 分支条件：ch == '/' and nxt == '*'。
# 分支条件：ch == '"'。
# 分支条件：ch == "'"。
# 遍历边界：(i, ch) ← enumerate(completion)。
# 返回路径：''。
# 返回路径：result。
# 返回路径：original。
# 直接调用：completion.strip。
# 直接调用：len。
# 直接调用：print。
# 直接调用：enumerate。
# 直接调用：completion[:i].strip。
# 直接调用：_log。
# 表达式分流：i + 1 < n；真值与假值分支分别计算，不同时执行。
# [/maint-note:1d935ccf5fea]
def process_java_completion(completion, add_log=False):
    """
    单行补全版后处理（C/Java）——只按换行截断：
    - 避开字符串/字符/注释
    - 顶层遇到 '\n' → 截断并返回（不包含换行）
    - 行注释 //... 内遇到 '\n' → 截断并返回（不包含换行）
    - ✅ 与 Java 规则对齐：块注释 /*...*/ 内遇到 '\n' 也立即截断（不包含换行）
    - ⚠️ ';' 不再作为终止符：不会因为 ';' 截断
    """
    if not completion:
        return ""

    original = completion.strip()

    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    n = len(completion)

    # [maint-note:0035877a57a0]
    # 维护导航：_log；下列为本函数源码中的接口与路径索引。
    # 输入 reason：类型由调用方与函数体约束。
    # 输入 result：类型由调用方与函数体约束。
    # 分支条件：add_log and len(result) < len(original)。
    # 直接调用：len。
    # 直接调用：print。
    # [/maint-note:0035877a57a0]
    def _log(reason, result):
        if add_log and len(result) < len(original):
            print(f"截断({reason}): '{original}' -> '{result}'")

    for i, ch in enumerate(completion):
        nxt = completion[i + 1] if i + 1 < n else ""

        # escape handling（仅对 string/char 生效）
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_string or in_char):
            escaped = True
            continue

        # line comment: 到换行就截断返回
        if in_line_comment:
            if ch == "\n":
                result = completion[:i].strip()
                _log("换行/行注释", result)
                return result
            continue

        # block comment: ✅ 在注释内遇到换行也截断返回
        if in_block_comment:
            if ch == "\n":
                result = completion[:i].strip()
                _log("换行/块注释", result)
                return result

            if ch == "*" and nxt == "/":
                in_block_comment = False
            continue

        # string
        if in_string:
            if ch == '"' and not escaped:
                in_string = False
            continue

        # char
        if in_char:
            if ch == "'" and not escaped:
                in_char = False
            continue

        # entering comments
        if ch == "/" and nxt == "/":
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            continue

        # entering string/char
        if ch == '"':
            in_string = True
            continue
        if ch == "'":
            in_char = True
            continue

        # 顶层：遇到换行就停（不含换行）
        if ch == "\n":
            result = completion[:i].strip()
            _log("换行", result)
            return result

        # ';'：不做任何处理（不截断）
        # if ch == ";": pass

    return original


# [maint-note:3720571bd3a9]
# 维护导航：generate_completion_batch_vllm；下列为本函数源码中的接口与路径索引。
# 输入 llm：类型由调用方与函数体约束。
# 输入 sampling_params：类型由调用方与函数体约束。
# 输入 prompts：类型由调用方与函数体约束。
# 分支条件：not prompts。
# 遍历边界：out ← outputs。
# 遍历边界：prompt ← prompts。
# 返回路径：[]。
# 返回路径：generations。
# 捕获异常：Exception。
# 直接调用：llm.generate。
# 直接调用：generations.append。
# 直接调用：process_java_completion。
# 直接调用：print。
# [/maint-note:3720571bd3a9]
def generate_completion_batch_vllm(llm, sampling_params, prompts):
    """
    使用 vLLM 对一批 prompts 做生成（Java后处理）
    """
    if not prompts:
        return []

    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        generations = []
        for out in outputs:
            text = out.outputs[0].text
            generations.append(process_java_completion(text, add_log=False))
        return generations

    except Exception as e:
        print(f"vLLM 批量生成过程中发生错误: {e}")
        print("尝试单个样本处理...")
        generations = []
        for prompt in prompts:
            try:
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                text = outputs[0].outputs[0].text
                generations.append(process_java_completion(text, add_log=False))
            except Exception as e2:
                print(f"单个样本处理失败: {e2}")
                generations.append("")
        return generations


# -------------------------
# Metrics
# -------------------------
# [maint-note:9e86b688f921]
# 维护导航：compute_exact_match；下列为本函数源码中的接口与路径索引。
# 输入 prediction：类型由调用方与函数体约束。
# 输入 ground_truth：类型由调用方与函数体约束。
# 分支条件：prediction is None。
# 分支条件：ground_truth is None。
# 返回路径：1 if str(prediction).strip() == str(ground_truth).strip() else 0。
# 直接调用：str(prediction).strip。
# 直接调用：str。
# 直接调用：str(ground_truth).strip。
# 表达式分流：str(prediction).strip() == str(ground_truth).strip()；真值与假值分支分别计算，不同时执行。
# [/maint-note:9e86b688f921]
def compute_exact_match(prediction, ground_truth):
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    return 1 if str(prediction).strip() == str(ground_truth).strip() else 0


# [maint-note:0fae345f1980]
# 维护导航：compute_edit_similarity；下列为本函数源码中的接口与路径索引。
# 输入 prediction：类型由调用方与函数体约束。
# 输入 ground_truth：类型由调用方与函数体约束。
# 分支条件：prediction is None。
# 分支条件：ground_truth is None。
# 分支条件：max_len == 0。
# 返回路径：1.0。
# 返回路径：1.0 - edit_distance / max_len。
# 直接调用：str(prediction).strip。
# 直接调用：str。
# 直接调用：str(ground_truth).strip。
# 直接调用：Levenshtein.distance。
# 直接调用：max。
# 直接调用：len。
# [/maint-note:0fae345f1980]
def compute_edit_similarity(prediction, ground_truth):
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    p = str(prediction).strip()
    g = str(ground_truth).strip()
    edit_distance = Levenshtein.distance(p, g)
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    return 1.0 - (edit_distance / max_len)


# -------------------------
# Main (3-way eval)
# -------------------------
# [maint-note:c7647d8c9df9]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 分支条件：not os.path.exists(DS_FILE)。
# 分支条件：not os.path.exists(PT_FILE)。
# 分支条件：not os.path.exists(LC_PT_FILE)。
# 分支条件：directory and (not os.path.exists(directory))。
# 分支条件：mp_valid_idx。
# 分支条件：lc_valid_idx。
# 分支条件：str(mp_pred).strip()。
# 分支条件：str(lc_pred).strip()。
# 分支条件：raw_em == 0 and mp_em == 1。
# 分支条件：raw_em == 0 and lc_em == 1。
# 分支条件：printed_samples < max_print_samples。
# 分支条件：improved_samples_mp。
# 分支条件：improved_samples_lc。
# 遍历边界：path ← [raw_result_file, mp_result_file, lc_result_file, raw_eval_file, mp_eval_file, lc_eval_file, imp_mp_file, imp_lc_file]。
# 遍历边界：i ← tqdm(range(0, num_samples, batch_size), desc='处理批次')。
# 遍历边界：(t, k) ← enumerate(mp_valid_idx)。
# 遍历边界：(t, k) ← enumerate(lc_valid_idx)。
# 遍历边界：(j, sid) ← enumerate(batch_ids)。
# 返回路径：无返回值。
# 返回路径：float(np.mean(xs)) if xs else 0.0。
# 写入目标：mp_preds[k]。
# 写入目标：lc_preds[k]。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：load_config。
# 直接调用：print。
# 直接调用：os.path.exists。
# 直接调用：load_jsonl。
# 直接调用：item.get。
# 直接调用：MODEL.lower。
# 直接调用：SamplingParams。
# 直接调用：LLM。
# 直接调用：RESULT_FILE.replace。
# 直接调用：EVAL_FILE.replace。
# 直接调用：IMP_FILE.replace。
# 直接调用：os.path.dirname。
# 直接调用：os.makedirs。
# 直接调用：len。
# 直接调用：tqdm。
# 直接调用：range。
# 直接调用：s.get。
# 直接调用：model_prompt_dict.get。
# 直接调用：lc_prompt_dict.get。
# 直接调用：generate_completion_batch_vllm。
# 直接调用：enumerate。
# 直接调用：compute_exact_match。
# 直接调用：compute_edit_similarity。
# 直接调用：raw_exact_matches.append。
# 直接调用：raw_edit_similarities.append。
# 直接调用：str(mp_pred).strip。
# 直接调用：str。
# 直接调用：mp_exact_matches.append。
# 直接调用：mp_edit_similarities.append。
# 直接调用：str(lc_pred).strip。
# 直接调用：lc_exact_matches.append。
# 直接调用：lc_edit_similarities.append。
# 直接调用：dataset_dict.get。
# 直接调用：improved_samples_mp.append。
# 直接调用：sd.get。
# 直接调用：improved_samples_lc.append。
# 直接调用：raw_results.append。
# 直接调用：mp_results.append。
# 直接调用：lc_results.append。
# 直接调用：float。
# 直接调用：np.mean。
# 直接调用：safe_mean。
# 直接调用：open。
# 直接调用：json.dump。
# 直接调用：f.write。
# 直接调用：write_eval。
# 集合派生：item 来自 model_prompts。
# 集合派生：item 来自 lc_prompts。
# 集合派生：item 来自 dataset。
# 集合派生：s 来自 batch。
# 集合派生：sid 来自 batch_ids。
# 集合派生：(k, p) 来自 enumerate(batch_model_prompts)；过滤 p。
# 集合派生：k 来自 mp_valid_idx。
# 集合派生：(k, p) 来自 enumerate(batch_lc_prompts)；过滤 p。
# 集合派生：k 来自 lc_valid_idx。
# 资源作用域：open(raw_result_file, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(mp_result_file, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(lc_result_file, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(imp_mp_file, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(imp_lc_file, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:c7647d8c9df9]
def main():
    parser = argparse.ArgumentParser(description=f"评估{MODEL}模型的 Java 代码补全性能（vLLM，三输入源）")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    args = parser.parse_args()

    config = load_config()
    max_to_generate = config["max_to_generate"]
    print("max_to_generate =", max_to_generate)

    # 检查文件
    if not os.path.exists(DS_FILE):
        print(f"错误：找不到数据集文件 {DS_FILE}")
        return
    if not os.path.exists(PT_FILE):
        print(f"错误：找不到模型提示文件 {PT_FILE}")
        return
    if not os.path.exists(LC_PT_FILE):
        print(f"错误：找不到 LangChain 提示文件 {LC_PT_FILE}")
        return

    dataset = load_jsonl(DS_FILE)
    model_prompts = load_jsonl(PT_FILE)
    lc_prompts = load_jsonl(LC_PT_FILE)

    model_prompt_dict = {item.get("id", ""): item.get("prompt", "") for item in model_prompts}
    lc_prompt_dict = {item.get("id", ""): item.get("prompt", "") for item in lc_prompts}
    dataset_dict = {item.get("id", ""): item for item in dataset}

    # ========= vLLM load =========
    model_path = config[f"{MODEL.lower()}_repo"]
    print(f"使用 vLLM 加载 {MODEL} 模型，自路径: {model_path}")

    sampling_params = SamplingParams(
        max_tokens=max_to_generate,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )

    llm = LLM(
        model=model_path,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=4,
    )
    print("vLLM 模型加载完成！")

    # ========= 输出文件（3 套）=========
    # 沿用 RESULT_FILE / EVAL_FILE 的目录，只改文件名后缀区分三路
    raw_result_file = RESULT_FILE.replace(".json", "_raw.json")
    mp_result_file = RESULT_FILE.replace(".json", "_modelprompt.json")
    lc_result_file = RESULT_FILE.replace(".json", "_langchain.json")

    raw_eval_file = EVAL_FILE.replace(".txt", "_raw.txt")
    mp_eval_file = EVAL_FILE.replace(".txt", "_modelprompt.txt")
    lc_eval_file = EVAL_FILE.replace(".txt", "_langchain.txt")

    imp_mp_file = IMP_FILE.replace(".json", "_modelprompt.json")
    imp_lc_file = IMP_FILE.replace(".json", "_langchain.json")

    # 确保目录存在
    for path in [
        raw_result_file, mp_result_file, lc_result_file,
        raw_eval_file, mp_eval_file, lc_eval_file,
        imp_mp_file, imp_lc_file
    ]:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # ========= 统计容器 =========
    raw_results, mp_results, lc_results = [], [], []
    raw_exact_matches, raw_edit_similarities = [], []
    mp_exact_matches, mp_edit_similarities = [], []
    lc_exact_matches, lc_edit_similarities = [], []

    improved_samples_mp = []
    improved_samples_lc = []

    batch_size = args.batch_size
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"总样本数: {num_samples}, 批大小: {batch_size}, 总批次: {num_batches}")

    max_print_samples = 10
    printed_samples = 0

    for i in tqdm(range(0, num_samples, batch_size), desc="处理批次"):
        batch = dataset[i: i + batch_size]
        batch_ids = [s.get("id", "") for s in batch]
        batch_inputs = [s.get("input", "") for s in batch]
        batch_gts = [s.get("gt", "") for s in batch]

        batch_model_prompts = [model_prompt_dict.get(sid, "") for sid in batch_ids]
        batch_lc_prompts = [lc_prompt_dict.get(sid, "") for sid in batch_ids]

        # 1) raw
        raw_preds = generate_completion_batch_vllm(llm, sampling_params, batch_inputs)

        # 2) model prompt（对非空 prompt 才生成；但评估时仍计入所有样本）
        mp_preds = [""] * len(batch_ids)
        mp_valid_idx = [k for k, p in enumerate(batch_model_prompts) if p]
        if mp_valid_idx:
            mp_valid_prompts = [batch_model_prompts[k] for k in mp_valid_idx]
            mp_valid_preds = generate_completion_batch_vllm(llm, sampling_params, mp_valid_prompts)
            for t, k in enumerate(mp_valid_idx):
                mp_preds[k] = mp_valid_preds[t]

        # 3) langchain prompt（对非空 prompt 才生成；但评估时仍计入所有样本）
        lc_preds = [""] * len(batch_ids)
        lc_valid_idx = [k for k, p in enumerate(batch_lc_prompts) if p]
        if lc_valid_idx:
            lc_valid_prompts = [batch_lc_prompts[k] for k in lc_valid_idx]
            lc_valid_preds = generate_completion_batch_vllm(llm, sampling_params, lc_valid_prompts)
            for t, k in enumerate(lc_valid_idx):
                lc_preds[k] = lc_valid_preds[t]

        # 4) metrics + results（✅ 三路都计入所有样本，包括空输出）
        for j, sid in enumerate(batch_ids):
            gt = batch_gts[j]
            raw_pred = raw_preds[j]
            mp_pred = mp_preds[j]
            lc_pred = lc_preds[j]

            # raw metrics
            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)
            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)

            # model prompt metrics (✅ always)
            mp_em = compute_exact_match(mp_pred, gt)
            mp_es = compute_edit_similarity(mp_pred, gt)
            if str(mp_pred).strip():  
                mp_exact_matches.append(mp_em)
                mp_edit_similarities.append(mp_es)
            # mp_exact_matches.append(mp_em)
            # mp_edit_similarities.append(mp_es)

            # langchain metrics (✅ always)
            lc_em = compute_exact_match(lc_pred, gt)
            lc_es = compute_edit_similarity(lc_pred, gt)
            if str(lc_pred).strip():  # ✅ 非空才计入指标
                lc_exact_matches.append(lc_em)
                lc_edit_similarities.append(lc_es)
            # lc_exact_matches.append(lc_em)
            # lc_edit_similarities.append(lc_es)

            # improved samples (只记录从 raw 错到 prompt 对的)
            if raw_em == 0 and mp_em == 1:
                sd = dataset_dict.get(sid, {})
                improved_samples_mp.append({
                    "id": sid,
                    "pkg": sd.get("pkg", ""),
                    "fpath": sd.get("fpath", ""),
                    "input": sd.get("input", ""),
                    "raw_res": raw_pred,
                    "prompt_res": mp_pred,
                    "gt": gt,
                    "improved_by": "model_prompt",
                })

            if raw_em == 0 and lc_em == 1:
                sd = dataset_dict.get(sid, {})
                improved_samples_lc.append({
                    "id": sid,
                    "pkg": sd.get("pkg", ""),
                    "fpath": sd.get("fpath", ""),
                    "input": sd.get("input", ""),
                    "raw_res": raw_pred,
                    "prompt_res": lc_pred,
                    "gt": gt,
                    "improved_by": "langchain_prompt",
                })

            # print samples
            if printed_samples < max_print_samples:
                print("\n========== 样本", sid, "==========")
                print("【Input】")
                print(batch_inputs[j])
                print("【Raw Prediction】")
                print(raw_pred)
                print("【Model Prompt Prediction】")
                print(mp_pred)
                print("【LangChain Prediction】")
                print(lc_pred)
                print("【Ground Truth】")
                print(gt)
                print("=====================================\n")
                printed_samples += 1

            raw_results.append({"id": sid, "pred": raw_pred, "gt": gt})
            mp_results.append({"id": sid, "pred": mp_pred, "gt": gt})
            lc_results.append({"id": sid, "pred": lc_pred, "gt": gt})

    # ========= 汇总 =========
    # [maint-note:3a69f4ff1ebf]
    # 维护导航：safe_mean；下列为本函数源码中的接口与路径索引。
    # 输入 xs：类型由调用方与函数体约束。
    # 返回路径：float(np.mean(xs)) if xs else 0.0。
    # 直接调用：float。
    # 直接调用：np.mean。
    # 表达式分流：xs；真值与假值分支分别计算，不同时执行。
    # [/maint-note:3a69f4ff1ebf]
    def safe_mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    avg_raw_em = safe_mean(raw_exact_matches)
    avg_raw_es = safe_mean(raw_edit_similarities)
    avg_mp_em = safe_mean(mp_exact_matches)
    avg_mp_es = safe_mean(mp_edit_similarities)
    avg_lc_em = safe_mean(lc_exact_matches)
    avg_lc_es = safe_mean(lc_edit_similarities)

    # ========= 保存三套 result =========
    with open(raw_result_file, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    with open(mp_result_file, "w", encoding="utf-8") as f:
        json.dump(mp_results, f, ensure_ascii=False, indent=2)
    with open(lc_result_file, "w", encoding="utf-8") as f:
        json.dump(lc_results, f, ensure_ascii=False, indent=2)

    # ========= 保存三套 eval =========
    # [maint-note:37cb1b7ecde0]
    # 维护导航：write_eval；下列为本函数源码中的接口与路径索引。
    # 输入 path：类型由调用方与函数体约束。
    # 输入 title：类型由调用方与函数体约束。
    # 输入 em：类型由调用方与函数体约束。
    # 输入 es：类型由调用方与函数体约束。
    # 输入 n_used：类型由调用方与函数体约束。
    # 直接调用：open。
    # 直接调用：f.write。
    # 资源作用域：open(path, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
    # [/maint-note:37cb1b7ecde0]
    def write_eval(path, title, em, es, n_used):
        with open(path, "w", encoding="utf-8") as f:
            f.write("评估结果汇总（vLLM，Java）\n")
            f.write("=" * 60 + "\n")
            f.write(f"{title}\n\n")
            f.write(f"Used samples (all samples): {n_used}\n")
            f.write(f"Exact Match     : {em:.4f}\n")
            f.write(f"Edit Similarity : {es:.4f}\n")

    # ✅ 三路 used samples 统一为总样本数
    write_eval(raw_eval_file, "1) raw(input) 评估结果", avg_raw_em, avg_raw_es, num_samples)
    write_eval(mp_eval_file, "2) model prompt 评估结果", avg_mp_em, avg_mp_es, num_samples)
    write_eval(lc_eval_file, "3) langchain prompt 评估结果", avg_lc_em, avg_lc_es, num_samples)

    # ========= 保存改进样本 =========
    if improved_samples_mp:
        with open(imp_mp_file, "w", encoding="utf-8") as f:
            json.dump(improved_samples_mp, f, ensure_ascii=False, indent=2)
        print(f"找到 {len(improved_samples_mp)} 个通过 model_prompt 改进的样本，已保存到 {imp_mp_file}")

    if improved_samples_lc:
        with open(imp_lc_file, "w", encoding="utf-8") as f:
            json.dump(improved_samples_lc, f, ensure_ascii=False, indent=2)
        print(f"找到 {len(improved_samples_lc)} 个通过 langchain_prompt 改进的样本，已保存到 {imp_lc_file}")

    print("\n✅ 三套评估完成，输出：")
    print("RAW result:", raw_result_file)
    print("RAW eval  :", raw_eval_file)
    print("MP  result:", mp_result_file)
    print("MP  eval  :", mp_eval_file)
    print("LC  result:", lc_result_file)
    print("LC  eval  :", lc_eval_file)


if __name__ == "__main__":
    main()
