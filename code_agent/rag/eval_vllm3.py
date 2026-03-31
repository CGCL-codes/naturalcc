#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import Levenshtein

from vllm import LLM, SamplingParams

from .utils import DS_FILE, PT_FILE, LC_PT_FILE, EVAL_FILE, RESULT_FILE, MODEL, IMP_FILE


# -------------------------
# IO
# -------------------------
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


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
# C completion postprocess (aligned with Java rules)
# -------------------------
def process_c_completion(completion, add_log=False):
    """
    单行补全版后处理（适合 C/Java 风格单行语句补全）：
    - 避开字符串/字符/注释
    - 在顶层遇到：
        1) ';'  → 截断并返回（包含 ';'）
        2) '\n' → 截断并返回（不包含换行）
    - ✅ 与 Java 规则对齐：在 /* ... */ 块注释内部遇到 '\n' 也立即截断（不含换行）
    - ✅ 行注释：在 // ... 内遇到 '\n' 立即截断（不含换行）
    """
    if not completion:
        return ""

    original = completion.strip()

    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    for i, ch in enumerate(completion):
        nxt = completion[i + 1] if i + 1 < len(completion) else ""

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
                if add_log and len(result) < len(original):
                    print(f"截断(换行/行注释): '{original}' -> '{result}'")
                return result
            continue

        # block comment: ✅ 在注释内遇到换行也截断返回
        if in_block_comment:
            if ch == "\n":
                result = completion[:i].strip()
                if add_log and len(result) < len(original):
                    print(f"截断(换行/块注释): '{original}' -> '{result}'")
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
            if add_log and len(result) < len(original):
                print(f"截断(换行): '{original}' -> '{result}'")
            return result

        # 顶层：遇到 ';' 就停（含 ';'）
        if ch == ";":
            result = completion[: i + 1].strip()
            if add_log and len(result) < len(original):
                print(f"截断(;): '{original}' -> '{result}'")
            return result

    return original


# -------------------------
# vLLM batched generation
# -------------------------
def generate_completion_batch_vllm(llm, sampling_params, prompts):
    """
    使用 vLLM 对一批 prompts 做生成（C后处理）
    """
    if not prompts:
        return []

    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        generations = []
        for out in outputs:
            text = out.outputs[0].text
            generations.append(process_c_completion(text, add_log=False))
        return generations

    except Exception as e:
        print(f"vLLM 批量生成过程中发生错误: {e}")
        print("尝试单个样本处理...")
        generations = []
        for prompt in prompts:
            try:
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                text = outputs[0].outputs[0].text
                generations.append(process_c_completion(text, add_log=False))
            except Exception as e2:
                print(f"单个样本处理失败: {e2}")
                generations.append("")
        return generations


# -------------------------
# Metrics
# -------------------------
def compute_exact_match(prediction, ground_truth):
    if prediction is None:
        prediction = ""
    if ground_truth is None:
        ground_truth = ""
    return 1 if str(prediction).strip() == str(ground_truth).strip() else 0


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
def main():
    parser = argparse.ArgumentParser(description=f"评估{MODEL}模型的代码补全性能（vLLM版，三输入源）")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    args = parser.parse_args()

    config = load_config()
    max_to_generate = config["max_to_generate"]
    print("max_to_generate =", max_to_generate)

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

    # ========= 使用 vLLM 加载模型 =========
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

    # ========= 输出文件（三套）=========
    result_dir = os.path.dirname(RESULT_FILE)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    raw_result_file = RESULT_FILE.replace(".json", "_raw.json")
    mp_result_file = RESULT_FILE.replace(".json", "_modelprompt.json")
    lc_result_file = RESULT_FILE.replace(".json", "_langchain.json")

    raw_eval_file = EVAL_FILE.replace(".txt", "_raw.txt")
    mp_eval_file = EVAL_FILE.replace(".txt", "_modelprompt.txt")
    lc_eval_file = EVAL_FILE.replace(".txt", "_langchain.txt")

    # ========= 三套评估容器 =========
    raw_results, mp_results, lc_results = [], [], []

    raw_exact_matches, raw_edit_similarities = [], []
    mp_exact_matches, mp_edit_similarities = [], []
    lc_exact_matches, lc_edit_similarities = [], []

    improved_samples_mp = []
    improved_samples_lc = []

    batch_size = args.batch_size
    num_samples = len(dataset)

    print(f"总样本数: {num_samples}, 批大小: {batch_size}")

    max_print_samples = 10
    printed_samples = 0

    for i in tqdm(range(0, num_samples, batch_size), desc="处理批次"):
        batch = dataset[i:i + batch_size]
        batch_ids = [s.get("id", "") for s in batch]
        batch_inputs = [s.get("input", "") for s in batch]
        batch_gts = [s.get("gt", "") for s in batch]

        batch_model_prompts = [model_prompt_dict.get(id_, "") for id_ in batch_ids]
        batch_lc_prompts = [lc_prompt_dict.get(id_, "") for id_ in batch_ids]

        # 1) raw
        raw_preds = generate_completion_batch_vllm(llm, sampling_params, batch_inputs)

        # 2) model prompt（有些 prompt 可能为空；但评估时仍计入所有样本）
        mp_preds = [""] * len(batch_ids)
        mp_valid_idx = [k for k, p in enumerate(batch_model_prompts) if p]
        if mp_valid_idx:
            mp_valid_prompts = [batch_model_prompts[k] for k in mp_valid_idx]
            mp_valid_preds = generate_completion_batch_vllm(llm, sampling_params, mp_valid_prompts)
            for t, k in enumerate(mp_valid_idx):
                mp_preds[k] = mp_valid_preds[t]

        # 3) langchain prompt（有些 prompt 可能为空；但评估时仍计入所有样本）
        lc_preds = [""] * len(batch_ids)
        lc_valid_idx = [k for k, p in enumerate(batch_lc_prompts) if p]
        if lc_valid_idx:
            lc_valid_prompts = [batch_lc_prompts[k] for k in lc_valid_idx]
            lc_valid_preds = generate_completion_batch_vllm(llm, sampling_params, lc_valid_prompts)
            for t, k in enumerate(lc_valid_idx):
                lc_preds[k] = lc_valid_preds[t]

        # 4) 计算指标 & 写结果（✅ 三路都计入所有样本，包括空输出）
        for j, sid in enumerate(batch_ids):
            gt = batch_gts[j]

            raw_pred = raw_preds[j]
            mp_pred = mp_preds[j]
            lc_pred = lc_preds[j]

            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)
            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)

            mp_em = compute_exact_match(mp_pred, gt)
            mp_es = compute_edit_similarity(mp_pred, gt)
            mp_exact_matches.append(mp_em)
            mp_edit_similarities.append(mp_es)

            lc_em = compute_exact_match(lc_pred, gt)
            lc_es = compute_edit_similarity(lc_pred, gt)
            lc_exact_matches.append(lc_em)
            lc_edit_similarities.append(lc_es)

            # 记录 “raw没对，但mp对了”
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

            # 记录 “raw没对，但lc对了”
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

            # 打印前若干个样本
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

    # ========= 汇总指标 =========
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
    def write_eval(path, title, em, es, n_used):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Used samples (all samples): {n_used}\n")
            f.write(f"Exact Match: {em:.4f}\n")
            f.write(f"Edit Similarity: {es:.4f}\n")

    # ✅ 三路 used samples 统一为总样本数
    write_eval(raw_eval_file, "1) raw(input) 评估结果", avg_raw_em, avg_raw_es, num_samples)
    write_eval(mp_eval_file, "2) model prompt 评估结果", avg_mp_em, avg_mp_es, num_samples)
    write_eval(lc_eval_file, "3) langchain prompt 评估结果", avg_lc_em, avg_lc_es, num_samples)

    # ========= 保存改进样本（分别两份）=========
    imp_mp_file = IMP_FILE.replace(".json", "_modelprompt.json")
    imp_lc_file = IMP_FILE.replace(".json", "_langchain.json")

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
