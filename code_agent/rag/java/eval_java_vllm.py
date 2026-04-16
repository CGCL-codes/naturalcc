#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# =========================
# vLLM 日志/进度条控制（必须在 import vllm 前设置）
# =========================
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 固定可见 GPU（vLLM 也会用这个环境变量）
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import json
import yaml
import numpy as np
import argparse
from tqdm import tqdm
import Levenshtein

from vllm import LLM, SamplingParams

from utils import DS_FILE, PT_FILE, EVAL_FILE, RESULT_FILE, MODEL, IMP_FILE


# -------------------------
# IO
# -------------------------

def load_config():
    with open("./config.yaml", "r") as f:
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
# Java completion postprocess
# -------------------------
def process_java_completion(completion, add_log=False):
    """
    单行补全版后处理（适合 Java/C 风格单行语句补全）：
    - 避开字符串/字符/注释
    - 在顶层遇到：
        1) ';'  → 截断并返回（包含 ';'）
        2) '\n' → 截断并返回（不包含换行）
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

        # line comment
        if in_line_comment:
            if ch == "\n":
                # 单行补全：到换行就结束
                result = completion[:i].strip()
                if add_log and len(result) < len(original):
                    print(f"截断(换行): '{original}' -> '{result}'")
                return result
            continue

        # block comment
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

        # 单行：遇到换行就停（不含换行）
        if ch == "\n":
            result = completion[:i].strip()
            if add_log and len(result) < len(original):
                print(f"截断(换行): '{original}' -> '{result}'")
            return result

        # 单行语句：遇到 ';' 就停（含 ';'）
        # if ch == ";":
        #     result = completion[: i + 1].strip()
        #     if add_log and len(result) < len(original):
        #         print(f"截断(;): '{original}' -> '{result}'")
        #     return result

    return original


def generate_completion_batch_vllm(llm, sampling_params, prompts):
    """
    使用 vLLM 对一批 prompts 做生成（Java后处理）
    """
    if not prompts:
        return []

    try:
        outputs = llm.generate(prompts, sampling_params,use_tqdm=False)
        generations = []
        for out in outputs:
            text = out.outputs[0].text
            processed_text = process_java_completion(text, add_log=False)
            generations.append(processed_text)
        return generations

    except Exception as e:
        print(f"vLLM 批量生成过程中发生错误: {e}")
        print("尝试单个样本处理...")
        generations = []
        for prompt in prompts:
            try:
                outputs = llm.generate([prompt], sampling_params)
                text = outputs[0].outputs[0].text
                processed_text = process_java_completion(text, add_log=False)
                generations.append(processed_text)
            except Exception as e2:
                print(f"单个样本处理失败: {e2}")
                generations.append("")
        return generations


# -------------------------
# Metrics
# -------------------------

def compute_exact_match(prediction, ground_truth):
    return 1 if prediction.strip() == ground_truth.strip() else 0


def compute_edit_similarity(prediction, ground_truth):
    p = prediction.strip()
    g = ground_truth.strip()
    edit_distance = Levenshtein.distance(p, g)
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    return 1.0 - (edit_distance / max_len)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description=f"评估{MODEL}模型的 Java 代码补全性能（vLLM版）")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    args = parser.parse_args()

    config = load_config()
    max_to_generate = config["max_to_generate"]
    print("max_to_generate =", max_to_generate)

    if not os.path.exists(DS_FILE):
        print(f"错误：找不到数据集文件 {DS_FILE}")
        return

    if not os.path.exists(PT_FILE):
        print(f"错误：找不到提示文件 {PT_FILE}")
        return

    dataset = load_jsonl(DS_FILE)
    prompts = load_jsonl(PT_FILE)

    prompt_dict = {item.get("id", ""): item.get("prompt", "") for item in prompts}
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
        tensor_parallel_size=4,     # ✅ 使用 4 张 GPU：4,5,6,7
    )

    print("vLLM 模型加载完成！")

    results = []
    improved_samples = []
    raw_exact_matches = []
    raw_edit_similarities = []
    prompt_exact_matches = []
    prompt_edit_similarities = []

    batch_size = args.batch_size
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"总样本数: {num_samples}, 批大小: {batch_size}, 总批次: {num_batches}")

    max_print_samples = 10
    printed_samples = 0

    for i in tqdm(range(0, num_samples, batch_size), desc="处理批次"):
        current_batch = dataset[i : i + batch_size]
        batch_ids = [sample.get("id", "") for sample in current_batch]
        batch_inputs = [sample.get("input", "") for sample in current_batch]
        batch_gts = [sample.get("gt", "") for sample in current_batch]

        batch_prompts = [prompt_dict.get(id_, "") for id_ in batch_ids]

        # 1) raw
        raw_preds = generate_completion_batch_vllm(llm, sampling_params, batch_inputs)

        # 2) prompt-enhanced
        valid_prompts = [p for p in batch_prompts if p]
        prompt_preds_map = {}

        if valid_prompts:
            valid_indices = [idx for idx, p in enumerate(batch_prompts) if p]
            valid_ids = [batch_ids[idx] for idx in valid_indices]

            prompt_preds = generate_completion_batch_vllm(llm, sampling_params, valid_prompts)

            for idx, id_ in enumerate(valid_ids):
                prompt_preds_map[id_] = prompt_preds[idx]

        # 3) metrics
        for j, sample_id in enumerate(batch_ids):
            gt = batch_gts[j]
            raw_pred = raw_preds[j]
            prompt_pred = prompt_preds_map.get(sample_id, "")

            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)
            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)

            if printed_samples < max_print_samples:
                print("\n========== 样本", sample_id, "==========")
                print("【Input】")
                print(batch_inputs[j])
                print("【Raw Prediction】")
                print(raw_pred)
                print("【Prompt Prediction】")
                print(prompt_pred)
                print("【Ground Truth】")
                print(gt)
                print("=====================================\n")
                printed_samples += 1

            if prompt_pred:
                prompt_em = compute_exact_match(prompt_pred, gt)
                prompt_es = compute_edit_similarity(prompt_pred, gt)

                if raw_em == 0 and prompt_em == 1:
                    sample_data = dataset_dict.get(sample_id, {})
                    improved_samples.append(
                        {
                            "id": sample_id,
                            "pkg": sample_data.get("pkg", ""),
                            "fpath": sample_data.get("fpath", ""),
                            "input": sample_data.get("input", ""),
                            "raw_res": raw_pred,
                            "prompt_res": prompt_pred,
                            "gt": gt,
                        }
                    )
                    print("✨ 提示增强带来改进的样本:", sample_id)
                    print("  Raw   =", raw_pred)
                    print("  Prompt=", prompt_pred)
                    print("  GT    =", gt)

                prompt_exact_matches.append(prompt_em)
                prompt_edit_similarities.append(prompt_es)

            results.append(
                {
                    "id": sample_id,
                    "raw_res": raw_pred,
                    "prompt_res": prompt_pred,
                    "gt": gt,
                }
            )

    # 汇总指标
    avg_raw_exact_match = float(np.mean(raw_exact_matches)) if raw_exact_matches else 0.0
    avg_raw_edit_similarity = float(np.mean(raw_edit_similarities)) if raw_edit_similarities else 0.0
    avg_prompt_exact_match = float(np.mean(prompt_exact_matches)) if prompt_exact_matches else 0.0
    avg_prompt_edit_similarity = float(np.mean(prompt_edit_similarities)) if prompt_edit_similarities else 0.0

    # 确保结果目录存在
    for path in [RESULT_FILE, EVAL_FILE, IMP_FILE]:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # 保存详细结果
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存评估指标
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        f.write("评估结果汇总（vLLM，Java）\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. 原始输入评估结果 (raw_res):\n")
        f.write(f"   - Exact Match: {avg_raw_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_raw_edit_similarity:.4f}\n\n")

        f.write("2. 提示输入评估结果 (prompt_res):\n")
        f.write(f"   - Exact Match: {avg_prompt_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_prompt_edit_similarity:.4f}\n")

    # 保存改进样本
    if improved_samples:
        with open(IMP_FILE, "w", encoding="utf-8") as f:
            json.dump(improved_samples, f, ensure_ascii=False, indent=2)
        print(f"找到 {len(improved_samples)} 个通过提示改进的样本，已保存到 {IMP_FILE}")

    print(f"评估完成，结果已保存到 {EVAL_FILE}，详细结果保存到 {RESULT_FILE}")


if __name__ == "__main__":
    main()
