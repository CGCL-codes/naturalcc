#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import utils

import json
import yaml
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import Levenshtein

from utils import DS_FILE, PT_FILE, EVAL_FILE, RESULT_FILE, MODEL, IMP_FILE


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


def get_input_device(model) -> torch.device:
    """
    ✅ 多卡 device_map 下不要用 model.device（可能不存在/不准确）
    ✅ 统一把输入放到“第一个参数所在设备”（通常是第一张卡）
    """
    return next(model.parameters()).device


def load_model_and_tokenizer(config):
    print(f"加载 {MODEL} 模型...")
    model_path = config[f"{MODEL.lower()}_repo"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        print("设置 pad_token 为 eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # 你可以按需改成 balanced / balanced_low_0 / auto / cuda:0
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print("模型加载完成！")
    return model, tokenizer


def process_c_completion(completion, add_log=False):
    """
    单行补全后处理（对齐你 vLLM 版）：
    - 避开字符串/字符/注释
    - 顶层遇到：
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

        # escape handling（仅 string/char 内有效）
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_string or in_char):
            escaped = True
            continue

        # line comment
        if in_line_comment:
            if ch == "\n":
                result = completion[:i].strip()
                if add_log and len(result) < len(original):
                    print(f"截断(换行): '{original}' -> '{result}'")
                return result
            continue

        # block comment
        if in_block_comment:
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

        # 顶层换行：单行补全截止（不含换行）
        if ch == "\n":
            result = completion[:i].strip()
            if add_log and len(result) < len(original):
                print(f"截断(换行): '{original}' -> '{result}'")
            return result

        # 顶层分号：语句结束（含分号）
        if ch == ";":
            result = completion[: i + 1].strip()
            if add_log and len(result) < len(original):
                print(f"截断(;): '{original}' -> '{result}'")
            return result

    return original


def generate_completion_batch(model, tokenizer, prompts, max_to_generate):
    if not prompts:
        return []

    device0 = get_input_device(model)

    try:
        # ✅ 多卡：不要 .to(model.device)，统一放到 device0
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        encoded = {k: v.to(device0) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask", None),
                max_new_tokens=max_to_generate,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generations = []
        for i, output in enumerate(outputs):
            input_len = encoded["input_ids"][i].shape[0]
            gen_text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            generations.append(process_c_completion(gen_text, add_log=False))
        return generations

    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        print("尝试单个样本处理...")

        generations = []
        for prompt in prompts:
            try:
                inp = tokenizer(prompt, return_tensors="pt")
                inp = {k: v.to(device0) for k, v in inp.items()}

                with torch.no_grad():
                    out = model.generate(
                        input_ids=inp["input_ids"],
                        attention_mask=inp.get("attention_mask", None),
                        max_new_tokens=max_to_generate,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                gen_text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                generations.append(process_c_completion(gen_text, add_log=False))

            except Exception as e2:
                print(f"单个样本处理失败: {e2}")
                generations.append("")
        return generations


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


def ensure_output_dirs():
    for p in [RESULT_FILE, EVAL_FILE, IMP_FILE]:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=f"评估 {MODEL} 模型的代码补全性能（HF版）")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
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

    model, tokenizer = load_model_and_tokenizer(config)

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

    for i in tqdm(range(0, num_samples, batch_size), desc="处理批次"):
        current_batch = dataset[i : i + batch_size]
        batch_ids = [s.get("id", "") for s in current_batch]
        batch_inputs = [s.get("input", "") for s in current_batch]
        batch_gts = [s.get("gt", "") for s in current_batch]
        batch_prompts = [prompt_dict.get(_id, "") for _id in batch_ids]

        # raw
        raw_preds = generate_completion_batch(model, tokenizer, batch_inputs, max_to_generate)

        # prompt
        valid_prompts = [p for p in batch_prompts if p]
        prompt_preds_map = {}
        if valid_prompts:
            valid_indices = [idx for idx, p in enumerate(batch_prompts) if p]
            valid_ids = [batch_ids[idx] for idx in valid_indices]
            prompt_preds = generate_completion_batch(model, tokenizer, valid_prompts, max_to_generate)
            for idx, _id in enumerate(valid_ids):
                prompt_preds_map[_id] = prompt_preds[idx]

        for j, sample_id in enumerate(batch_ids):
            gt = batch_gts[j]
            raw_pred = raw_preds[j]
            prompt_pred = prompt_preds_map.get(sample_id, "")

            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)
            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)

            if prompt_pred:
                prompt_em = compute_exact_match(prompt_pred, gt)
                prompt_es = compute_edit_similarity(prompt_pred, gt)
                prompt_exact_matches.append(prompt_em)
                prompt_edit_similarities.append(prompt_es)

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

            results.append(
                {
                    "id": sample_id,
                    "raw_res": raw_pred,
                    "prompt_res": prompt_pred,
                    "gt": gt,
                }
            )

    avg_raw_exact_match = float(np.mean(raw_exact_matches)) if raw_exact_matches else 0.0
    avg_raw_edit_similarity = float(np.mean(raw_edit_similarities)) if raw_edit_similarities else 0.0
    avg_prompt_exact_match = float(np.mean(prompt_exact_matches)) if prompt_exact_matches else 0.0
    avg_prompt_edit_similarity = float(np.mean(prompt_edit_similarities)) if prompt_edit_similarities else 0.0

    ensure_output_dirs()

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        f.write("评估结果汇总（HF）\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. 原始输入评估结果 (raw_res):\n")
        f.write(f"   - Exact Match: {avg_raw_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_raw_edit_similarity:.4f}\n\n")
        f.write("2. 提示输入评估结果 (prompt_res):\n")
        f.write(f"   - Exact Match: {avg_prompt_exact_match:.4f}\n")
        f.write(f"   - Edit Similarity: {avg_prompt_edit_similarity:.4f}\n")

    if improved_samples:
        with open(IMP_FILE, "w", encoding="utf-8") as f:
            json.dump(improved_samples, f, ensure_ascii=False, indent=2)
        print(f"找到 {len(improved_samples)} 个通过提示改进的样本，已保存到 {IMP_FILE}")

    print(f"评估完成，结果已保存到 {EVAL_FILE}，详细结果保存到 {RESULT_FILE}")


if __name__ == "__main__":
    main()
