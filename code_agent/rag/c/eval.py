#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 固定可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import yaml
import torch
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import Levenshtein

from .utils import DS_FILE, PT_FILE, EVAL_FILE, RESULT_FILE, MODEL, IMP_FILE


# [maint-note:1d3aaf684836]
# 维护导航：load_config；下列为本函数源码中的接口与路径索引。
# 返回路径：config。
# 直接调用：open。
# 直接调用：os.path.join。
# 直接调用：os.path.dirname。
# 直接调用：yaml.safe_load。
# 资源作用域：open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:1d3aaf684836]
def load_config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


# [maint-note:a3a510d4b462]
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
# [/maint-note:a3a510d4b462]
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# [maint-note:ea042fdcf69b]
# 维护导航：load_model_and_tokenizer；下列为本函数源码中的接口与路径索引。
# 输入 config：类型由调用方与函数体约束。
# 分支条件：tokenizer.pad_token is None。
# 返回路径：(model, tokenizer)。
# 写入目标：tokenizer.pad_token。
# 直接调用：print。
# 直接调用：MODEL.lower。
# 直接调用：AutoTokenizer.from_pretrained。
# 直接调用：AutoModelForCausalLM.from_pretrained。
# 直接调用：model.eval。
# [/maint-note:ea042fdcf69b]
def load_model_and_tokenizer(config):
    print(f"加载{MODEL}模型...")
    model_path = config[f"{MODEL.lower()}_repo"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        print("设置pad_token为eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",          # 使用 auto，配合 CUDA_VISIBLE_DEVICES
        trust_remote_code=True
    )
    model.eval()  # 进入推理模式

    print("模型加载完成！")
    return model, tokenizer


# [maint-note:a94d1cdaf702]
# 维护导航：process_c_completion；下列为本函数源码中的接口与路径索引。
# 输入 completion：类型由调用方与函数体约束。
# 输入 add_log：类型由调用方与函数体约束；缺省表达式 False。
# 分支条件：not completion。
# 分支条件：char == '\\' and (not escaped)。
# 分支条件：escaped。
# 分支条件：in_line_comment。
# 分支条件：char == '\n'。
# 分支条件：in_block_comment。
# 分支条件：char == '*' and i + 1 < len(completion) and (completion[i + 1] == '/')。
# 分支条件：in_string。
# 分支条件：char == '"'。
# 分支条件：in_char。
# 分支条件：char == "'"。
# 分支条件：char == '/' and i + 1 < len(completion)。
# 分支条件：completion[i + 1] == '/'。
# 分支条件：completion[i + 1] == '*'。
# 分支条件：char == ';'。
# 遍历边界：(i, char) ← enumerate(completion)。
# 返回路径：''。
# 返回路径：result。
# 返回路径：original。
# 直接调用：completion.strip。
# 直接调用：enumerate。
# 直接调用：len。
# 直接调用：completion[:i + 1].strip。
# [/maint-note:a94d1cdaf702]
def process_c_completion(completion, add_log=False):
    if not completion:
        return ""

    original = completion.strip()

    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    for i, char in enumerate(completion):
        if char == '\\' and not escaped:
            escaped = True
            continue

        if escaped:
            escaped = False
            continue

        if in_line_comment:
            if char == '\n':
                in_line_comment = False
            continue

        if in_block_comment:
            if char == '*' and i + 1 < len(completion) and completion[i + 1] == '/':
                in_block_comment = False
                i += 1
            continue

        if in_string:
            if char == '"':
                in_string = False
            continue

        if in_char:
            if char == '\'':
                in_char = False
            continue

        if char == '/' and i + 1 < len(completion):
            if completion[i + 1] == '/':
                in_line_comment = True
                i += 1
                continue
            elif completion[i + 1] == '*':
                in_block_comment = True
                i += 1
                continue

        if char == '"':
            in_string = True
            continue

        if char == '\'':
            in_char = True
            continue

        if char == ';':
            result = completion[:i + 1].strip()
            # if add_log and len(result) < len(original):
            #     print(f"截断: '{original}' -> '{result}'")
            return result

    return original


# [maint-note:17e9aeebbd2d]
# 维护导航：generate_completion_batch；下列为本函数源码中的接口与路径索引。
# 输入 model：类型由调用方与函数体约束。
# 输入 tokenizer：类型由调用方与函数体约束。
# 输入 prompts：类型由调用方与函数体约束。
# 输入 max_to_generate：类型由调用方与函数体约束。
# 分支条件：not prompts。
# 遍历边界：(i, output) ← enumerate(outputs)。
# 遍历边界：prompt ← prompts。
# 返回路径：[]。
# 返回路径：generations。
# 捕获异常：Exception。
# 直接调用：torch.device。
# 直接调用：torch.cuda.is_available。
# 直接调用：tokenizer(prompts, padding=True, return_tensors='pt').to。
# 直接调用：tokenizer。
# 直接调用：torch.no_grad。
# 直接调用：model.generate。
# 直接调用：enumerate。
# 直接调用：tokenizer.decode。
# 直接调用：process_c_completion。
# 直接调用：generations.append。
# 直接调用：print。
# 直接调用：tokenizer(prompt, return_tensors='pt').to。
# 表达式分流：torch.cuda.is_available()；真值与假值分支分别计算，不同时执行。
# 资源作用域：torch.no_grad()；退出该 with 时调用对应上下文清理。
# [/maint-note:17e9aeebbd2d]
def generate_completion_batch(model, tokenizer, prompts, max_to_generate):
    if not prompts:
        return []

    # 对于分片模型，统一把输入放到 cuda:0（逻辑 GPU0，对应物理 1/2/4/7 中的第一个）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        encoded_inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask,
                max_new_tokens=max_to_generate,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generations = []
        for i, output in enumerate(outputs):
            input_length = encoded_inputs.input_ids[i].shape[0]
            generated_text = tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            )
            # 评估阶段不需要打印截断日志
            processed_text = process_c_completion(generated_text, add_log=False)
            generations.append(processed_text)

        return generations

    except Exception as e:
        print(f"生成过程中发生错误: {e}")
        print("尝试单个样本处理...")
        generations = []
        for prompt in prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_to_generate,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                processed_text = process_c_completion(generated_text, add_log=False)
                generations.append(processed_text)
            except Exception as e2:
                print(f"单个样本处理失败: {e2}")
                generations.append("")
        return generations


# [maint-note:d6b4495fadf7]
# 维护导航：compute_exact_match；下列为本函数源码中的接口与路径索引。
# 输入 prediction：类型由调用方与函数体约束。
# 输入 ground_truth：类型由调用方与函数体约束。
# 返回路径：1 if prediction.strip() == ground_truth.strip() else 0。
# 直接调用：prediction.strip。
# 直接调用：ground_truth.strip。
# 表达式分流：prediction.strip() == ground_truth.strip()；真值与假值分支分别计算，不同时执行。
# [/maint-note:d6b4495fadf7]
def compute_exact_match(prediction, ground_truth):
    return 1 if prediction.strip() == ground_truth.strip() else 0


# [maint-note:66ba9d6c0057]
# 维护导航：compute_edit_similarity；下列为本函数源码中的接口与路径索引。
# 输入 prediction：类型由调用方与函数体约束。
# 输入 ground_truth：类型由调用方与函数体约束。
# 分支条件：max_len == 0。
# 返回路径：1.0。
# 返回路径：similarity。
# 直接调用：prediction.strip。
# 直接调用：ground_truth.strip。
# 直接调用：Levenshtein.distance。
# 直接调用：max。
# 直接调用：len。
# [/maint-note:66ba9d6c0057]
def compute_edit_similarity(prediction, ground_truth):
    p = prediction.strip()
    g = ground_truth.strip()
    edit_distance = Levenshtein.distance(p, g)
    max_len = max(len(p), len(g))

    if max_len == 0:
        return 1.0

    similarity = 1.0 - (edit_distance / max_len)
    return similarity


# [maint-note:e7329f3d4db4]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 分支条件：not os.path.exists(DS_FILE)。
# 分支条件：not os.path.exists(PT_FILE)。
# 分支条件：valid_prompts。
# 分支条件：printed_samples < max_print_samples。
# 分支条件：prompt_pred。
# 分支条件：raw_em == 0 and prompt_em == 1。
# 分支条件：improved_samples。
# 遍历边界：i ← tqdm(range(0, num_samples, batch_size), desc='处理批次')。
# 遍历边界：(idx, id_) ← enumerate(valid_ids)。
# 遍历边界：(j, sample_id) ← enumerate(batch_ids)。
# 返回路径：无返回值。
# 写入目标：prompt_preds_map[id_]。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：load_config。
# 直接调用：print。
# 直接调用：os.path.exists。
# 直接调用：load_jsonl。
# 直接调用：item.get。
# 直接调用：load_model_and_tokenizer。
# 直接调用：len。
# 直接调用：tqdm。
# 直接调用：range。
# 直接调用：sample.get。
# 直接调用：prompt_dict.get。
# 直接调用：generate_completion_batch。
# 直接调用：enumerate。
# 直接调用：prompt_preds_map.get。
# 直接调用：compute_exact_match。
# 直接调用：compute_edit_similarity。
# 直接调用：raw_exact_matches.append。
# 直接调用：raw_edit_similarities.append。
# 直接调用：dataset_dict.get。
# 直接调用：sample_data.get。
# 直接调用：improved_samples.append。
# 直接调用：prompt_exact_matches.append。
# 直接调用：prompt_edit_similarities.append。
# 直接调用：results.append。
# 直接调用：np.mean。
# 直接调用：open。
# 直接调用：json.dump。
# 直接调用：f.write。
# 集合派生：item 来自 prompts。
# 集合派生：item 来自 dataset。
# 集合派生：sample 来自 current_batch。
# 集合派生：id_ 来自 batch_ids。
# 集合派生：p 来自 batch_prompts；过滤 p。
# 集合派生：(idx, p) 来自 enumerate(batch_prompts)；过滤 p。
# 集合派生：idx 来自 valid_indices。
# 表达式分流：prompt_exact_matches；真值与假值分支分别计算，不同时执行。
# 表达式分流：prompt_edit_similarities；真值与假值分支分别计算，不同时执行。
# 资源作用域：open(RESULT_FILE, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(EVAL_FILE, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 资源作用域：open(IMP_FILE, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:e7329f3d4db4]
def main():
    parser = argparse.ArgumentParser(description=f"评估{MODEL}模型的代码补全性能")
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

    # 用来控制打印前多少个样本的详细信息
    max_print_samples = 10
    printed_samples = 0

    for i in tqdm(range(0, num_samples, batch_size), desc="处理批次"):
        current_batch = dataset[i:i + batch_size]
        batch_ids = [sample.get("id", "") for sample in current_batch]
        batch_inputs = [sample.get("input", "") for sample in current_batch]
        batch_gts = [sample.get("gt", "") for sample in current_batch]

        batch_prompts = [prompt_dict.get(id_, "") for id_ in batch_ids]

        # 1. 原始输入预测
        raw_preds = generate_completion_batch(
            model, tokenizer, batch_inputs, max_to_generate
        )

        # 2. 带 prompt 的预测
        valid_prompts = [p for p in batch_prompts if p]
        prompt_preds_map = {}

        if valid_prompts:
            valid_indices = [idx for idx, p in enumerate(batch_prompts) if p]
            valid_ids = [batch_ids[idx] for idx in valid_indices]

            prompt_preds = generate_completion_batch(
                model, tokenizer, valid_prompts, max_to_generate
            )

            for idx, id_ in enumerate(valid_ids):
                prompt_preds_map[id_] = prompt_preds[idx]

        # 3. 计算指标 & 记录结果
        for j, sample_id in enumerate(batch_ids):
            gt = batch_gts[j]
            raw_pred = raw_preds[j]
            prompt_pred = prompt_preds_map.get(sample_id, "")

            raw_em = compute_exact_match(raw_pred, gt)
            raw_es = compute_edit_similarity(raw_pred, gt)

            raw_exact_matches.append(raw_em)
            raw_edit_similarities.append(raw_es)

            # 打印前若干个样本的详细补全情况
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
                    improved_sample = {
                        "id": sample_id,
                        "pkg": sample_data.get("pkg", ""),
                        "fpath": sample_data.get("fpath", ""),
                        "input": sample_data.get("input", ""),
                        "raw_res": raw_pred,
                        "prompt_res": prompt_pred,
                        "gt": gt
                    }
                    improved_samples.append(improved_sample)

                # 只在 prompt 比 raw 好的时候打印一条简要提示（方便你观察提升的案例）
                    print("✨ 提示增强带来改进的样本:", sample_id)
                    print("  Raw   =", raw_pred)
                    print("  Prompt=", prompt_pred)
                    print("  GT    =", gt)

                prompt_exact_matches.append(prompt_em)
                prompt_edit_similarities.append(prompt_es)

            result = {
                "id": sample_id,
                "raw_res": raw_pred,
                "prompt_res": prompt_pred,
                "gt": gt
            }
            results.append(result)

    # 汇总指标
    avg_raw_exact_match = np.mean(raw_exact_matches)
    avg_raw_edit_similarity = np.mean(raw_edit_similarities)
    avg_prompt_exact_match = np.mean(prompt_exact_matches) if prompt_exact_matches else 0.0
    avg_prompt_edit_similarity = np.mean(prompt_edit_similarities) if prompt_edit_similarities else 0.0

    # 保存详细结果
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存评估指标
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        f.write("评估结果汇总\n")
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
