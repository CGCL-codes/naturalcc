#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import signal
from argparse import ArgumentParser

from .generator import CGenerator
from .utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR, PT_FILE, MODEL


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("处理超时")


def _append_jsonl(path: str, items: list):
    """Append a list of json-serializable dicts to jsonl file."""
    if not items:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            json.dump(it, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", default=MODEL,
        help="代码模型，支持: deepseekcoder, codegen, codegen25, santacoder, starcoder, codellama, gpt35, gpt4"
    )
    parser.add_argument("-f", "--file", default=PT_FILE, help="输出提示文件路径（jsonl）")
    parser.add_argument("-c", "--c_dataset", default=None, help="C语言数据集文件路径，不指定则使用默认路径")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="单个样本处理超时时间（秒）")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="批处理大小，每处理这么多样本保存一次结果")
    args = parser.parse_args()

    print(f"使用模型: {args.model}")
    print(f"输出提示文件: {args.file}")
    print(f"C语言数据集文件: {args.c_dataset}")
    print(f"单个样本处理超时时间: {args.timeout}秒")
    print(f"批处理大小: {args.batch_size}")

    generator = CGenerator(DS_REPO_DIR, DS_GRAPH_DIR, args.model.lower())

    dataset_file = args.c_dataset if args.c_dataset else DS_FILE
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    print(f"总共有 {len(dataset)} 个样本待处理")

    # 断点续跑：输出文件已有多少行
    start_idx = 0
    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            processed = len([ln for ln in f if ln.strip()])
            if processed > 0:
                start_idx = processed
                print(f"检测到 {args.file} 已处理 {processed} 个样本，从第 {start_idx + 1} 个样本继续处理")

    # ✅ 关键修复：如果已经处理完，直接退出（避免 i 未定义）
    if start_idx >= len(dataset):
        print(f"检测到 {args.file} 已处理完全部 {len(dataset)} 个样本，无需继续，退出。")
        exit(0)

    ret = []
    timeout_samples = []

    signal.signal(signal.SIGALRM, timeout_handler)

    processed_cnt = 0
    skipped_cnt = 0
    failed_cnt = 0

    for i, item in enumerate(dataset[start_idx:], start=start_idx):
        if i % 10 == 0:
            print(f"正在处理第 {i}/{len(dataset)} 个样本...")

        # batch flush
        if i > start_idx and (i - start_idx) % args.batch_size == 0:
            print(f"正在保存批处理结果... 已完成 {i} 个样本")
            _append_jsonl(args.file, ret)
            ret = []

        fpath = os.path.join(DS_REPO_DIR, item["fpath"])

        try:
            if fpath.endswith(".c") or fpath.endswith(".h"):
                signal.alarm(args.timeout)

                start_time = time.time()
                prompt_text = generator.retrieve_prompt(item["pkg"], fpath, item["input"])
                signal.alarm(0)

                process_time = time.time() - start_time
                if process_time > 5:
                    print(f"样本 {i} 处理时间较长: {process_time:.2f}秒")

                ret.append({
                    "id": item.get("id", i + 1),
                    "prompt": prompt_text
                })
                processed_cnt += 1

            else:
                skipped_cnt += 1
                # 如果你不想刷屏可以注释掉下一行
                print(f"跳过非C语言文件: {fpath}")

        except TimeoutException:
            print(f"警告: 处理样本 {i}, 文件 {item['fpath']} 超时，已跳过")
            timeout_samples.append({"id": item.get("id", i + 1), "fpath": item["fpath"]})
            signal.alarm(0)
            continue

        except Exception as e:
            failed_cnt += 1
            print(f"处理样本 {i}, 文件 {item['fpath']} 时出错: {repr(e)}")
            signal.alarm(0)
            continue

    # 写最后残留
    _append_jsonl(args.file, ret)

    # 写超时文件（覆盖写，也可以改 append）
    if timeout_samples:
        timeout_file = args.file + ".timeout"
        os.makedirs(os.path.dirname(timeout_file) or ".", exist_ok=True)
        with open(timeout_file, "w", encoding="utf-8") as f:
            for it in timeout_samples:
                json.dump(it, f, ensure_ascii=False)
                f.write("\n")
        print(f"超时样本信息已保存到 {timeout_file}")

    print("=== 完成 ===")
    print(f"成功生成 prompt 数: {processed_cnt}")
    print(f"跳过非C样本数: {skipped_cnt}")
    print(f"失败样本数: {failed_cnt}")
    print(f"超时样本数: {len(timeout_samples)}")
    print(f"结果已保存到 {args.file}")
