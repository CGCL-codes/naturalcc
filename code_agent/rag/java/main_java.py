#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import signal
from argparse import ArgumentParser

from generator_java import JavaGenerator
from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR, PT_FILE, MODEL


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("处理超时")


def _norm_path(p: str) -> str:
    return p.replace("\\", "/")


def _resolve_abs_fpath(ds_repo_dir: str, pkg: str, fpath_in_ds: str) -> str:
    """
    dataset 的 fpath 可能是：
    - "pkg/src/.../A.java"
    - "src/.../A.java"
    这里统一转成绝对路径。
    """
    fp = _norm_path(fpath_in_ds).lstrip("/")
    if fp.startswith(_norm_path(pkg) + "/"):
        return os.path.join(ds_repo_dir, fp)
    return os.path.join(ds_repo_dir, pkg, fp)


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
    parser.add_argument("-f", "--file", default=PT_FILE, help="输出提示文件路径（jsonl，每行: {id,prompt}）")
    parser.add_argument("--dataset", default=None, help="数据集文件路径（jsonl），不指定则用 DS_FILE")
    parser.add_argument("-t", "--timeout", type=int, default=30, help="单个样本处理超时时间（秒）")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="批处理大小，每处理这么多样本保存一次结果")
    parser.add_argument(
        "--noctx_file", default=None,
        help="没检索到任何上下文(has_ctx=False)的样本输出文件（jsonl），默认: <file>.noctx"
    )
    args = parser.parse_args()

    print(f"使用模型: {args.model}")
    print(f"输出提示文件: {args.file}")
    print(f"数据集文件: {args.dataset}")
    print(f"单个样本处理超时时间: {args.timeout}秒")
    print(f"批处理大小: {args.batch_size}")

    noctx_file = args.noctx_file if args.noctx_file else args.file + ".noctx"
    print(f"无上下文样本输出文件: {noctx_file}")

    generator = JavaGenerator(DS_REPO_DIR, DS_GRAPH_DIR, args.model.lower())

    dataset_file = args.dataset if args.dataset else DS_FILE
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

    ret = []               # 正常 prompt 输出缓冲
    noctx_samples = []     # 无上下文样本缓冲
    timeout_samples = []   # 超时样本缓冲

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
            _append_jsonl(noctx_file, noctx_samples)
            ret = []
            noctx_samples = []

        pkg = item["pkg"]
        abs_fpath = _resolve_abs_fpath(DS_REPO_DIR, pkg, item["fpath"])

        try:
            if abs_fpath.endswith(".java"):
                signal.alarm(args.timeout)

                start_time = time.time()

                # 期望 generator 返回 (prompt_text, has_ctx)
                # 为了兼容未修改 generator 的情况，做一个兜底
                out = generator.retrieve_prompt(pkg, abs_fpath, item["input"])
                try:
                    prompt_text, has_ctx = out
                except Exception:
                    prompt_text, has_ctx = out, (True if (out and str(out).strip()) else False)

                signal.alarm(0)

                process_time = time.time() - start_time
                if process_time > 5:
                    print(f"样本 {i} 处理时间较长: {process_time:.2f}秒")

                ret.append({
                    "id": item.get("id", i + 1),
                    "prompt": prompt_text
                })
                processed_cnt += 1

                if not has_ctx:
                    noctx_samples.append({
                        "id": item.get("id", i + 1),
                        "pkg": pkg,
                        "fpath": item.get("fpath", ""),
                        "abs_fpath": abs_fpath
                    })

            else:
                skipped_cnt += 1

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
    _append_jsonl(noctx_file, noctx_samples)

    if timeout_samples:
        timeout_file = args.file + ".timeout"
        # 覆盖写 timeout（也可改成 append）
        os.makedirs(os.path.dirname(timeout_file) or ".", exist_ok=True)
        with open(timeout_file, "w", encoding="utf-8") as f:
            for it in timeout_samples:
                json.dump(it, f, ensure_ascii=False)
                f.write("\n")
        print(f"超时样本信息已保存到 {timeout_file}")

    print("=== 完成 ===")
    print(f"成功生成 prompt 数: {processed_cnt}")
    print(f"跳过非Java样本数: {skipped_cnt}")
    print(f"失败样本数: {failed_cnt}")
    print(f"超时样本数: {len(timeout_samples)}")
    print(f"无上下文样本数: {sum(1 for _ in open(noctx_file, 'r', encoding='utf-8')) if os.path.exists(noctx_file) else 0}")
    print(f"结果已保存到 {args.file}")
    print(f"无上下文样本已保存到 {noctx_file}")
