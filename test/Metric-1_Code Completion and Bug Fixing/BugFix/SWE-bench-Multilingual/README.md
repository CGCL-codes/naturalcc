# SWE-bench Multilingual 样本池（10 题）

来源：[官方数据集卡](https://huggingface.co/datasets/SWE-bench/SWE-bench_Multilingual)、[官方评测仓库](https://github.com/SWE-bench/SWE-bench)。

从 300 条官方 test split 中，先保证 C、C++、Go、Java、JavaScript、PHP、Ruby、Rust、TypeScript 各有一题，再按 gold patch 涉及的非测试源文件数、改动行数和 FAIL_TO_PASS 数量选每种语言的高复杂候选；第 10 题取剩余候选中分数最高者。该分数是本地筛选启发式，不是官方难度标签。

每个样本含 issue、仓库与 base commit、测试标识和官方容器镜像元数据。完整源仓库与 SWE-bench harness 需按官方环境准备；`patch` 和 `test_patch` 不导出到 agent 可见目录，避免泄露答案及新增测试。
