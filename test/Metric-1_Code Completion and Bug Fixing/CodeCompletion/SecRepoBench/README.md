# SecRepoBench 样本池（10 题）

来源：[官方实现与数据集](https://github.com/ai-sec-lab/SecRepoBench)、[官方数据集卡](https://huggingface.co/datasets/ai-sec-lab/SecRepoBench)。

按目标函数的非空行数、控制流和指针/索引操作复杂度排序，优先抽取不同 CWE 与不同项目；此为本地筛选启发式，不是官方难度标签。每题提供官方 masked function、描述提示和固定修复 commit 元数据。官方测评仍需在完整 C/C++ 仓库环境中编译，并运行开发者单测和 OSS-Fuzz PoC；此处的单函数 workspace 适合 UI 演示，不等同官方计分。

`TASK.md` 和 `workspace/` 是给 NaturalCC 的输入。官方修复补丁与隐藏测试未复制进样本目录。
