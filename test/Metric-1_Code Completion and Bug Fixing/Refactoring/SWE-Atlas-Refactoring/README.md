# SWE-Atlas Refactoring 样本池（10 题）

来源：[官方 benchmark 仓库](https://github.com/scaleapi/swe-atlas)。官方 Refactoring 集合有 70 个任务，任务元数据标注为 hard。这里从官方任务中优先覆盖不同语言标签，再按官方 gold patch 所涉及文件数、改动行数和 instruction 长度排序选取。分数仅用于本次抽样，不是官方 ranking。

每个样本保留官方 instruction、task id、语言标签、相关文件/测试文件名和运行镜像标识；不复制 gold patch、rubrics 或测试脚本。正式计分应通过 Harbor 和 SWE-Atlas 官方 sandbox/validator 执行。当前文件夹是 agent/UI 的任务卡，不是完整项目 checkout。
