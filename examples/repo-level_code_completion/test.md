# 仓库级别代码补全测试
## 目录结构
repo-level_code_completion/
├── config.yaml                # accelerate配置文件
├── custom_generate.py         # 自定义生成逻辑
├── data_merge.py             # 数据合并工具
├── eval.py                   # 评估主程序
├── eval_metric.py            # 评估指标计算
├── eval_utils.py             # 评估工具函数
├── merged_line_completion.jsonl # 合并后的测试数据
├── repofuse-main/            # 主实验结果的输出目录
│   └── Qwen2.5-Coder-7B-Instruct/ # 特定模型的评估结果
├── run_batch_experiments.py  # 实验运行脚本
└── test.md                   # 测试文档

## 运行评估
python run_batch_experiments.py --task repofuse-main

## 评估指标
- Exact Match (EM)
- Edit Similarity (ES)
- Identifier Match (IM)