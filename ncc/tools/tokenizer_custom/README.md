# 自定义 Tokenizer 项目说明

本项目为 **Tokenizer 自定义与微调工具箱**，支持对现有 tokenizer 的改造、训练新的 tokenizer 以及自定义 tokenizer 的实验与评估。项目结构清晰，模块化设计，适用于现代大模型训练场景，支持多卡并行、精度加速等优化。

### 项目目录结构

```text
├── data
├── evaluate
│   └── tokenizer_evaluate
│       ├── compute_metrics.sh       # 评估脚本（shell）
│       └── metrics.py               # 评估指标计算模块
├── experiments
│   └── tasks
│       ├── Code_Generation
│       │   ├── bleu.py
│       │   ├── cal_codebleu.py
│       │   ├── run_modern_mapping.py
│       │   ├── run_modern_mapping.sh
│       │   ├── run_modern.py
│       │   ├── run_modern.sh
│       │   └── utils.py
│       └── Code_Summarization
│           ├── bleu.py
│           ├── cal_codebleu.py
│           ├── run_modern_mapping.py
│           ├── run_modern_mapping.sh
│           ├── run_modern.py
│           ├── run_modern.sh
│           └── utils.py
├── modules
│   ├── token_affix
│   │   └── modify_mapping.py         # token 前后缀处理工具
│   ├── token_semantics
│   │   ├── extract-data-codegeneration.py
│   │   └── extract-data-codesummarization.py
│   │       # 提取 token 语义信息，支持代码生成与代码总结任务
│   └── vocabulary_transfer
│       ├── match.py                  # 词表匹配工具
│       ├── train_vocab.py            # 训练自定义词表
│       ├── transfer.py               # 词表迁移脚本
│       └── transfer.sh               # 自动化运行脚本
├── README.md                          # 项目说明文档
└── requirements.txt                   # Python 依赖包列表
```

###  目录说明
**`data`**

存放原始数据集与预处理数据，用于 tokenizer 训练和实验任务。

**`evaluate/tokenizer_evaluate`**

提供 tokenizer 评估工具，包括指标计算脚本和评估脚本，可用于验证 tokenizer 的覆盖率、语义保持能力等。

**`modules`**

核心工具模块，用于自定义 tokenizer 功能：

| 模块 | 功能 |
|------|------|
| `vocabulary_transfer` | 词表迁移、训练自定义词表和匹配工具 |
| `token_semantics`    | 提取 token 语义信息，支持代码生成与代码总结任务 |
| `token_affix`       | 处理 token 的前后缀映射，便于微调 tokenizer 结构 |

**`experiments/tasks`**

存放针对具体任务的实验脚本，经过现代化改造：

- **支持多卡并行训练**  
- **支持 FP16 / BF16 精度加速**

任务分类：

- **Code_Generation**：代码生成任务实验脚本  
- **Code_Summarization**：代码总结任务实验脚本

现代化训练脚本：
| 脚本 | 功能 |
|------|------|
| `run_modern_mapping.py / run_modern_mapping.sh` | 兼容现代训练框架的微调脚本 |
| `run_modern.py / run_modern.sh`                 | 标准训练脚本 |
| `bleu.py / cal_codebleu.py`                   | 评价指标计算工具 |
| `utils.py`                                     | 通用辅助函数 |


**`README.md`**

项目说明文档，提供目录结构、模块功能及使用说明。

**`requirements.txt`**

记录项目 Python 环境依赖



# 使用教程

## 环境配置

1. 创建虚拟环境

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```
>env_name 是你想要创建的环境的名称 
python=3.10 表示创建的环境将使用 Python 3.10 版本。你可以根据需要选择更高版本的 Python

2. 安装依赖包

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **注意：** 为保持依赖稳定，建议首先下载适合系统版本的torch