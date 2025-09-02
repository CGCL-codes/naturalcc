# 功能2：代码语言模型指令微调（Instruction Fine-tuning）

## 功能简介

主要功能是对代码语言模型进行指令微调，通过构建不同格式和复杂度的训练数据，以及多种高效微调方法，研究数据特征和微调策略对模型性能的影响。具体流程如下：

### 数据处理与增强

- 项目提供 `data_process` 模块，用于生成和处理指令微调训练数据。
- **CoT_complex_data_generate.py**：对带有复杂思维链（CoT）的训练数据进行简化，生成不同复杂度的思维链数据：
  - `normal_think_step`：正常复杂度
  - `simple_think_step`：简化后的思维链
- **remove_comment_util.py**：可选地去除代码注释信息，研究注释对微调效果的影响。
- 支持分批处理大规模数据，并自动保存为 JSONL 格式。

### 指令微调实验

- 使用 `InstructionTrainer` 类对基础模型进行指令微调。
- 实验主要包括：
  - 标准种子数据集微调
  - 去除注释的指令数据微调
  - 带有思维链信息的数据微调
  - 不同复杂度（简单 / 正常 / 复杂）的思维链数据微调
- 通过 YAML 配置文件（`configs/training/instruction_finetune.yaml`、`configs/training/peft.yaml`）管理训练参数。

### 高效微调方法比较

- 使用 `peft_compare.py` 比较不同高效微调方法（PEFT）对模型性能的影响：
  - LoRA
  - AdaLoRA
  - Prefix Tuning
  - P-tuning
  - Prompt Tuning
- 所有方法均使用相同的标准指令数据集进行微调，以保证可比性。

### 训练与实验流程

- 配置环境变量：
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/naturalcc/ncc"
```

- 运行微调脚本：
```bash
python instruction_data_compare.py   # 或 peft_compare.py
```
脚本会读取指定数据集，自动进行训练，并生成微调后的模型。
