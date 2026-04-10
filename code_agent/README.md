# code_agent 说明文档

## 1. 这个目录在做什么

`code_agent` 是一个把 **NaturalCC 的静态语义解析能力** 和 **Aider 的代码修改能力** 组合起来的代码代理实现，目标是让大模型在修改 C/C++ 项目代码时，不只是看当前片段，而是先理解项目里的函数、变量、结构体、头文件关系，再生成更贴近真实工程上下文的 prompt，最后交给 Aider 执行修改。

从代码职责上看，它可以分成两层：

1. `agent_ui.py`、`aider_runner.py`、`completion_prompt_agent.py`
   这部分是“可交互的 agent 主链路”，负责收集用户输入、构造 prompt、调用 Aider。
2. `rag/`
   这部分是“项目解析与提示增强底座”，负责解析 C/C++ 项目、抽取符号关系、拼接检索上下文，以及配套的离线评测脚本。

换句话说，这个目录实现的并不是一个通用聊天 agent，而是一个更偏向 **C/C++ 项目代码补全、函数实现补全、重构辅助** 的上下文增强代理。

## 2. 核心执行流程

主链路可以概括为：

1. 用户在 Gradio 页面或命令行里输入需求、目标文件、模型、API Key。
2. `aider_runner.py` 负责规范化项目目录和目标文件路径。
3. `CompletionPromptAgent` 调用 `rag/preprocess.py` 对项目做静态解析。
4. 解析器基于 `libclang` 提取函数、变量、结构体、typedef、include 和符号关系。
5. `CompletionPromptAgent` 根据用户意图推断补全类型：
   - 成员补全
   - 变量补全
   - 函数签名补全
   - 函数体补全
   - 类型补全
6. 生成的项目语义 prompt 会被包进一段更高层的系统指令里。
7. `aider_runner.py` 把最终 prompt 写入临时文件，并调用 `aider --message-file ...`。
8. Aider 根据这个上下文去修改目标文件。

其中一个很关键的实现细节是：

- Aider 可以接收多个目标文件。
- 但当前 prompt 构造时，**第一个目标文件** 会被当作主要解析目标文件，用来推断符号和生成上下文。

## 3. 主模块解读

### 3.1 `agent_ui.py`

这是前端入口，使用 Gradio 搭了一个本地页面，核心职责包括：

- 顶部提供统一的软件抬头，下方采用左右双栏布局：
  - 左栏负责项目与模型配置，内部分为“常规配置”和“工程概览 / 进阶控制”两个按钮面板
  - 右栏负责任务与执行，内部分为“开发指令”“命令行生成内容”“操作说明 / 状态”三个按钮面板
- 提供项目目录选择、目标文件多选、模型选择、API Key 输入，并对目录、目标文件、模型、Key 回退行为做动态提示。
- 提供高级选项：`symbol`、`completion_type`、`prefix`，同时展示项目概览、主解析文件和当前控制摘要。
- 支持三类任务动作：
  - 仅预览 prompt
  - 直接运行 agent 并修改代码
  - 清空右侧任务内容但保留左侧配置
- 在“命令行生成内容”面板中同时展示等效 CLI 命令草案和 prompt / Aider 日志。
- 在“操作说明 / 状态”面板中展示操作说明和实时状态摘要。
- 自动扫描项目文件并过滤部分噪音目录和扩展名，例如 `.git`、`__pycache__`、`node_modules`、`.log`、`.ps`。
- 控制台面板默认优先展示“命令行生成内容”，并把“等效命令行”放在右侧辅助区，滚动条默认隐藏、交互时再显现。

这个文件本质上是 UI 壳层，真正执行逻辑都委托给 `aider_runner.py`。

### 3.1.1 `agent_ui_assets.py`

这个文件存放 `agent_ui.py` 的静态界面资源，方便后续单独维护样式与文案，当前主要包含：

- 顶部抬头 HTML
- 自定义 CSS
- 右侧“操作说明 / 状态”面板的说明文案

### 3.1.2 `agent_ui_bindings.py`

这个文件负责 Gradio 按钮和输入框的事件绑定，把主文件底部那一大段 `.click()` / `.change()` 注册逻辑收拢到一起，当前主要包含：

- 组件引用的数据结构
- 回调引用的数据结构
- 左右面板切换、配置刷新、任务预览/执行/清空等按钮的绑定逻辑

### 3.2 `aider_runner.py`

这是实际的调度中心，也是命令行入口。它做了几件关键事情：

- 缓存了一个全局 `PROMPT_AGENT`，避免同一进程里重复解析同一个项目。
- 规范化项目目录与目标文件路径。
- 根据模型名粗略判断 provider，例如 `openrouter` 或 `openai`。
- 生成最终 prompt，并包上一层“你是一个由 NaturalCC 驱动的高级代码智能体”的指令模板。
- 把 prompt 写入临时文件后调用 `aider`。
- 提供三种使用方式：
  - `run_aider_cli()`：命令行直接执行
  - `run_aider_stream()`：流式返回日志，供 Gradio 页面展示
  - `preview_prompt()`：只看 prompt，不实际执行

此外，这个文件对输出编码做了兼容处理，尤其照顾了中文日志和 Windows 管道输出。

### 3.3 `completion_prompt_agent.py`

这是“语义 prompt 编排器”，属于当前目录最核心的业务逻辑之一。它不是直接改代码，而是负责把项目解析结果变成适合 LLM/Aider 的提示词。

它的主要能力有：

- 载入并缓存项目解析结果。
- 从自然语言中自动推断目标符号，例如从“补全 parse_flags 函数”提取 `parse_flags`。
- 自动推断补全类型。
- 从解析结果中查找：
  - 当前文件里的符号
  - 某个变量的类型
  - 某个类型对应的结构体/联合体
  - 某个结构体的成员列表
  - 同文件相关函数
  - 项目内其他文件里的同名声明或定义
- 最终生成不同场景下的 prompt 模板。

和 `rag/completion_prompt_agent_origin.py` 相比，这个版本更完整，明显增加了：

- `function_body` 场景
- 从自然语言自动识别 `symbol`
- 更稳妥的上下文获取封装

## 4. `rag/` 子目录解读

`rag/` 是这个 agent 的“项目理解引擎”。如果不看这里，就只能把它理解成一个 Aider UI；但真正体现 NaturalCC 特征的内容，大都在这一层。

### 4.1 `preprocess.py`

定义了 `CProjectParser`，负责：

- 遍历项目目录中的 `.c/.cpp/.h/.hpp` 文件。
- 调用 `CParser` 逐个解析文件。
- 汇总成项目级 `parse_res`。
- 初始化 `CProjectSearcher`。
- 清洗符号关系，把一些无效的项目内引用剔除掉。

这个文件相当于项目级入口，负责把“很多源码文件”变成“可检索的语义图”。

### 4.2 `cfile_parse.py`

这是底层 AST 解析器，依赖 `clang.cindex` 与 `libclang`。它做的是最基础但最关键的工作：

- 自动尝试寻找 `libclang` 动态库。
- 解析 C 文件 AST。
- 提取并存储：
  - include
  - 全局变量
  - 局部变量
  - 函数定义/声明
  - 函数体
  - 结构体 / 联合体 / 枚举
  - typedef
  - docstring / 注释
  - 类型关系 `Typeof`
  - 赋值关系 `Assign`

最终输出是一个按符号名组织的结构化字典，供上层搜索和 prompt 生成使用。

### 4.3 `node_prompt.py`

这是项目级语义搜索器 `CProjectSearcher`，主要负责：

- 规范化 `parse_res` 的文件结构。
- 判断某个 include 是否是项目内头文件。
- 从某个符号出发，沿 `include` 和 `rels` 做 DFS。
- 把跨文件相关符号按近似拓扑顺序组织起来。
- 最终生成适合喂给模型的代码上下文片段。

可以把它理解为：

- `cfile_parse.py` 负责“抽取知识”
- `node_prompt.py` 负责“组织知识”

### 4.4 `generator.py`

这个模块更偏向离线 RAG prompt 构造，主要服务 `rag/main.py` 和评测流程。它会：

- 读取某个项目的已解析图文件。
- 从源码中抽取用户 include 的头文件。
- 找到对应头文件里的函数定义信息。
- 把这些上下文和原始代码拼接起来。
- 使用 `tokenizer.py` 控制长度，避免超过模型输入上限。

### 4.5 `tokenizer.py`

这是模型输入裁剪器，适配了多种代码模型与 GPT 系列，包括：

- CodeGen / CodeGen2.5
- SantaCoder
- StarCoder / StarCoder2
- CodeLlama
- DeepSeek-Coder
- Qwen2.5-Coder / Qwen3
- Llama3
- GPT-3.5 / GPT-4

核心职责是：

- 根据模型配置加载 tokenizer。
- 计算 token 长度。
- 估算 prompt 可用长度。
- 以“prompt + suffix + program”的形式做截断和拼接。

`tokenizer copy.py` 基本上是旧版本备份，保留了较早的实现，不属于当前主链路。

### 4.6 `main.py`

这是一个离线批处理入口，不是 UI agent 的主入口。它面向评测数据集，负责：

- 读取数据集 JSONL。
- 为每个样本调用 `CGenerator.retrieve_prompt()`。
- 支持超时控制、断点续跑、分批写出。

更像是“批量生成 prompt 数据”的脚本。

### 4.7 评测脚本

`rag/` 里有多份评测脚本，明显能看出这是一个持续试验中的研究型目录：

- `eval.py`
  使用 Hugging Face 模型做评测，对比 raw input 与 prompt-enhanced input。
- `evaluation.py`
  也是 HF 评测脚本，和 `eval.py` 高度相似，像是较早版本或保留版本。
- `eval_vllm.py`
  使用 vLLM 做双路评测。
- `eval_vllm3.py`
  三路评测：`raw`、`model prompt`、`langchain prompt`。

这些文件都不是在线 agent 的必要依赖，而是用于验证 prompt 增强是否真的提升代码补全效果。

### 4.8 其他辅助/实验文件

- `completion_prompt_agent_origin.py`
  旧版 prompt agent，保留了更早的补全逻辑。
- `json2neo4j.py`
  把解析结果导入 Neo4j，属于实验型脚本，而且路径和连接信息是硬编码的。
- `config.yaml`
  模型路径和 token 上限配置文件，里面包含大量本地路径，说明当前实现偏研究/本地实验环境。
- `utils.py`
  统一放了一批常量，例如数据集目录、结果输出目录、默认模型名、CUDA 可见卡配置等。

## 5. 目录结构概览

```text
code_agent/
├── __init__.py
├── agent_ui.py                      # Gradio 页面入口
├── agent_ui_assets.py               # UI 静态 HTML/CSS/说明文案
├── agent_ui_bindings.py             # UI 按钮/输入事件绑定
├── aider_runner.py                  # Aider 调度与命令行入口
├── completion_prompt_agent.py       # 当前主用的 Prompt 编排器
├── test_api.py                      # OpenRouter API 连通性测试
└── rag/
    ├── __init__.py
    ├── cfile_parse.py               # 基于 libclang 的 C/C++ AST 解析
    ├── preprocess.py                # 项目级解析与关系清洗
    ├── node_prompt.py               # 基于符号图的上下文拼接
    ├── generator.py                 # 离线 prompt 生成器
    ├── tokenizer.py                 # 模型 token 长度控制
    ├── tokenizer copy.py            # 旧版 tokenizer 备份
    ├── completion_prompt_agent_origin.py
    ├── main.py                      # 批量生成 prompt 的脚本入口
    ├── eval.py                      # HF 评测
    ├── evaluation.py                # 旧版/平行评测脚本
    ├── eval_vllm.py                 # vLLM 双路评测
    ├── eval_vllm3.py                # vLLM 三路评测
    ├── json2neo4j.py                # 图谱导入 Neo4j 的实验脚本
    ├── config.yaml                  # 模型与 token 配置
    └── utils.py                     # 数据集/结果/默认模型常量
```

## 6. 这个 agent 的特点

和纯粹把代码片段直接发给模型相比，这个 `code_agent` 的特点是：

- 它先理解项目，再让模型动手。
- 它特别针对 C/C++ 项目。
- 它不仅支持补全，还能驱动 Aider 直接修改文件。
- 它同时保留了研究实验脚本，说明这个目录既有“可用工具”的一面，也有“论文/实验代码”的一面。

当前最像“产品主链路”的文件是：

- `agent_ui.py`
- `aider_runner.py`
- `completion_prompt_agent.py`
- `rag/preprocess.py`
- `rag/cfile_parse.py`
- `rag/node_prompt.py`

## 7. 运行方式

### 启动 UI

```bash
python -m code_agent.agent_ui
```

或直接：

```bash
python code_agent/agent_ui.py
```

启动后页面会显示：

- 顶部紧凑抬头区，整体横向居中铺开
- 左侧“项目与模型配置”卡片：底部两个按钮切换“常规配置”和“工程概览 / 进阶控制”
- 右侧“任务与执行”卡片：底部三个按钮切换“开发指令”“命令行生成内容”“操作说明 / 状态”
- 右侧三个视图上方不再额外放统一“任务状态”条，状态信息收拢到各自视图内部
- 右侧执行动作触发后，会自动跳转到“命令行生成内容”面板展示结果
- 左右卡片底部按钮固定在卡片外层底部，正文改为独立滚动区，避免内容从按钮下方“透出来”
- 桌面端左右主卡片默认压在首屏内，并按浏览器宽高自适应收缩；超出的内容会在各自卡片主体区滚动，滚动条默认隐藏，交互时再显现
- “命令行生成内容”面板内部采用双栏：左侧优先放日志/输出，右侧放等效 CLI 命令草案
- `agent_ui.py` 负责主布局与核心逻辑，`agent_ui_assets.py` 负责抬头、CSS 和静态说明文案，`agent_ui_bindings.py` 负责事件绑定

### 命令行模式

```bash
python -m code_agent.aider_runner \
  -f src/main.c \
  -i "补全 parse_flags 函数的完整实现" \
  -m openrouter/deepseek/deepseek-chat
```

### 仅预览 prompt

```bash
python -m code_agent.aider_runner \
  -f src/main.c \
  -i "补全 parse_flags 函数的完整实现" \
  --preview
```

## 8. 依赖与注意事项

从当前代码看，运行这套 agent 至少要注意下面几点：

- 需要安装并可执行 `aider`。
- 需要 `libclang`，否则 `rag/cfile_parse.py` 无法工作。
- 需要对应 Python 依赖，如 `gradio`、`clang`、`transformers`、`torch`、`tiktoken`、`yaml`、`attridict`、`requests`、`vllm`、`python-Levenshtein`、`py2neo` 等。
- `rag/config.yaml` 和 `rag/utils.py` 中包含大量本地实验路径，开箱即用性一般，需要按本机环境调整。
- `rag/utils.py` 里直接设置了 `CUDA_VISIBLE_DEVICES`，这会影响运行环境。
- `json2neo4j.py` 中 Neo4j 地址、账号和输入文件路径都是硬编码的。
- 这套解析链目前明显是围绕 C/C++ 设计的，不是通用多语言 agent。

## 9. 总结

如果把这个目录用一句话概括，可以说：

> 这是一个面向 C/C++ 工程的、基于静态解析增强上下文的代码代理，它用 NaturalCC 理解项目结构，用 Prompt Agent 组织上下文，再借助 Aider 实际落地代码修改。
