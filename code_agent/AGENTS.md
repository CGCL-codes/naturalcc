# Agent Init

## 1. 项目定位

`code_agent` 不是通用聊天机器人，而是一个面向上下文增强代码代理。它把两部分能力拼在一起：

- `NaturalCC` 风格的静态语义解析：先解析项目里的函数、变量、结构体、include 和符号关系。
- `Aider` 的代码修改能力：把整理好的语义 prompt 交给大模型智能体Aider，再驱动目标文件修改。

仓库当前主要解决的任务是：

- 函数体补全
- 函数签名补全
- 变量补全
- 成员补全
- 类型相关补全
- 基于已有工程风格的代码修改/重构辅助

## 2. 主入口

- `agent_ui.py`
  Gradio 页面入口。负责项目目录选择、目标文件多选、模型/API Key 输入、prompt 预览和流式运行。

- `aider_runner.py`
  实际调度中心，也是命令行入口。负责路径归一化、prompt 生成、Aider 调用、日志输出和 provider 推断。

- `completion_prompt_agent.py`
  当前主用的 prompt 编排器。它会触发项目解析、推断 symbol 与 completion type，并把解析结果组织成适合 LLM/Aider 的提示词。

- `rag/`
  项目语义理解底座。主链路依赖 `preprocess.py`、`cfile_parse.py`、`node_prompt.py`；其余很多文件是离线生成或评测脚本。

## 3. 真实执行链路

在线 agent 主链路如下：

1. 用户从 UI 或 CLI 输入 `project_dir`、`target_files`、`instruction`、`model`、`api_key`。
2. `aider_runner.py` 规范化项目目录和目标文件路径。
3. `aider_runner.generate_completion_prompt()` 取第一个目标文件作为主解析文件，并把相对路径传给 `CompletionPromptAgent`。
4. `CompletionPromptAgent` 在项目首次加载或 `project_dir` 变化时，调用 `CProjectParser.parse_dir()` 解析整个项目。
5. `CProjectParser` 遍历项目中的 `.c/.cpp/.h/.hpp` 文件，逐个调用 `CParser.parse()` 做 AST 抽取。
6. `CProjectSearcher` 基于 `include` 和 `rels` 关系把相关定义拼成 prompt 片段。
7. `CompletionPromptAgent` 根据场景生成成员/变量/函数/函数体/类型 prompt。
8. `aider_runner.py` 再包一层高层系统指令，写入临时 `message-file`。
9. 最后执行 `aider <target_files...> --message-file <tmp>`，Aider 负责实际修改文件。

需要注意：

- 可以给 Aider 传多个目标文件。
- 但 prompt 构造只会把第一个目标文件当作主要解析文件。

## 4. 核心数据结构

项目解析结果保存在 `parse_res`，结构大致如下：

```python
parse_res = {
    "src/foo.c": {
        "": {"type": "Module", "file_path": "...", "docstring": "..."},
        "foo": {
            "type": "Function",
            "def": "int foo(int x)",
            "body": "{ ... }",
            "sline": 12,
            "docstring": "...",
            "rels": [["int", "Typeof"]]
        },
        "bar": {
            "type": "Variable",
            "def": "int bar = foo();",
            "sline": 30,
            "rels": [["int", "Typeof"], ["foo", "Assign"]]
        },
        "node": {"type": "Struct", "def": "struct node ..."},
        "node.next": {
            "type": "Variable",
            "def": "struct node *next;",
            "in_struct": "node",
            "rels": [["struct node *", "Typeof"]]
        }
    }
}
```

字段语义：

- `type`: 常见值有 `Module`、`Function`、`Variable`、`Struct`、`Union`、`Enum`
- `def`: 定义或声明文本
- `body`: 函数体或类型体内容
- `sline`: 起始行号
- `docstring`: 注释
- `in_struct`: 表示结构体成员
- `in_function`: 表示局部变量属于哪个函数
- `include`: include 关系
- `rels`: 符号关系，目前主要是 `Typeof` 和 `Assign`

## 5. 各文件职责

### 顶层文件

- `agent_ui.py`
  纯 UI 壳层。扫描项目文件时会忽略 `.git`、`__pycache__`、`node_modules`、`datasets`、`.log`、`.ps` 等噪音内容。

- `aider_runner.py`
  关键点：
  - 全局缓存 `PROMPT_AGENT = CompletionPromptAgent()`，避免同进程内重复解析项目。
  - `detect_provider()` 根据模型名推断 `openrouter` 或 `openai`。
  - `build_aider_context_and_command()` 会生成最终 prompt、落临时文件，并拼出 Aider 命令。
  - `run_aider_stream()` 会做流式日志转发，并对 ANSI 控制符和中英文编码兼容处理。

- `completion_prompt_agent.py`
  当前主用 prompt agent。相较旧版，新增或强化了：
  - 自然语言自动抽取 `symbol`
  - `function_body` 场景
  - 更稳妥的上下文访问封装
  - 默认更偏向“函数实现补全”而不是“签名补全”

- `test_api.py`
  不是单元测试。它是 OpenRouter API 联通性和额度检查脚本，可选做一次 `max_tokens=1` 的最小生成验证。

### `rag/` 主链路文件

- `rag/preprocess.py`
  定义 `CProjectParser`。负责项目目录遍历、逐文件解析、构建 `proj_searcher`，以及清洗 `rels/include`。

- `rag/cfile_parse.py`
  基于 `clang.cindex` / `libclang` 做 AST 抽取。负责提取 include、变量、函数、结构体、union、enum、typedef、注释和关系边。

- `rag/node_prompt.py`
  定义 `CProjectSearcher`。负责：
  - `include` 归一化
  - 根据 `rels/include` 做 DFS
  - 近似拓扑排序相关文件
  - 以代码片段形式输出 prompt 上下文

- `rag/generator.py`
  离线 prompt 构造器。主要服务 `rag/main.py` 这条评测链路，不属于在线 agent 主入口。

- `rag/tokenizer.py`
  输入长度裁剪器。支持 CodeGen、SantaCoder、StarCoder、CodeLlama、DeepSeek-Coder、Qwen2.5-Coder、Qwen3、Llama3、GPT 系列。

### `rag/` 非主链路或实验文件

- `rag/main.py`
  离线批量 prompt 生成入口，面向数据集 JSONL，不是交互 agent 入口。

- `rag/eval.py`
- `rag/evaluation.py`
- `rag/eval_vllm.py`
- `rag/eval_vllm3.py`
  都是离线评测脚本，用于比较 raw prompt 与增强 prompt 的补全效果。

- `rag/completion_prompt_agent_origin.py`
  旧版 prompt agent，保留做参考，不是当前主用实现。

- `rag/tokenizer copy.py`
  旧版 tokenizer 备份。

- `rag/json2neo4j.py`
  实验脚本，含硬编码 Neo4j 连接和输入路径，不属于正常运行链路。

## 6. 代码里真实存在的限制和坑点

- 仓库宣称支持 C/C++，但 `rag/cfile_parse.py` 调 libclang 时固定使用 `-x c`，也就是当前底层解析是按 C 模式走的。目录扫描会收 `.cpp/.hpp`，但 C++ 语法覆盖并不稳。

- `CompletionPromptAgent._infer_completion_type()` 对含“补全 / 完善 / 实现”这类词的请求高度偏向 `function_body`。这能提升可用性，但也意味着很多模糊请求会被当成“补完整函数实现”处理。

- `CProjectParser._get_all_c_file_paths()` 会跳过隐藏文件/目录，并且只接受名称匹配 `[\w-]+` 的目录和文件 stem。含空格或特殊字符的路径可能直接不参与解析。

- `rag/utils.py` 会在导入时直接设置 `CUDA_VISIBLE_DEVICES="0,1,2,3"`。这对离线评测脚本影响较大。

- `rag/config.yaml` 含多处本地绝对路径和占位配置，不是开箱即用的通用配置。

- 当前仓库没有正式的自动化测试体系。`test_api.py` 是接口检查脚本，不覆盖 parser、searcher 或 prompt 生成逻辑。

- `aider_runner.py` 依赖外部 `aider` 可执行文件存在于 PATH 中，否则会直接报错。

- `rag/cfile_parse.py` 依赖 `libclang`，并会在导入阶段尝试自动定位。如果环境里没有 `libclang`，主链路无法工作。

## 7. 常见改动应该落在哪

- 想改 UI 行为、文件筛选、页面交互：看 `agent_ui.py`
- 想改 Aider 调用方式、provider 推断、CLI 参数：看 `aider_runner.py`
- 想改 prompt 模板、symbol 推断、completion type 推断：看 `completion_prompt_agent.py`
- 想改 AST 抽取字段、关系构建、注释抽取：看 `rag/cfile_parse.py`
- 想改项目遍历或关系清洗：看 `rag/preprocess.py`
- 想改跨文件检索、DFS 或上下文组织顺序：看 `rag/node_prompt.py`
- 想改离线数据集 prompt 生成：看 `rag/generator.py`、`rag/main.py`
- 想改离线评测逻辑：看 `rag/eval*.py`

## 8. 最小运行方式

### 运行前先激活环境
```bash
conda activate naturalcc
```

### 启动 UI

```bash
python agent_ui.py
```

默认会启动 Gradio，本地端口是 `7860`。

### 命令行预览 prompt

```bash
python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "补全 foo 函数实现" \
  --preview
```

### 命令行直接运行 Aider

```bash
python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "根据现有风格完善 foo 函数实现" \
  -m openrouter/deepseek/deepseek-chat
```

### 检查 OpenRouter Key

```bash
python test_api.py --generation-check
```

## 9. 给后续代理的工作建议

- 先判断任务属于哪条链路：在线交互链路，还是离线评测链路。不要一上来就在 `rag/eval*.py` 里改用户真正不会走到的逻辑。

- 如果用户反馈“生成的修改不对”或“prompt 上下文不准”，优先检查：
  - `completion_prompt_agent.py` 的类型推断
  - `rag/node_prompt.py` 的上下文检索
  - `rag/cfile_parse.py` 的关系抽取

- 如果用户反馈“找不到符号”或“某些文件没被解析”，优先检查：
  - 目标文件是不是第一个 `target_file`
  - 路径是否被标准化为相对项目根目录
  - 文件/目录名是否被 `_get_all_c_file_paths()` 的规则跳过
  - 当前代码是否因为 `-x c` 造成 C++ 解析失败

- 如果要做真正的工程化改进，优先级通常是：
  1. 去掉硬编码路径和环境变量
  2. 为 parser / prompt agent 补最小可运行测试
  3. 明确 C 和 C++ 的编译参数
  4. 把在线链路与离线实验脚本再拆干净

- 代码，运行方式，软件的改动和有信息价值（小细节不需要记录）最终同步到本AGENTS.md中，同时修改README.md中的对应内容。
