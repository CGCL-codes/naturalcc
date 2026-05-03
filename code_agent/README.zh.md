# NaturalCC Code Agent

[English README](README.md)

`code_agent` 是一个本地代码编辑代理，把静态项目理解能力和 Aider 的文件修改能力组合在一起。

它会先解析目标项目，收集函数、变量、类型、成员、include 和符号关系；再根据用户任务生成语义增强 prompt；最后把 prompt 交给 Aider，由 Aider 修改选中的目标文件。

项目提供两种使用方式：

- 图形界面：FastAPI 后端 + React/Vite 前端。
- 命令行：`aider_runner.py`。

注意：第一个目标文件始终是 NaturalCC 构造 prompt 时使用的主解析文件。Aider 仍然可以接收并修改多个目标文件。

## 项目是什么

这不是通用聊天机器人，而是面向项目上下文的代码补全和代码修改代理。

典型任务：

- 补全函数体
- 补全函数签名
- 补全变量、成员或类型
- 按项目现有风格做小范围代码修改
- 检测潜在漏洞并按需自动修复
- 在执行 Aider 前预览最终语义 prompt

主要文件：

- `agent_web_api.py`：FastAPI 后端，同时可服务打包后的前端。
- `webui/`：React + Vite 图形界面。
- `aider_runner.py`：CLI 入口和 Aider 调度逻辑。
- `completion_prompt_agent.py`：语义 prompt 构造逻辑。
- `rag/c/`：C/C++ 解析和上下文检索。
- `rag/java/`：Java 解析和 prompt 路径。
- `test_api.py`：OpenRouter key / 连通性检查脚本。

## 环境搭建

### 前置条件

- [uv](https://docs.astral.sh/uv/getting-started/installation/)（Python 包管理器）
- Node.js & npm（前端构建）

### 1. 安装系统依赖

C/C++ 解析需要系统级 `libclang` 库。在运行 `uv sync` 之前，通过系统包管理器安装：

- **Ubuntu/Debian**: `sudo apt install libclang1`
- **macOS**: `brew install libclang`
- **其他发行版**: 在系统包仓库中搜索 `libclang` 并安装

### 2. 创建 Python 环境

在 `code_agent/` 目录下执行：

```bash
uv sync
```

此命令会自动创建 `.venv` 虚拟环境并安装所有锁定的 Python 依赖，无需手动配置 conda 或执行 pip。

项目运行至少需要以下能力（均由 `uv sync` 自动处理）：

- `fastapi`
- `uvicorn`
- `clang` Python bindings
- `aider` 可执行命令在 `PATH` 中

如需 GPU 支持（例如运行基于 vLLM 的离线评估），可手动安装：

```bash
uv pip install vllm
```

调用 OpenRouter / OpenAI 时，可以在界面或 CLI 中传入 API Key，也可以设置环境变量：

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENAI_API_KEY=sk-...
```

### 3. 安装前端依赖

在 `code_agent/` 目录下执行：

```bash
cd webui
npm install
```

## 使用图形界面

图形界面由 FastAPI 后端和 React 前端组成。

### 一键启动

如果你已安装图形终端模拟器（gnome-terminal、konsole、alacritty 等），或已安装 `tmux` 作为回退：

```bash
./start.sh
```

该脚本会自动使用 `.venv` 中的 Python，并打开两个终端窗口（或 tmux 分屏）：
- 一个运行 FastAPI 后端
- 一个运行 Vite 前端开发服务器

使用 tmux 时，session 名称为 `ncc-agent`。按 `Ctrl+B` 再按 `D` 可分离会话；重新 attach 用 `tmux attach -t ncc-agent`。

### 开发模式

适合修改前端代码时使用，支持 Vite 热更新。

终端 1：

```bash
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

终端 2：

```bash
cd webui
npm run dev
```

打开：

```text
http://127.0.0.1:5173/
```

开发模式下，Vite 在 `5173` 提供前端页面，并把 `/api/*` 请求代理到 `7860` 的 FastAPI 后端。

### 本地打包模式

适合只启动一个服务来同时提供 UI 和 API。

```bash
cd webui
npm run build
cd ..
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

打开：

```text
http://127.0.0.1:7860/
```

打包模式下，FastAPI 会服务 `webui/dist`，并在同一端口提供所有后端 API。

### UI 使用流程

1. 设置项目根目录。
2. 选择一个或多个目标文件。
3. 确认第一个目标文件是主解析文件。
4. 选择或输入模型名。
5. 可选填写 API Key、symbol、completion type 或 prefix。
6. 输入开发指令。
7. 点击 preview 预览最终 prompt，或点击 execute 调用 Aider 执行修改。

## 使用 CLI

在 `code_agent/` 目录下执行命令。

### 仅预览 Prompt

```bash
uv run python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "补全 foo 函数实现" \
  --preview
```

### 执行 Aider 修改文件

```bash
uv run python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "根据现有风格完善 foo 函数实现" \
  -m openrouter/deepseek/deepseek-chat
```

### 常用 CLI 参数

```bash
-dir /path/to/project
-f src/foo.c include/foo.h
-i "你的修改或补全需求"
-m openrouter/deepseek/deepseek-chat
-key sk-...
-s parse_flags
-t function_body
--prefix parse_
--preview
```

`-t` 支持：

```text
member
variable
function
function_body
type
```

## API 接口

`agent_web_api.py` 提供：

- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/models`
- `GET|POST /api/workspace/scan`
- `GET /api/browse`
- `POST /api/command-preview`
- `POST /api/prompt/preview`
- `POST /api/run`

`/api/run` 会返回按行分隔的 JSON 事件，前端用它实时显示 Aider 日志。

## Feature Plugin 系统

**Advanced** 面板现在由插件架构驱动。每个功能都是 `plugins/` 下的一个 `FeaturePlugin`。前端根据每个插件的 `config_schema` 动态渲染表单，因此添加新功能**不需要**修改任何前端代码。

### 架构

- `plugins/base.py` — `FeaturePlugin` 抽象基类、`ExecutionMode`（`aider`/`direct`/`hybrid`）、`ConfigField` 表单定义、`ExecutionContext`。
- `plugins/registry.py` — `@register_plugin` 类装饰器；插件在导入时自动注册。
- `plugins/dispatcher.py` — 将执行路由到 AIDER、DIRECT 或 HYBRID 模式。
- `plugins/code_completion.py` — 原有的 `symbol`/`completion_type`/`prefix` 逻辑，已迁移为插件。
- `plugins/code_summary.py` — NaturalCC + Aider dry-run 代码总结。
- `plugins/code_repair.py` — AIDER 模式的代码修复提示词，用于 bug、编译错误和测试失败。
- `plugins/vulnerability_detection.py` — 漏洞分析插件，支持可选的 Aider 自动修复。

### 执行模式

| 模式 | 行为 | 示例 |
|------|------|------|
| `aider` | 生成 prompt → 调用 Aider → 修改代码文件或输出 dry-run 报告 | 代码补全、代码修复、代码总结 |
| `direct` | 直接分析 → 返回报告 / 写入文件 | 静态报告 |
| `hybrid` | 通过 API 分析 → 生成修复 prompt → Aider 修复 | 漏洞检测 |

### 内置代码总结功能

功能名：`code_summary`（AIDER 模式）

执行方式：
- 对选中的目标文件，或项目下的源码文件，构造正常的 NaturalCC 语义 prompt。
- 使用 `--dry-run` 调用 Aider，因此总结过程不会修改文件。
- 使用所选模型生成更深入的代码理解报告。

主要配置项：
- `summary_scope`：`targets`（仅目标文件）或 `project`（全项目源码）
- `detail_level`：`brief` / `standard` / `detailed`
- `include_symbols`：要求 Aider 包含关键符号和数据流
- `max_files`：发送给 NaturalCC 和 Aider 的文件数量上限

### NaturalCC / libclang 版本对齐

NaturalCC 要求 Python `clang` bindings 与系统安装的 `libclang` 版本匹配。本项目固定 `clang==18.1.8`，对应 Ubuntu LLVM 18 / `libclang1-18` 系列。如果系统使用其他 LLVM 主版本，需要把 `clang` 依赖和锁文件调整为与 `libclang.so` 相同的主版本。

### 内置代码修复功能

功能名：`code_repair`（AIDER 模式）

执行方式：
- 根据用户指令、修复类型、可选错误日志和可选额外上下文生成聚焦修复提示词。
- 复用现有 NaturalCC 语义 prompt 路径，然后交给 Aider 修改目标文件。
- 默认偏向最小修复，并尽量保持现有接口不变。

主要配置项：
- `repair_type`：`bug_fix` / `compile_error` / `test_failure` / `safe_refactor`
- `failure_log`：编译错误、测试失败、堆栈或运行时报错
- `extra_context`：约束、期望行为或复现说明
- `allow_refactor`：必要时允许小范围辅助重构

### 内置漏洞检测功能

功能名：`vulnerability_detection`（HYBRID 模式）

执行方式：
- 阶段 1：进行基于规则的静态漏洞扫描并生成报告。
- 阶段 2（可选）：当 `auto_fix=true` 时，生成修复指令并调用 Aider 对目标文件进行修复。

主要配置项：
- `scan_scope`：`targets`（仅目标文件）或 `project`（全项目）
- `severity_threshold`：`low` / `medium` / `high` / `critical`
- `rule_profile`：`default` / `c_cpp` / `web`
- `auto_fix`：是否执行自动修复阶段
- `max_findings`：报告中最大告警条数
- `extra_instruction`：额外修复约束

使用建议：
- 如果要开启 `auto_fix`，先选择好目标文件。
- 建议先用 `auto_fix=false` 查看扫描结果，再决定是否自动修复。

### 如何添加新功能插件

1. 在 `plugins/` 下创建新文件，例如 `plugins/my_feature.py`。
2. 继承 `FeaturePlugin`，实现 `metadata`、`config_schema` 和 `execute`。
3. 用 `@register_plugin` 装饰类。
4. 重启后端。前端会自动显示新功能并渲染其表单。

示例：

```python
# plugins/my_feature.py
from typing import Any, Dict, Generator, List, Optional
from code_agent.plugins.base import (
    FeaturePlugin, FeatureMetadata, ExecutionMode,
    ConfigField, ConfigFieldType, ExecutionContext, PluginResult,
)
from code_agent.plugins.registry import register_plugin


@register_plugin
class MyFeaturePlugin(FeaturePlugin):

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="my_feature",           # 唯一标识
            label="My Feature",          # 显示名称
            description="功能描述",
            execution_mode=ExecutionMode.DIRECT,  # 或 AIDER / HYBRID
        )

    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="my_param",
                label="My Parameter",
                type=ConfigFieldType.TEXT,   # text / textarea / select / switch / file
                required=True,
                default="",
                placeholder="输入值",
                help_text="显示在字段下方的帮助文本",
            ),
        ]

    def execute(self, context: ExecutionContext) -> Generator[str, None, None]:
        # yield 字符串作为日志输出
        yield "开始执行...\n"
        # ... 你的业务逻辑 ...
        # 完成后 yield PluginResult（用于 DIRECT / HYBRID 模式）
        yield PluginResult(success=True, message="完成！")
```

### 配置字段类型

| 类型 | 渲染为 | 额外属性 |
|------|--------|---------|
| `text` | `<input type="text">` | `placeholder`, `default` |
| `textarea` | `<textarea>` | `placeholder`, `default` |
| `select` | `<select>` | `options: [{value, label}]`, `default` |
| `switch` | `<input type="checkbox">` | `default` (bool) |
| `file` | `<input type="file">` | `accept`, `multiple` |

### 文件上传

如果插件配置包含 `file` 类型字段，前端会自动以 `multipart/form-data` 发送请求。上传的文件在 `context.uploaded_files` 中以 `{field_name: UploadFile}` 的形式提供。

### 插件相关的 API 变更

`/api/bootstrap` 现在返回：

```json
{
  "features": [{"name": "...", "label": "...", "execution_mode": "..."}],
  "schemas": {"feature_name": [{"name": "...", "type": "...", ...}]},
  "default_feature": "code_completion"
}
```

`/api/run` 同时接受 JSON（向后兼容）和 `multipart/form-data`（用于文件上传）。请求体应包含：

```json
{
  "feature": "my_feature",
  "feature_config": {"my_param": "value"}
}
```

## 注意事项和限制

- C/C++ 解析依赖 `libclang`。
- 部分解析路径仍使用偏 C 语言的 libclang 参数，C++ 语法覆盖可能不完整。
- `rag/` 中包含离线研究和评测脚本，其中部分脚本带有本地路径假设。
- 当前项目主要依赖 smoke check，还没有正式自动化测试体系。
- `test_api.py` 只用于 API 连通性检查，不是 parser 或 UI 测试。
