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

### 1. 激活 Python 环境

```bash
conda activate naturalcc
```

### 2. 安装 Python 依赖

项目运行至少需要：

- `fastapi`
- `uvicorn`
- `clang` Python bindings
- `libclang`
- `aider` 可执行命令在 `PATH` 中

如果当前环境缺少依赖，可按你的本地环境策略安装。常见方式：

```bash
pip install fastapi uvicorn clang aider-chat
conda install -c conda-forge libclang
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

### 开发模式

适合修改前端代码时使用，支持 Vite 热更新。

终端 1：

```bash
python agent_web_api.py --host 127.0.0.1 --port 7860
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
python agent_web_api.py --host 127.0.0.1 --port 7860
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
python aider_runner.py \
  -dir /path/to/project \
  -f src/foo.c include/foo.h \
  -i "补全 foo 函数实现" \
  --preview
```

### 执行 Aider 修改文件

```bash
python aider_runner.py \
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

## 注意事项和限制

- C/C++ 解析依赖 `libclang`。
- 部分解析路径仍使用偏 C 语言的 libclang 参数，C++ 语法覆盖可能不完整。
- `rag/` 中包含离线研究和评测脚本，其中部分脚本带有本地路径假设。
- 当前项目主要依赖 smoke check，还没有正式自动化测试体系。
- `test_api.py` 只用于 API 连通性检查，不是 parser 或 UI 测试。
