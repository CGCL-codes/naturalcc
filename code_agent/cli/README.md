# naturalcc CLI

基于 NaturalCC 语义图谱的命令行代码智能助手。

## 环境准备

### 1. 安装运行时

```bash
# Bun（TypeScript 运行时）
curl -fsSL https://bun.sh/install | bash

# uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | bash
```

### 2. 安装 Python 依赖

```bash
cd code_agent
uv sync
```

### 3. 安装系统 libclang（C/C++ 项目必需）

```bash
# Debian/Ubuntu
sudo apt install libclang1-18

# macOS
brew install llvm@18
```

如果只处理 Java、Python 等非 C/C++ 项目，可跳过此步。

### 4. 安装 CLI 依赖

```bash
cd code_agent/cli
bun install
```

### 5. 设置 API Key

```bash
export OPENROUTER_API_KEY=sk-xxx
# 或
export OPENAI_API_KEY=sk-xxx
```

## 运行

```bash
# 激活 Python 虚拟环境后运行
source ../.venv/bin/activate && bun run naturalcc

# 或直接传参
source ../.venv/bin/activate && bun run naturalcc \
  -f ../test1.cpp ../test2.cpp \
  -i "补全 foo 函数的实现" \
  --preview
```

## 命令行模式

```bash
source ../.venv/bin/activate && bun run naturalcc \
  -f test1.cpp test2.cpp \
  -i "补全 foo 函数的实现" \
  --preview
```

| 参数 | 说明 |
|------|------|
| `-f, --file` | 目标文件列表 |
| `-i, --instruction` | 修改/补全需求 |
| `-m, --model` | 模型名称（默认 `openrouter/deepseek/deepseek-chat`） |
| `-k, --apiKey` | API Key（默认读环境变量） |
| `-d, --projectDir` | 项目根目录（默认当前目录） |
| `-s, --symbol` | 目标符号（可选） |
| `-t, --completionType` | 补全类型，可选值：`member` / `variable` / `function` / `function_body` / `type` |
| `--prefix` | 补全前缀 |
| `--preview` | 仅预览 Prompt，不执行 Aider |

## REPL 交互模式

不传参直接运行 `bun run naturalcc` 进入 REPL 交互模式。

### 设置命令（`-` 开头）

| 命令 | 说明 |
|------|------|
| `-f <files...>` | 设置目标文件列表 |
| `-m <model>` | 设置模型 |
| `-k <key>` | 设置 API Key |
| `-d <dir>` | 设置项目根目录 |
| `-s <symbol>` | 设置目标符号 |
| `-t <type>` | 设置补全类型 |
| `--prefix <text>` | 设置补全前缀 |
| `--preview` | 切换预览模式开关 |
| `--run` | 以当前设置重新执行上一次的 instruction |

### 控制命令（`/` 开头）

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/settings` | 显示/隐藏当前设置面板 |
| `/clear` | 清除对话历史 |
| `/exit` | 退出 REPL |

### 交互流程

1. 先用 `-f` 设置目标文件（必须）
2. 根据需要调整 `-m`、`-d`、`-t` 等参数
3. 直接输入自然语言指令并回车，开始执行
4. 按 `Esc` 可中断正在执行的任务
5. 按两次 `Ctrl+C` 退出
