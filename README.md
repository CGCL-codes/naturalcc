# NaturalCC · Code Agent

这个仓库包含两个相互关联的部分：

| 目录 | 是什么 | 文档 |
|---|---|---|
| `code_agent/` | **开源的主角**：一个上下文感知的本地编码 Agent，提供 Web UI、CLI 和 VS Code 插件三种使用方式 | [📖 中文教程](code_agent/README.zh.md) |
| `ncc/` | NaturalCC 上游代码智能研究工具集（代码解析、模型训练/评估、PEFT），`code_agent` 复用了它的语义解析能力 | — |
| `examples/` | 基于 NaturalCC 的示例（代码补全评估、指令微调等） | — |

## 快速开始

```bash
# 1. 安装（Python 3.12 + uv + Node.js）
cd code_agent
uv sync
uv run python scripts/install_deepseek_tokenizer.py
cd webui && npm install && cd ..

# 2. 安装 CodeGraph CLI（知识图谱能力）
# macOS / Linux / WSL:
curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh
# Windows PowerShell:
# irm https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.ps1 | iex
codegraph --version

# 3. 构建前端并启动本地服务
cd webui && npm run build && cd ..
uv run python agent_web_api.py --host 127.0.0.1 --port 7860
```

浏览器打开 **http://127.0.0.1:7860/** 即可。首次使用知识图谱时，在 Settings → Knowledge graph 中打开开关并点击 **Initialize**，它会在当前 workspace 本地生成 `.codegraph/` 索引目录。API Key 不需要写进代码或 README；Web UI 的 Agent/Pipeline 模式都可在页面 Settings 中填写，CLI 和脚本也兼容环境变量方式，详情见 **[code_agent/README.zh.md](code_agent/README.zh.md)**。

## 它是什么

`code_agent` 是一个本地运行的代码编辑 Agent：

- **上下文感知**：结合 NaturalCC 静态解析（C/C++ 用 libclang，Java 用 tree-sitter）与 21 个内置工具（读写文件、跑测试、Git 状态、知识图谱检索……），不是通用聊天机器人；
- **持久化 Agent 模式**：事件溯源（SQLite）保证每一步可审计可恢复，多轮会话、Token 精确预算、上下文自动压缩、长期记忆（候选 → 人工确认 → FTS5 检索）；
- **安全可控**：所有高风险操作需人工审批，命令白名单、Git 子命令黑名单、敏感路径禁读、工具结果自动脱敏；
- **三种入口**：React Web UI（FastAPI 后端）、Bun 终端 CLI（Ink TUI）、VS Code 插件。

## 特性一览

- 🧩 传统 Pipeline 模式：代码补全 / 代码修复 / 漏洞检测与自动修复 / 代码总结 / 设计稿转代码 / 知识图谱可视化
- 🧠 Durable Agent 模式：多轮会话、事件溯源、上下文压缩 checkpoint、长期记忆提案与人工治理
- 🛡️ 审批流与预算控制：LLM 调用数 / 工具调用数 / 输入 Token / 时长 / 成本，超限自动停止
- 🗺️ CodeGraph 知识图谱：符号探索、调用路径、影响分析、交互式 HTML 可视化
- 🔌 VS Code 扩展：编辑器内直接打开 Agent 面板

## License

[MIT](LICENSE)
