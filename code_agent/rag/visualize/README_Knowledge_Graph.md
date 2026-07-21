# RAG 知识图谱可视化插件（Knowledge Graph）说明

> 本文档整理了此前为 `code_agent` 项目新增的 **RAG JSON 结果可视化图谱** 功能：将 C/C++ 或 Java 项目的 RAG 解析结果转换成可交互的 `vis.js` 力导向知识图谱，并以插件形式注册到 Web UI 中。

---

## 1. 功能概述

- 新增名为 `knowledge_graph` 的 Feature Plugin，自动注册到插件系统中。
- 支持解析 **C/C++** 与 **Java** 项目，复用已有的 `rag/c/` 与 `rag/java/` 解析器。
- 将解析得到的项目级 JSON 转换为 **单个 HTML 文件**，内嵌 `vis-network`，无需额外后端即可用浏览器打开。
- 在 Web UI 中直接**预览图谱 iframe**，并提供**下载 HTML**按钮。
- 可选**保留中间 JSON**文件，便于后续二次处理或调试。

交互特性：

- 力导向布局 +  stabilization 后冻结物理模拟
- 节点按类型着色（Module / Function / Method / Class / Struct / Field / Import / External 等）
- 节点大小按度数（degree）缩放，低度节点自动隐藏标签减少 clutter
- 点击节点查看信息面板、邻居列表，点击邻居跳转
- 顶部搜索框快速定位节点
- 右侧 legend 可单独显示/隐藏某类节点，支持“全选/取消全选”

---

## 2. 新增 / 修改文件清单

### 2.1 新增文件

| 文件 | 作用 |
|------|------|
| `plugins/knowledge_graph.py` | Feature Plugin 主体：参数校验、调用解析器、生成 HTML、返回结果 |
| `rag/visualize/generate_graph.py` | 命令行/库接口：封装 C/C++ 与 Java 项目解析，输出 RAG JSON |
| `rag/visualize/visualize.py` | 将 RAG JSON 转换成 vis.js 图谱数据并渲染成单个 HTML |

### 2.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `webui/src/App.jsx` | `knowledge_graph` 允许不填 instruction；运行结果中渲染 HTML artifact iframe 与下载按钮 |
| `webui/src/styles.css` | 新增 `.artifact-viewer`、`.artifact-header`、`.artifact-iframe` 样式 |

### 2.3 未改动但复用的文件

| 文件 | 说明 |
|------|------|
| `plugins/__init__.py` | 自动扫描并导入 `plugins/` 下所有模块，`knowledge_graph.py` 通过 `@register_plugin` 自动注册 |
| `plugins/registry.py` | 插件注册表，无需手动添加 |
| `plugins/dispatcher.py` | 按 `ExecutionMode.DIRECT` 调度插件执行 |
| `plugins/base.py` | `FeaturePlugin`、`PluginResult`、`ExecutionContext` 基类 |
| `agent_web_api.py` | 无需改动；`/api/bootstrap` 会自动列出本插件及其 schema |

---

## 3. 实现细节

### 3.1 插件：`plugins/knowledge_graph.py`

- **注册方式**：类装饰器 `@register_plugin`
- **执行模式**：`ExecutionMode.DIRECT`（不调用 Aider，本地直接完成）
- **metadata**：
  - name: `knowledge_graph`
  - label: `Knowledge Graph`
- **配置 schema**：

| 字段名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `language` | select | 是 | `c` | 项目语言：`c`（C/C++）或 `java` |
| `output_name` | text | 否 | `knowledge_graph.html` | 生成的 HTML 文件名（相对于项目目录） |
| `title` | text | 否 | `Project Knowledge Graph` | 浏览器标签页与页面标题 |
| `save_json` | switch | 否 | `false` | 是否同时保存中间 RAG JSON |
| `json_name` | text | 否 | `knowledge_graph.json` | 中间 JSON 文件名（`save_json=true` 时生效） |

- **执行流程**：
  1. 校验 `project_dir` 存在
  2. 根据 `language` 调用 `rag/visualize/generate_graph.py` 中的解析函数
  3. 若 `save_json=true`，将 RAG JSON 写入项目目录
  4. 调用 `rag/visualize/visualize.py` 生成 `nodes / edges / legend / stats`
  5. 写入 HTML 文件
  6. 返回 `PluginResult`，`artifacts` 中携带完整 HTML 字符串、`html_path`、节点数、边数等

- **错误提示**：
  - C/C++ 解析失败且异常含 `libclang` / `clang` 时，提示安装 `libclang1-18`
  - Java 解析失败且异常含 `tree` / `sitter` 时，提示安装 `tree-sitter-java`

### 3.2 RAG 解析封装：`rag/visualize/generate_graph.py`

提供两个核心函数：

```python
from code_agent.rag.visualize.generate_graph import parse_c_project, parse_java_project

rag_data = parse_c_project("/path/to/c_project")
rag_data = parse_java_project("/path/to/java_project")
```

实现要点：

- C/C++：动态加载 `rag/c/cfile_parse.py`、`rag/c/node_prompt.py`、`rag/c/preprocess.py`，实例化 `CProjectParser().parse_dir(...)`
- Java：动态加载 `rag/java/javafile_parse_ts.py`、`rag/java/node_prompt_java_ts.py`、`rag/java/java_project_parser_ts.py`，实例化 `JavaProjectParserTS().parse_dir(...)`
- 动态加载是为了绕过相对导入限制，使脚本既可作为模块调用，也可独立运行

CLI 用法（独立运行）：

```bash
# 生成 C/C++ 项目的 RAG JSON
python rag/visualize/generate_graph.py -d /path/to/c_project -l c -o c_project.json

# 生成 Java 项目的 RAG JSON
python rag/visualize/generate_graph.py -d /path/to/java_project -l java -o java_project.json
```

### 3.3 可视化：`rag/visualize/visualize.py`

核心函数：

```python
from code_agent.rag.visualize.visualize import rag_to_vis, generate_html

nodes, edges, legend, stats = rag_to_vis(rag_data)
html = generate_html(nodes, edges, legend, stats, title="My Graph")
```

- **节点分组与颜色**：使用 `GROUP_COLORS` 调色板，按节点类型（Function / Class / Module / External 等）分组着色
- **边关系**：解析 `include` / `import` 与 `rels` 字段，去重后生成带标签的有向边
- **目标解析策略**（`resolve_target`）：
  1. 直接命中已有节点 ID
  2. 同模块下符号 `module::name`
  3. 全局 label 索引（Java FQCN 也加入索引）
  4. 模块路径后缀 / basename 匹配
  5. 无法解析时生成灰色 `External` 节点
- **HTML 模板**：内嵌 `vis-network@9.1.6` CDN，带 SRI，无需 npm 安装

CLI 用法（独立运行）：

```bash
python rag/visualize/visualize.py -i c_project.json -o graph.html --title "C Project Graph"
```

### 3.4 前端改动

#### `webui/src/App.jsx`

1. **允许空 instruction 执行**：

   ```js
   function canRunWithoutInstruction() {
     // ...
     if (currentFeature === "knowledge_graph") {
       return true;
     }
     // ...
   }
   ```

2. **默认任务标签**：

   ```js
   if (currentFeature === "knowledge_graph") {
     return "Knowledge graph generation";
   }
   ```

3. **结果渲染**：在 assistant 消息气泡中，当 `msg.artifacts?.html` 存在时：
   - 显示节点数 / 边数统计
   - 提供 **Download** 按钮，下载 `knowledge_graph.html`
   - 使用 `<iframe srcDoc={msg.artifacts.html}>` 在聊天区直接预览图谱

#### `webui/src/styles.css`

新增三段样式，控制 artifact 预览区域：

```css
.artifact-viewer { margin-top: 12px; }
.artifact-header { /* flex 标题栏 */ }
.artifact-iframe { width: 100%; height: 420px; border-radius: 8px; }
```

---

## 4. 使用方法

### 4.1 Web UI

1. 启动后端：`uv run python agent_web_api.py --host 127.0.0.1 --port 7860`
2. 启动前端：`cd webui && npm install && npm run dev`
3. 在右侧 Settings → Feature 中选择 **Knowledge Graph**
4. 填写配置（通常只需选 `language`）
5. 主界面无需填写 instruction，直接点击 **Execute**
6. 执行完成后，聊天区会显示：
   - 运行日志
   - 节点 / 边数量
   - **Download** 按钮
   - 可直接交互的图谱 iframe

### 4.2 命令行（仅生成 JSON / HTML）

```bash
# 1. 生成 RAG JSON
python rag/visualize/generate_graph.py -d /path/to/project -l c -o project.json

# 2. 生成可视化 HTML
python rag/visualize/visualize.py -i project.json -o project_graph.html --title "Project Graph"
```

### 4.3 API 调用

```bash
curl -N -X POST http://127.0.0.1:7860/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "project_dir": "/path/to/project",
    "target_files": [],
    "instruction": "",
    "model": "openrouter/deepseek/deepseek-chat",
    "api_key": null,
    "feature": "knowledge_graph",
    "feature_config": {
      "language": "c",
      "output_name": "knowledge_graph.html",
      "title": "Project Knowledge Graph",
      "save_json": true,
      "json_name": "knowledge_graph.json"
    }
  }'
```

返回为 NDJSON 流，最后一条 `type: done` 事件会包含 `artifacts.html`（完整 HTML 字符串）。

---

## 5. 依赖

- **C/C++**：系统 `libclang1-18` + Python `clang==18.1.8`（项目已 pin）
- **Java**：`tree-sitter` + `tree-sitter-java`（已在 `pyproject.toml`）
- **可视化**：`vis-network@9.1.6` 通过 CDN 加载，无需前端 npm 依赖

---

## 6. 输出产物

执行后会在 `project_dir` 下生成：

- `<output_name>`（默认 `knowledge_graph.html`）：可交互图谱，直接用浏览器打开
- `<json_name>`（仅当 `save_json=true`，默认 `knowledge_graph.json`）：中间 RAG JSON

---

## 7. 已知限制与后续可扩展点

1. **C++ 覆盖不完整**：底层解析器部分参数仍偏 C 导向，复杂 C++ 模板/命名空间可能解析不完美。
2. **外部符号**：无法解析的目标统一显示为灰色 `External` 节点，后续可尝试结合 import/include 做更精确的 FQCN/路径解析。
3. **Legend 仅按节点类型过滤**：目前只支持节点分组的显隐；后续可扩展按边类型（calls/includes/extends/implements 等）过滤。
4. **布局算法单一**：当前使用 `forceAtlas2Based`；后续可提供布局切换（hierarchical、static 等）。
5. **语言扩展**：目前仅 C/C++ 与 Java；后续可接入 `rag/` 下其他语言解析器（如 Python、Go 等），只需在 `generate_graph.py` 与 `knowledge_graph.py` 的 `language` 选项中扩展。
6. **CLI 入口**：当前插件主要通过 Web API 使用；如需要，可在 `aider_runner.py` 中增加 `--feature knowledge_graph` 的 CLI 支持。
7. **离线环境**：HTML 依赖 CDN 加载 `vis-network`；若需在完全离线环境使用，可将 `vis-network.min.js` 下载到本地并修改 `visualize.py` 的 script src。

---

## 8. 关键代码索引

| 功能 | 文件 | 关键符号 |
|------|------|----------|
| 插件注册 | `plugins/knowledge_graph.py` | `@register_plugin` / `KnowledgeGraphPlugin` |
| 项目解析 | `rag/visualize/generate_graph.py` | `parse_c_project` / `parse_java_project` |
| 图谱生成 | `rag/visualize/visualize.py` | `rag_to_vis` / `generate_html` |
| 前端执行入口 | `webui/src/App.jsx` | `canRunWithoutInstruction` / `defaultTaskLabel` / artifact iframe |
| 前端样式 | `webui/src/styles.css` | `.artifact-viewer` / `.artifact-iframe` |
