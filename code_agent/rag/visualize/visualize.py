#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 代码知识图谱可视化

将 rag/c 或 rag/java 生成的项目级 JSON（parse_res）转换成可交互的
vis.js 力导向图谱，输出为单个 HTML 文件，直接用浏览器打开即可。

核心 borrowed from graphify/graphify/export.py 的 HTML 生成逻辑，
去掉了对 graphify 包、networkx 和社区检测的依赖，改为按节点类型（Module /
Function / Class / Method / Field / Struct / ...）分组着色。
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. 颜色与常量
# ---------------------------------------------------------------------------

GROUP_COLORS = [
    "#2563EB",  # Module        — vivid blue
    "#EA580C",  # Function      — vivid orange
    "#DC2626",  # Class / Struct — vivid red
    "#0891B2",  # Field / Variable — vivid cyan
    "#16A34A",  # Enum / Union  — vivid green
    "#CA8A04",  # Interface / Annotation — vivid gold
    "#9333EA",  # Constructor   — vivid purple
    "#DB2777",  # Import / Include — vivid pink
    "#7C3AED",  # Record / Typedef — vivid violet
    "#6B7280",  # Other         — neutral gray
]

_EXTERNAL_COLOR = "#374151"  # dark gray, clearly visible on light background
_MAX_LABEL_LEN = 120


# ---------------------------------------------------------------------------
# 2. 安全辅助
# ---------------------------------------------------------------------------

_Control_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# [maint-note:fda7ebdbfe2e]
# 维护导航：_sanitize；下列为本函数源码中的接口与路径索引。
# 输入 text：Any。
# 输入 max_len：int；缺省表达式 _MAX_LABEL_LEN。
# 返回契约：str。
# 分支条件：text is None。
# 分支条件：len(s) > max_len。
# 返回路径：''。
# 返回路径：s。
# 直接调用：_Control_RE.sub。
# 直接调用：str。
# 直接调用：len。
# [/maint-note:fda7ebdbfe2e]
def _sanitize(text: Any, max_len: int = _MAX_LABEL_LEN) -> str:
    if text is None:
        return ""
    s = _Control_RE.sub("", str(text))
    if len(s) > max_len:
        s = s[:max_len]
    return s


# [maint-note:6911b539d53a]
# 维护导航：_js_safe；下列为本函数源码中的接口与路径索引。
# 输入 obj：Any。
# 返回契约：str。
# 返回路径：json.dumps(obj, ensure_ascii=False).replace('</', '<\\/')。
# 直接调用：json.dumps(obj, ensure_ascii=False).replace。
# 直接调用：json.dumps。
# [/maint-note:6911b539d53a]
def _js_safe(obj: Any) -> str:
    """转义 </script>，防止嵌入 JSON 时破坏 <script> 标签。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


# ---------------------------------------------------------------------------
# 3. HTML 模板（从 graphify/export.py 提取并简化）
# ---------------------------------------------------------------------------

# [maint-note:8f75065a007b]
# 维护导航：_html_styles；下列为本函数源码中的接口与路径索引。
# 返回契约：str。
# 返回路径：'<style>\n * { box-sizing: border-box; margin: 0; padding: 0; }\n body { background: #f3f4f6; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; flex-direction: column; height: 100vh; over。
# [/maint-note:8f75065a007b]
def _html_styles() -> str:
    """Light-theme CSS injected around the Pyvis container."""
    return """<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f3f4f6; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  .kg-header { height: 52px; flex-shrink: 0; display: flex; align-items: center; gap: 18px; padding: 0 18px; border-bottom: 1px solid #d1d5db; background: #ffffff; }
  .kg-header strong { font-size: 14px; color: #111827; }
  .kg-header span { color: #6b7280; font-size: 12px; }
  .kg-header .hint { margin-left: auto; color: #9ca3af; font-size: 11px; }
  .kg-body { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr) 320px; min-height: 0; }
  #kg-graph { background: #ffffff; border-right: 1px solid #e5e7eb; min-height: 0; }
  #kg-sidebar { background: #ffffff; display: flex; flex-direction: column; overflow: hidden; }
  #kg-search-wrap { padding: 12px; border-bottom: 1px solid #e5e7eb; }
  #kg-search { width: 100%; background: #ffffff; border: 1px solid #d1d5db; color: #111827; padding: 7px 10px; border-radius: 6px; font-size: 13px; outline: none; }
  #kg-search:focus { border-color: #2563EB; box-shadow: 0 0 0 2px rgba(37,99,235,0.15); }
  #kg-search-results { max-height: 140px; overflow-y: auto; padding: 4px 12px; border-bottom: 1px solid #e5e7eb; display: none; background: #ffffff; }
  .kg-search-item { padding: 4px 6px; cursor: pointer; border-radius: 4px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .kg-search-item:hover { background: #eff6ff; }
  #kg-info-panel { padding: 14px; border-bottom: 1px solid #e5e7eb; min-height: 140px; }
  #kg-info-panel h3 { font-size: 13px; color: #6b7280; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  #kg-info-content { font-size: 13px; color: #374151; line-height: 1.6; }
  #kg-info-content .field { margin-bottom: 5px; }
  #kg-info-content .field b { color: #111827; }
  #kg-info-content .empty { color: #9ca3af; font-style: italic; }
  .kg-neighbor-link { display: block; padding: 2px 6px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid #d1d5db; }
  .kg-neighbor-link:hover { background: #eff6ff; }
  #kg-neighbors-list { max-height: 160px; overflow-y: auto; margin-top: 4px; }
  #kg-legend-wrap { flex: 1; overflow-y: auto; padding: 12px; }
  #kg-legend-wrap h3 { font-size: 13px; color: #6b7280; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .kg-legend-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; border-radius: 4px; font-size: 12px; }
  .kg-legend-item:hover { background: #f3f4f6; padding-left: 4px; }
  .kg-legend-item.dimmed { opacity: 0.35; }
  .kg-legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08); }
  .kg-legend-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #111827; }
  .kg-legend-count { color: #9ca3af; font-size: 11px; }
  #kg-stats { padding: 10px 14px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af; }
  #kg-legend-controls { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding: 4px 0; }
  #kg-legend-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 12px; color: #4b5563; user-select: none; }
  #kg-legend-controls label:hover { color: #111827; }
  .kg-legend-cb, #kg-select-all-cb { appearance: none; -webkit-appearance: none; width: 14px; height: 14px; border: 1.5px solid #d1d5db; border-radius: 3px; background: #ffffff; cursor: pointer; position: relative; flex-shrink: 0; }
  .kg-legend-cb:checked, #kg-select-all-cb:checked { background: #2563EB; border-color: #2563EB; }
  .kg-legend-cb:checked::after, #kg-select-all-cb:checked::after { content: ''; position: absolute; left: 3.5px; top: 1px; width: 4px; height: 7px; border: solid #fff; border-width: 0 2px 2px 0; transform: rotate(45deg); }
  #kg-select-all-cb:indeterminate { background: #2563EB; border-color: #2563EB; }
  #kg-select-all-cb:indeterminate::after { content: ''; position: absolute; left: 2px; top: 5px; width: 8px; height: 2px; background: #fff; border: none; transform: none; }
</style>"""


# [maint-note:b6151f65f57d]
# 维护导航：_html_script；下列为本函数源码中的接口与路径索引。
# 输入 nodes_json：str。
# 输入 edges_json：str。
# 输入 legend_json：str。
# 返回契约：str。
# 返回路径：f"""<script>\nconst RAW_NODES = {nodes_json};\nconst RAW_EDGES = {edges_json};\nconst LEGEND = {legend_json};\n\nfunction esc(s) {{\n return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').re。
# [/maint-note:b6151f65f57d]
def _html_script(nodes_json: str, edges_json: str, legend_json: str) -> str:
    """Sidebar script — drives the Pyvis network via the global `network` variable."""
    return f"""<script>
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const LEGEND = {legend_json};

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

const nodesDS = (typeof network !== 'undefined' && network.body && network.body.nodes) ? network.body.nodes : null;
const edgesDS = (typeof network !== 'undefined' && network.body && network.body.edges) ? network.body.edges : null;

function lookupNode(id) {{
  if (nodesDS) {{
    if (typeof nodesDS.get === 'function') return nodesDS.get(id);
    return nodesDS[id] || null;
  }}
  return RAW_NODES.find(n => n.id === id) || null;
}}

function focusNode(nodeId) {{
  if (typeof network === 'undefined') return;
  network.focus(nodeId, {{ scale: 1.4, animation: true }});
  network.selectNodes([nodeId]);
  showInfo(nodeId);
}}

function showInfo(nodeId) {{
  const n = lookupNode(nodeId);
  if (!n) return;
  let neighborIds = [];
  if (typeof network !== 'undefined' && network.getConnectedNodes) {{
    try {{ neighborIds = network.getConnectedNodes(nodeId) || []; }} catch (_) {{ neighborIds = []; }}
  }} else {{
    neighborIds = Array.from(new Set([
      ...RAW_EDGES.filter(e => e.to === nodeId).map(e => e.from),
      ...RAW_EDGES.filter(e => e.from === nodeId).map(e => e.to),
    ]));
  }}
  const neighborItems = neighborIds.map(nid => {{
    const nb = lookupNode(nid);
    const color = nb ? (nb.color && nb.color.background) || '#374151' : '#374151';
    return `<span class="kg-neighbor-link" style="border-left-color:${{esc(color)}}" onclick="focusNode(${{JSON.stringify(nid)}})">${{esc(nb ? nb.label : nid)}}</span>`;
  }}).join('');
  const group = n._group || (n.group) || 'unknown';
  const sourceFile = n._source_file || n.source_file || '-';
  const degree = n._degree || n.degree || 0;
  document.getElementById('kg-info-content').innerHTML = `
    <div class="field"><b>${{esc(n.label)}}</b></div>
    <div class="field">Group: ${{esc(group)}}</div>
    <div class="field">Source: ${{esc(sourceFile)}}</div>
    <div class="field">Degree: ${{degree}}</div>
    ${{neighborIds.length ? `<div class="field" style="margin-top:8px;color:#6b7280;font-size:11px">Neighbors (${{neighborIds.length}})</div><div id="kg-neighbors-list">${{neighborItems}}</div>` : ''}}
  `;
}}

let hoveredNodeId = null;
if (typeof network !== 'undefined') {{
  network.on('hoverNode', params => {{ hoveredNodeId = params.node; const c = document.getElementById('kg-graph'); if (c) c.style.cursor = 'pointer'; }});
  network.on('blurNode', () => {{ hoveredNodeId = null; const c = document.getElementById('kg-graph'); if (c) c.style.cursor = 'default'; }});
  network.on('click', params => {{
    if (params.nodes.length > 0) {{ showInfo(params.nodes[0]); }}
    else if (hoveredNodeId === null) {{ document.getElementById('kg-info-content').innerHTML = '<span class="empty">Click a node to inspect it</span>'; }}
  }});
  network.once('stabilizationIterationsDone', () => {{
    try {{ network.setOptions({{ physics: {{ enabled: false }} }}); }} catch (_) {{}}
  }});
}}

const searchInput = document.getElementById('kg-search');
const searchResults = document.getElementById('kg-search-results');
searchInput.addEventListener('input', () => {{
  const q = searchInput.value.toLowerCase().trim();
  searchResults.innerHTML = '';
  if (!q) {{ searchResults.style.display = 'none'; return; }}
  const matches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  if (!matches.length) {{ searchResults.style.display = 'none'; return; }}
  searchResults.style.display = 'block';
  matches.forEach(n => {{
    const el = document.createElement('div');
    el.className = 'kg-search-item';
    el.textContent = n.label;
    const bg = (n.color && n.color.background) || '#374151';
    el.style.borderLeft = `3px solid ${{bg}}`;
    el.style.paddingLeft = '8px';
    el.onclick = () => {{
      focusNode(n.id);
      searchResults.style.display = 'none';
      searchInput.value = '';
    }};
    searchResults.appendChild(el);
  }});
}});
document.addEventListener('click', e => {{
  if (!searchResults.contains(e.target) && e.target !== searchInput)
    searchResults.style.display = 'none';
}});

const hiddenGroups = new Set();
const selectAllCb = document.getElementById('kg-select-all-cb');

function updateSelectAllState() {{
  const total = LEGEND.length;
  const hidden = hiddenGroups.size;
  selectAllCb.checked = hidden === 0;
  selectAllCb.indeterminate = hidden > 0 && hidden < total;
}}

function toggleAllGroups(hide) {{
  document.querySelectorAll('.kg-legend-item').forEach(item => {{ hide ? item.classList.add('dimmed') : item.classList.remove('dimmed'); }});
  document.querySelectorAll('.kg-legend-cb').forEach(cb => {{ cb.checked = !hide; }});
  LEGEND.forEach(g => {{ if (hide) hiddenGroups.add(g.group); else hiddenGroups.delete(g.group); }});
  const updates = RAW_NODES.map(n => ({{ id: n.id, hidden: hide }}));
  if (nodesDS && typeof nodesDS.update === 'function') nodesDS.update(updates);
  updateSelectAllState();
}}

const legendEl = document.getElementById('kg-legend');
LEGEND.forEach(g => {{
  const item = document.createElement('div');
  item.className = 'kg-legend-item';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'kg-legend-cb';
  cb.checked = true;
  cb.addEventListener('change', (e) => {{
    e.stopPropagation();
    if (cb.checked) {{
      hiddenGroups.delete(g.group);
      item.classList.remove('dimmed');
    }} else {{
      hiddenGroups.add(g.group);
      item.classList.add('dimmed');
    }}
    const updates = RAW_NODES.filter(n => n.group === g.group).map(n => ({{ id: n.id, hidden: !cb.checked }}));
    if (nodesDS && typeof nodesDS.update === 'function') nodesDS.update(updates);
    updateSelectAllState();
  }});
  item.innerHTML = `<div class="kg-legend-dot" style="background:${{g.color}}"></div>
    <span class="kg-legend-label">${{g.label}}</span>
    <span class="kg-legend-count">${{g.count}}</span>`;
  item.prepend(cb);
  item.onclick = (e) => {{
    if (e.target === cb) return;
    cb.checked = !cb.checked;
    cb.dispatchEvent(new Event('change'));
  }};
  legendEl.appendChild(item);
}});
</script>"""


# ---------------------------------------------------------------------------
# 4. RAG JSON -> vis.js 数据
# ---------------------------------------------------------------------------

# [maint-note:9e6a606ea179]
# 维护导航：_norm_path；下列为本函数源码中的接口与路径索引。
# 输入 p：str。
# 返回契约：str。
# 返回路径：str(p).replace('\\', '/')。
# 直接调用：str(p).replace。
# 直接调用：str。
# [/maint-note:9e6a606ea179]
def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


# [maint-note:18e638ff73db]
# 维护导航：load_rag_graph；下列为本函数源码中的接口与路径索引。
# 输入 path：Path。
# 返回契约：Dict[str, Dict[str, Any]]。
# 分支条件：not isinstance(data, dict)。
# 返回路径：data。
# 显式抛出：ValueError(f'RAG graph 必须是 dict，实际为 {type(data).__name__}')。
# 直接调用：open。
# 直接调用：json.load。
# 直接调用：isinstance。
# 直接调用：ValueError。
# 直接调用：type。
# 资源作用域：open(path, 'r', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# [/maint-note:18e638ff73db]
def load_rag_graph(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"RAG graph 必须是 dict，实际为 {type(data).__name__}")
    return data


# [maint-note:d4f43ec82054]
# 维护导航：_group_for_type；下列为本函数源码中的接口与路径索引。
# 输入 node_type：str。
# 返回契约：str。
# 返回路径：mapping.get(t, t)。
# 直接调用：str(node_type or 'Other').strip。
# 直接调用：str。
# 直接调用：mapping.get。
# [/maint-note:d4f43ec82054]
def _group_for_type(node_type: str) -> str:
    """按节点类型归并到可读的分组名。"""
    t = str(node_type or "Other").strip()
    mapping = {
        "Module": "Module",
        "Function": "Function",
        "Method": "Method",
        "Constructor": "Constructor",
        "Class": "Class",
        "Struct": "Struct",
        "Interface": "Interface",
        "Annotation": "Annotation",
        "Record": "Record",
        "Enum": "Enum",
        "Union": "Union",
        "Field": "Field",
        "Variable": "Variable",
        "Import": "Import",
    }
    return mapping.get(t, t)


# [maint-note:1f8a63624e44]
# 维护导航：_color_for_group；下列为本函数源码中的接口与路径索引。
# 输入 group：str。
# 输入 palette：List[str]。
# 返回契约：str。
# 分支条件：idx is None。
# 返回路径：palette[idx]。
# 直接调用：mapping.get。
# 直接调用：sum。
# 直接调用：ord。
# 直接调用：max。
# 直接调用：len。
# 集合派生：c 来自 group。
# [/maint-note:1f8a63624e44]
def _color_for_group(group: str, palette: List[str]) -> str:
    # 显式 group -> palette 下标，保证节点颜色与图例一致
    mapping = {
        "Module": 0,
        "Function": 1,
        "Method": 1,
        "Class": 2,
        "Struct": 2,
        "Field": 3,
        "Variable": 3,
        "Enum": 4,
        "Union": 4,
        "Interface": 5,
        "Annotation": 5,
        "Constructor": 6,
        "Import": 7,
        "Record": 8,
        "Other": 9,
    }
    idx = mapping.get(group)
    if idx is None:
        # 未知 group 仍用哈希作为稳定回退
        idx = sum(ord(c) for c in group) % max(len(palette), 1)
    return palette[idx]


# [maint-note:8d09f3a354b2]
# 维护导航：rag_to_vis；下列为本函数源码中的接口与路径索引。
# 输入 rag：Dict[str, Dict[str, Any]]。
# 返回契约：Tuple[List[dict], List[dict], List[dict], str]。
# 分支条件：nid in node_ids。
# 分支条件：meta。
# 分支条件：meta.get('type')。
# 分支条件：meta.get('sline') is not None。
# 分支条件：not module。
# 分支条件：not isinstance(mod_info, dict)。
# 分支条件：pkg。
# 分支条件：not name or name == module。
# 分支条件：not isinstance(info, dict)。
# 分支条件：target_path in module_ids。
# 分支条件：candidates。
# 分支条件：eid not in node_ids。
# 分支条件：not target_name。
# 分支条件：target_name in node_ids。
# 分支条件：qualified in node_ids。
# 分支条件：same_module。
# 分支条件：mod。
# 分支条件：not name。
# 分支条件：sid not in node_ids。
# 分支条件：includes。
# 分支条件：isinstance(includes, list)。
# 分支条件：not inc。
# 分支条件：target。
# 分支条件：isinstance(rels, list)。
# 分支条件：not isinstance(r, (list, tuple))。
# 分支条件：len(r) >= 3。
# 分支条件：len(r) == 2。
# 分支条件：not target_name or not rel_type。
# 分支条件：src == tgt。
# 分支条件：key in seen_edges。
# 分支条件：grp == 'External'。
# 遍历边界：(module, file_info) ← rag.items()。
# 遍历边界：(name, info) ← file_info.items()。
# 遍历边界：inc ← includes。
# 遍历边界：r ← rels。
# 遍历边界：(src, tgt, rel) ← raw_edges。
# 遍历边界：n ← vis_nodes。
# 遍历边界：(grp, count) ← sorted(group_counter.items(), key=lambda x: -x[1])。
# 返回路径：无返回值。
# 返回路径：target_path。
# 返回路径：sorted(candidates, key=lambda x: len(x))[0]。
# 返回路径：None。
# 返回路径：eid。
# 返回路径：ensure_external('?')。
# 返回路径：target_name。
# 返回路径：qualified。
# 返回路径：same_module[0]。
# 返回路径：candidates[0]。
# 返回路径：mod。
# 返回路径：ensure_external(target_name)。
# 返回路径：(vis_nodes, vis_edges, legend_data, stats)。
# 写入目标：group_counter[group]。
# 写入目标：module_to_package[module]。
# 写入目标：degree[src]。
# 写入目标：degree[tgt]。
# 写入目标：n['degree']。
# 写入目标：n['color']。
# 写入目标：n['size']。
# 写入目标：n['font']['size']。
# 直接调用：set。
# 直接调用：defaultdict。
# 直接调用：Counter。
# 直接调用：node_ids.add。
# 直接调用：meta.get。
# 直接调用：title_parts.append。
# 直接调用：_html.escape。
# 直接调用：'\n'.join。
# 直接调用：vis_nodes.append。
# 直接调用：_sanitize。
# 直接调用：label_to_ids[_sanitize(label)].append。
# 直接调用：rag.items。
# 直接调用：_norm_path。
# 直接调用：file_info.get。
# 直接调用：isinstance。
# 直接调用：add_node。
# 直接调用：Path。
# 直接调用：mod_info.get。
# 直接调用：file_info.items。
# 直接调用：_group_for_type。
# 直接调用：info.get。
# 直接调用：label_to_ids[fqcn].append。
# 直接调用：m.endswith。
# 直接调用：sorted。
# 直接调用：external_ids.add。
# 直接调用：str(target_name or '').strip。
# 直接调用：str。
# 直接调用：ensure_external。
# 直接调用：label_to_ids.get。
# 直接调用：c.startswith。
# 直接调用：resolve_module。
# 直接调用：raw_edges.append。
# 直接调用：len。
# 直接调用：resolve_target。
# 直接调用：str(rel_type).lower。
# 直接调用：seen_edges.add。
# 直接调用：vis_edges.append。
# 直接调用：max。
# 直接调用：degree.values。
# 直接调用：_color_for_group。
# 直接调用：degree.get。
# 直接调用：group_color.get。
# 直接调用：round。
# 直接调用：group_counter.items。
# 直接调用：legend_data.append。
# 集合派生：n 来自 vis_nodes；过滤 n['group'] == 'Module'。
# 表达式分流：name == module；真值与假值分支分别计算，不同时执行。
# 表达式分流：edge_index % 2 == 0；真值与假值分支分别计算，不同时执行。
# 集合派生：g 来自 group_counter。
# 表达式分流：deg >= max_deg * 0.12；真值与假值分支分别计算，不同时执行。
# [/maint-note:8d09f3a354b2]
def rag_to_vis(rag: Dict[str, Dict[str, Any]]) -> Tuple[List[dict], List[dict], List[dict], str]:
    """把 RAG parse_res 转换成 vis.js 需要的 nodes/edges/legend/stats。"""

    vis_nodes: List[dict] = []
    node_ids: set = set()
    label_to_ids: Dict[str, List[str]] = defaultdict(list)
    module_to_package: Dict[str, str] = {}

    group_counter: Counter = Counter()

    # [maint-note:35e643deec9f]
    # 维护导航：add_node；下列为本函数源码中的接口与路径索引。
    # 输入 nid：str。
    # 输入 label：str。
    # 输入 source_file：str。
    # 输入 group：str。
    # 输入 meta：Dict[str, Any] | None；缺省表达式 None。
    # 分支条件：nid in node_ids。
    # 分支条件：meta。
    # 分支条件：meta.get('type')。
    # 分支条件：meta.get('sline') is not None。
    # 返回路径：无返回值。
    # 写入目标：group_counter[group]。
    # 直接调用：node_ids.add。
    # 直接调用：meta.get。
    # 直接调用：title_parts.append。
    # 直接调用：_html.escape。
    # 直接调用：'\n'.join。
    # 直接调用：vis_nodes.append。
    # 直接调用：_sanitize。
    # 直接调用：label_to_ids[_sanitize(label)].append。
    # [/maint-note:35e643deec9f]
    def add_node(nid: str, label: str, source_file: str, group: str, meta: Dict[str, Any] | None = None):
        if nid in node_ids:
            return
        node_ids.add(nid)
        group_counter[group] += 1
        title_parts = [f"{label}"]
        if meta:
            if meta.get("type"):
                title_parts.append(f"Type: {meta['type']}")
            if meta.get("sline") is not None:
                title_parts.append(f"Line: {meta['sline']}")
        title = _html.escape("\n".join(title_parts))
        vis_nodes.append({
            "id": nid,
            "label": _sanitize(label),
            "group": group,
            "source_file": _sanitize(source_file),
            "title": title,
            "degree": 0,
            # 以下字段在生成后再补
            "color": None,
            "size": 10,
            "font": {"size": 12, "color": "#111827"},
        })
        # 索引 label -> ids，便于 rel 目标解析
        label_to_ids[_sanitize(label)].append(nid)

    # ---------- 第一遍：创建模块节点 + 符号节点 ----------
    for module, file_info in rag.items():
        module = _norm_path(module)
        if not module:
            continue

        # 模块节点
        mod_info = file_info.get(module) or file_info.get("") or {}
        if not isinstance(mod_info, dict):
            mod_info = {}
        add_node(module, label=Path(module).name or module,
                 source_file=module, group="Module", meta=mod_info)

        # Java 的 package 信息用于后续 FQCN 映射
        pkg = mod_info.get("package")
        if pkg:
            module_to_package[module] = pkg

        for name, info in file_info.items():
            if not name or name == module:
                continue
            if not isinstance(info, dict):
                continue
            sid = f"{module}::{name}"
            group = _group_for_type(info.get("type"))
            add_node(sid, label=name, source_file=module, group=group, meta=info)

            # Java：用 package + qname 再建一条 FQCN 索引，方便解析 import/call/access
            if pkg:
                fqcn = f"{pkg}.{name}"
                label_to_ids[fqcn].append(sid)

    # ---------- 第二遍：创建边 ----------
    raw_edges: List[Tuple[str, str, str]] = []  # (source, target, relation)

    module_ids = {n["id"] for n in vis_nodes if n["group"] == "Module"}

    # [maint-note:b7cf633cb372]
    # 维护导航：resolve_module；下列为本函数源码中的接口与路径索引。
    # 输入 target_path：str。
    # 返回契约：str | None。
    # 分支条件：target_path in module_ids。
    # 分支条件：candidates。
    # 返回路径：target_path。
    # 返回路径：sorted(candidates, key=lambda x: len(x))[0]。
    # 返回路径：None。
    # 直接调用：_norm_path。
    # 直接调用：m.endswith。
    # 直接调用：sorted。
    # 直接调用：Path。
    # 集合派生：m 来自 module_ids；过滤 m.endswith(target_path)。
    # 集合派生：m 来自 module_ids；过滤 Path(m).name == base。
    # [/maint-note:b7cf633cb372]
    def resolve_module(target_path: str) -> str | None:
        target_path = _norm_path(target_path)
        if target_path in module_ids:
            return target_path
        # 尝试后缀匹配（C include 经常是相对路径）
        candidates = [m for m in module_ids if m.endswith(target_path)]
        if candidates:
            # 优先最短（最精确）
            return sorted(candidates, key=lambda x: len(x))[0]
        # basename 匹配
        base = Path(target_path).name
        candidates = [m for m in module_ids if Path(m).name == base]
        if candidates:
            return sorted(candidates, key=lambda x: len(x))[0]
        return None

    external_ids: set = set()

    # [maint-note:a76586bd01ba]
    # 维护导航：ensure_external；下列为本函数源码中的接口与路径索引。
    # 输入 target_name：str。
    # 返回契约：str。
    # 分支条件：eid not in node_ids。
    # 返回路径：eid。
    # 直接调用：_sanitize。
    # 直接调用：add_node。
    # 直接调用：external_ids.add。
    # [/maint-note:a76586bd01ba]
    def ensure_external(target_name: str) -> str:
        eid = f"external::{_sanitize(target_name, max_len=80)}"
        if eid not in node_ids:
            add_node(eid, label=target_name, source_file="", group="External", meta={"type": "External"})
            external_ids.add(eid)
        return eid

    # [maint-note:81fb76b8d3a5]
    # 维护导航：resolve_target；下列为本函数源码中的接口与路径索引。
    # 输入 target_name：str。
    # 输入 source_module：str。
    # 返回契约：str。
    # 分支条件：not target_name。
    # 分支条件：target_name in node_ids。
    # 分支条件：qualified in node_ids。
    # 分支条件：same_module。
    # 分支条件：candidates。
    # 分支条件：mod。
    # 返回路径：ensure_external('?')。
    # 返回路径：target_name。
    # 返回路径：qualified。
    # 返回路径：same_module[0]。
    # 返回路径：candidates[0]。
    # 返回路径：mod。
    # 返回路径：ensure_external(target_name)。
    # 直接调用：str(target_name or '').strip。
    # 直接调用：str。
    # 直接调用：ensure_external。
    # 直接调用：label_to_ids.get。
    # 直接调用：c.startswith。
    # 直接调用：resolve_module。
    # 集合派生：c 来自 candidates；过滤 c.startswith(source_module + '::')。
    # [/maint-note:81fb76b8d3a5]
    def resolve_target(target_name: str, source_module: str) -> str:
        target_name = str(target_name or "").strip()
        if not target_name:
            return ensure_external("?")

        # 1) 直接命中已有节点 id（Java 解析后 target 可能是 module::xxx 或 FQCN）
        if target_name in node_ids:
            return target_name

        # 2) 同一模块下的符号
        qualified = f"{source_module}::{target_name}"
        if qualified in node_ids:
            return qualified

        # 3) 全局 label 查找
        candidates = label_to_ids.get(target_name, [])
        # 优先同模块
        same_module = [c for c in candidates if c.startswith(source_module + "::")]
        if same_module:
            return same_module[0]
        if candidates:
            return candidates[0]

        # 4) 可能是模块路径（Java import target / C header）
        mod = resolve_module(target_name)
        if mod:
            return mod

        return ensure_external(target_name)

    for module, file_info in rag.items():
        module = _norm_path(module)
        if not module:
            continue

        for name, info in file_info.items():
            if not name:
                continue
            sid = module if name == module else f"{module}::{name}"
            if sid not in node_ids:
                continue
            if not isinstance(info, dict):
                continue

            # include / import 关系
            includes = info.get("include")
            if includes:
                if isinstance(includes, list):
                    for inc in includes:
                        if not inc:
                            continue
                        target = resolve_module(str(inc))
                        if target:
                            raw_edges.append((sid, target, "includes"))
                else:
                    target = resolve_module(str(includes))
                    if target:
                        raw_edges.append((sid, target, "includes"))

            # rels 关系
            rels = info.get("rels")
            if isinstance(rels, list):
                for r in rels:
                    if not isinstance(r, (list, tuple)):
                        continue
                    if len(r) >= 3:
                        target_name, rel_type = r[0], r[-1]
                    elif len(r) == 2:
                        target_name, rel_type = r[0], r[1]
                    else:
                        continue
                    if not target_name or not rel_type:
                        continue
                    tid = resolve_target(target_name, module)
                    raw_edges.append((sid, tid, str(rel_type).lower()))

    # ---------- 计算度 + 去重边 ----------
    degree = Counter()
    seen_edges: set = set()
    vis_edges: List[dict] = []
    for src, tgt, rel in raw_edges:
        if src == tgt:
            continue
        key = (src, tgt, rel)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        degree[src] += 1
        degree[tgt] += 1
        edge_index = len(vis_edges)
        vis_edges.append({
            "from": src,
            "to": tgt,
            "label": rel,
            "title": _html.escape(f"{rel}"),
            "dashes": False,
            "width": 1.5,
            "smooth": {
                "type": "curvedCW" if edge_index % 2 == 0 else "curvedCCW",
                "roundness": 0.28,
            },
            "font": {
                "size": 10,
                "color": "#374151",
                "background": "#ffffff",
                "strokeWidth": 4,
                "strokeColor": "#ffffff",
                "align": "horizontal",
            },
            "color": {"color": "#6b7280", "opacity": 0.75, "highlight": "#111827"},
        })

    max_deg = max(degree.values(), default=1) or 1
    group_color = {g: _color_for_group(g, GROUP_COLORS) for g in group_counter}

    # 补全 node 的 size / color / degree / font
    for n in vis_nodes:
        deg = degree.get(n["id"], 1)
        n["degree"] = deg
        grp = n["group"]
        if grp == "External":
            color = _EXTERNAL_COLOR
        else:
            color = group_color.get(grp, _EXTERNAL_COLOR)
        n["color"] = {
            "background": color,
            "border": color,
            "highlight": {"background": "#e5e7eb", "border": color},
        }
        n["size"] = round(10 + 30 * (deg / max_deg), 1)
        # 低度节点默认隐藏标签，减少 clutter；hover 和点击后仍可见
        n["font"]["size"] = 12 if deg >= max_deg * 0.12 else 0

    # Legend：按 group 统计
    legend_data = []
    for grp, count in sorted(group_counter.items(), key=lambda x: -x[1]):
        if grp == "External":
            color = _EXTERNAL_COLOR
        else:
            color = group_color.get(grp, _EXTERNAL_COLOR)
        legend_data.append({
            "group": grp,
            "label": grp,
            "color": color,
            "count": count,
        })

    stats = f"{len(vis_nodes)} nodes &middot; {len(vis_edges)} edges &middot; {len(group_counter)} groups"
    return vis_nodes, vis_edges, legend_data, stats


# ---------------------------------------------------------------------------
# 5. 生成完整 HTML
# ---------------------------------------------------------------------------

# [maint-note:7562ce30817f]
# 维护导航：_pyvis_light_options；下列为本函数源码中的接口与路径索引。
# 返回契约：dict。
# 返回路径：{'interaction': {'hover': True, 'tooltipDelay': 120, 'navigationButtons': True, 'keyboard': {'enabled': True, 'bindToWindow': False}, 'multiselect': True, 'dragNodes': True, 'hideEdgesOnDrag': True}, 'physics': {'enabled': True, 'solver': '。
# [/maint-note:7562ce30817f]
def _pyvis_light_options() -> dict:
    """Light-theme Pyvis options tuned for a code-knowledge-graph layout."""
    return {
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": {"enabled": True, "bindToWindow": False},
            "multiselect": True,
            "dragNodes": True,
            "hideEdgesOnDrag": True,
        },
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -55,
                "centralGravity": 0.010,
                "springLength": 130,
                "springConstant": 0.08,
                "damping": 0.45,
                "avoidOverlap": 0.18,
            },
            "stabilization": {
                "enabled": True,
                "iterations": 220,
                "updateInterval": 25,
                "fit": True,
            },
            "minVelocity": 0.6,
        },
        "nodes": {
            "borderWidth": 1.5,
            "shadow": False,
            "font": {
                "size": 12,
                "face": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                "color": "#111827",
            },
            "scaling": {"min": 10, "max": 30},
        },
        "edges": {
            # Curved edges keep relation labels away from the node-to-node
            # axis. The opaque label background hides the line underneath.
            "smooth": {"type": "curvedCW", "roundness": 0.28},
            "scaling": {"min": 1, "max": 3},
            "color": {"inherit": "from"},
            "font": {
                "size": 10,
                "color": "#6b7280",
                "background": "#ffffff",
                "strokeWidth": 4,
                "strokeColor": "#ffffff",
                "align": "middle",
            },
        },
    }


# [maint-note:7a9700d46df5]
# 维护导航：generate_html；下列为本函数源码中的接口与路径索引。
# 输入 vis_nodes：List[dict]。
# 输入 vis_edges：List[dict]。
# 输入 legend_data：List[dict]。
# 输入 stats：str。
# 输入 title：str；缺省表达式 'RAG Knowledge Graph'。
# 返回契约：str。
# 遍历边界：node ← vis_nodes。
# 遍历边界：edge ← vis_edges。
# 返回路径：pyvis_html。
# 直接调用：_js_safe。
# 直接调用：_html.escape。
# 直接调用：Network。
# 直接调用：net.set_options。
# 直接调用：json.dumps。
# 直接调用：_pyvis_light_options。
# 直接调用：node.get。
# 直接调用：color_obj.get。
# 直接调用：net.add_node。
# 直接调用：edge.get。
# 直接调用：net.add_edge。
# 直接调用：edge_color_obj.get。
# 直接调用：net.generate_html。
# 直接调用：pyvis_html.replace。
# 直接调用：re.sub。
# 直接调用：_html_styles。
# 直接调用：_html_script。
# [/maint-note:7a9700d46df5]
def generate_html(vis_nodes: List[dict], vis_edges: List[dict], legend_data: List[dict],
                  stats: str, title: str = "RAG Knowledge Graph") -> str:
    """Build a self-contained interactive HTML knowledge graph using Pyvis.

    Pyvis (via vis-network) provides force-directed layout, neighbor highlighting,
    navigation buttons, and an inlined library for offline viewing. We keep our
    custom sidebar (search / info / legend / stats) by post-processing Pyvis's
    output: rename ``mynetwork`` -> ``kg-graph``, inject the header before the
    container, and append the sidebar after it.
    """
    nodes_json = _js_safe(vis_nodes)
    edges_json = _js_safe(vis_edges)
    legend_json = _js_safe(legend_data)
    safe_title = _html.escape(title)

    # Imported lazily so the module is importable even when Pyvis is not yet
    # installed (the import error should surface at visualization time only).
    from pyvis.network import Network

    net = Network(
        height="100%",
        width="100%",
        directed=True,
        notebook=False,
        neighborhood_highlight=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="in_line",
        heading="",
    )
    net.set_options(json.dumps(_pyvis_light_options()))

    # Pyvis writes graph nodes/edges via JSON; mirror the colors produced by
    # ``rag_to_vis`` so the visual style is consistent.
    for node in vis_nodes:
        color_obj = node.get("color") or {}
        background = color_obj.get("background") or "#374151"
        border = color_obj.get("border") or background
        highlight = color_obj.get("highlight") or {
            "background": "#e5e7eb",
            "border": background,
        }
        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("title", ""),
            color={
                "background": background,
                "border": border,
                "highlight": highlight,
            },
            size=node.get("size", 10),
            shape="dot",
            group=node.get("group", "Other"),
            source_file=node.get("source_file", ""),
            degree=node.get("degree", 0),
            font=node.get("font", {"size": 12, "color": "#111827"}),
        )

    for edge in vis_edges:
        edge_color_obj = edge.get("color") or {}
        net.add_edge(
            edge["from"],
            edge["to"],
            title=edge.get("title", ""),
            label=edge.get("label", ""),
            color=edge_color_obj.get("color", "#6b7280"),
            arrows="to",
            width=edge.get("width", 2),
            smooth=edge.get("smooth"),
            font=edge.get("font"),
            labelHighlightBold=True,
        )

    pyvis_html = net.generate_html()

    # Pyvis mounts its graph on ``#mynetwork``; rename so it lands in our cell
    # and our CSS selectors match.
    pyvis_html = pyvis_html.replace("mynetwork", "kg-graph")

    # Replace Pyvis's default page title with our domain-specific one.
    pyvis_html = re.sub(
        r"<title>.*?</title>",
        f"<title>{safe_title}</title>",
        pyvis_html,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Inject our custom CSS right before </head>.
    pyvis_html = re.sub(
        r"</head>",
        _html_styles() + "</head>",
        pyvis_html,
        count=1,
        flags=re.IGNORECASE,
    )

    # Inject the header right after <body>, then open the grid wrapper.
    header_block = (
        '<div class="kg-header">'
        f"<strong>{safe_title}</strong>"
        f"<span>{stats}</span>"
        '<span class="hint">Drag nodes &middot; scroll to zoom &middot; hover for details</span>'
        "</div>"
        '<div class="kg-body">'
    )
    pyvis_html = re.sub(
        r"<body[^>]*>",
        lambda match: match.group(0) + header_block,
        pyvis_html,
        count=1,
        flags=re.IGNORECASE,
    )

    # Inject the sidebar + close the grid wrapper right before </body>.
    sidebar_block = (
        '<aside id="kg-sidebar">'
        '<div id="kg-search-wrap">'
        '<input id="kg-search" type="text" placeholder="Search nodes..." autocomplete="off">'
        '<div id="kg-search-results"></div>'
        "</div>"
        '<div id="kg-info-panel">'
        "<h3>Node Info</h3>"
        '<div id="kg-info-content"><span class="empty">Click a node to inspect it</span></div>'
        "</div>"
        '<div id="kg-legend-wrap">'
        "<h3>Node Types</h3>"
        '<div id="kg-legend-controls">'
        '<label><input type="checkbox" id="kg-select-all-cb" checked '
        'onchange="toggleAllGroups(!this.checked)">Select All</label>'
        "</div>"
        '<div id="kg-legend"></div>'
        "</div>"
        f'<div id="kg-stats">{stats}</div>'
        "</aside>"
        "</div>"  # close .kg-body
        + _html_script(nodes_json, edges_json, legend_json)
    )
    pyvis_html = re.sub(
        r"</body>",
        sidebar_block + "</body>",
        pyvis_html,
        count=1,
        flags=re.IGNORECASE,
    )

    return pyvis_html


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

# [maint-note:8f696b71b1f9]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 返回契约：None。
# 直接调用：argparse.ArgumentParser。
# 直接调用：parser.add_argument。
# 直接调用：parser.parse_args。
# 直接调用：Path(args.input).expanduser().resolve。
# 直接调用：Path(args.input).expanduser。
# 直接调用：Path。
# 直接调用：Path(args.output).expanduser().resolve。
# 直接调用：Path(args.output).expanduser。
# 直接调用：load_rag_graph。
# 直接调用：rag_to_vis。
# 直接调用：generate_html。
# 直接调用：output_path.parent.mkdir。
# 直接调用：output_path.write_text。
# 直接调用：print。
# 直接调用：len。
# [/maint-note:8f696b71b1f9]
def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 RAG 项目级 parse_res JSON 转换成可交互的 HTML 知识图谱"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="输入的 RAG graph JSON 文件路径（如 rag/c/CEval/call_graph/foo.json）")
    parser.add_argument("-o", "--output", required=True,
                        help="输出的 HTML 文件路径（如 foo_graph.html）")
    parser.add_argument("--title", default="RAG Knowledge Graph",
                        help="HTML 页面标题")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rag = load_rag_graph(input_path)
    vis_nodes, vis_edges, legend_data, stats = rag_to_vis(rag)

    html = generate_html(vis_nodes, vis_edges, legend_data, stats, title=args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"已生成可视化图谱：{output_path}")
    print(f"  节点数：{len(vis_nodes)}，边数：{len(vis_edges)}，类型分组：{len(legend_data)}")
    print(f"  直接用浏览器打开该 HTML 文件即可查看")


if __name__ == "__main__":
    main()
