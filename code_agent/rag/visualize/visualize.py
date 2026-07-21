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


def _sanitize(text: Any, max_len: int = _MAX_LABEL_LEN) -> str:
    if text is None:
        return ""
    s = _Control_RE.sub("", str(text))
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _js_safe(obj: Any) -> str:
    """转义 </script>，防止嵌入 JSON 时破坏 <script> 标签。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


# ---------------------------------------------------------------------------
# 3. HTML 模板（从 graphify/export.py 提取并简化）
# ---------------------------------------------------------------------------

def _html_styles() -> str:
    return """<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f3f4f6; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; height: 100vh; overflow: hidden; }
  #graph { flex: 1; background: #ffffff; }
  #sidebar { width: 320px; background: #ffffff; border-left: 1px solid #d1d5db; display: flex; flex-direction: column; overflow: hidden; box-shadow: -2px 0 8px rgba(0,0,0,0.04); }
  #search-wrap { padding: 12px; border-bottom: 1px solid #e5e7eb; }
  #search { width: 100%; background: #ffffff; border: 1px solid #d1d5db; color: #111827; padding: 7px 10px; border-radius: 6px; font-size: 13px; outline: none; }
  #search:focus { border-color: #2563EB; box-shadow: 0 0 0 2px rgba(37,99,235,0.15); }
  #search-results { max-height: 140px; overflow-y: auto; padding: 4px 12px; border-bottom: 1px solid #e5e7eb; display: none; background: #ffffff; }
  .search-item { padding: 4px 6px; cursor: pointer; border-radius: 4px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .search-item:hover { background: #eff6ff; }
  #info-panel { padding: 14px; border-bottom: 1px solid #e5e7eb; min-height: 140px; }
  #info-panel h3 { font-size: 13px; color: #6b7280; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  #info-content { font-size: 13px; color: #374151; line-height: 1.6; }
  #info-content .field { margin-bottom: 5px; }
  #info-content .field b { color: #111827; }
  #info-content .empty { color: #9ca3af; font-style: italic; }
  .neighbor-link { display: block; padding: 2px 6px; margin: 2px 0; border-radius: 3px; cursor: pointer; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid #d1d5db; }
  .neighbor-link:hover { background: #eff6ff; }
  #neighbors-list { max-height: 160px; overflow-y: auto; margin-top: 4px; }
  #legend-wrap { flex: 1; overflow-y: auto; padding: 12px; }
  #legend-wrap h3 { font-size: 13px; color: #6b7280; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .legend-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; border-radius: 4px; font-size: 12px; }
  .legend-item:hover { background: #f3f4f6; padding-left: 4px; }
  .legend-item.dimmed { opacity: 0.35; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08); }
  .legend-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #111827; }
  .legend-count { color: #9ca3af; font-size: 11px; }
  #stats { padding: 10px 14px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af; }
  #legend-controls { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding: 4px 0; }
  #legend-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 12px; color: #4b5563; user-select: none; }
  #legend-controls label:hover { color: #111827; }
  .legend-cb, #select-all-cb { appearance: none; -webkit-appearance: none; width: 14px; height: 14px; border: 1.5px solid #d1d5db; border-radius: 3px; background: #ffffff; cursor: pointer; position: relative; flex-shrink: 0; }
  .legend-cb:checked, #select-all-cb:checked { background: #2563EB; border-color: #2563EB; }
  .legend-cb:checked::after, #select-all-cb:checked::after { content: ''; position: absolute; left: 3.5px; top: 1px; width: 4px; height: 7px; border: solid #fff; border-width: 0 2px 2px 0; transform: rotate(45deg); }
  #select-all-cb:indeterminate { background: #2563EB; border-color: #2563EB; }
  #select-all-cb:indeterminate::after { content: ''; position: absolute; left: 2px; top: 5px; width: 8px; height: 2px; background: #fff; border: none; transform: none; }
</style>"""


def _html_script(nodes_json: str, edges_json: str, legend_json: str) -> str:
    return f"""<script>
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};
const LEGEND = {legend_json};

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}}

const nodesDS = new vis.DataSet(RAW_NODES.map(n => {{
  return {{ id: n.id, label: n.label, color: n.color, size: n.size,
    font: n.font, title: n.title,
    _group: n.group, _source_file: n.source_file, _degree: n.degree }};
}}));

const edgesDS = new vis.DataSet(RAW_EDGES.map((e, i) => {{
  return {{ id: i, from: e.from, to: e.to,
    label: e.label || '',
    title: e.title,
    dashes: e.dashes,
    width: e.width,
    color: e.color,
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }} }};
}}));

const container = document.getElementById('graph');
const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -60,
      centralGravity: 0.005,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.4,
      avoidOverlap: 0.8,
    }},
    stabilization: {{ iterations: 200, fit: true }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true,
    navigationButtons: false,
    keyboard: false,
  }},
  nodes: {{ shape: 'dot', borderWidth: 1.5 }},
  edges: {{ smooth: {{ type: 'continuous', roundness: 0.2 }}, selectionWidth: 3 }},
}});

network.once('stabilizationIterationsDone', () => {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});

function showInfo(nodeId) {{
  const n = nodesDS.get(nodeId);
  if (!n) return;
  const neighborIds = network.getConnectedNodes(nodeId);
  const neighborItems = neighborIds.map(nid => {{
    const nb = nodesDS.get(nid);
    const color = nb ? nb.color.background : '#374151';
    return `<span class="neighbor-link" style="border-left-color:${{esc(color)}}" onclick="focusNode(${{JSON.stringify(nid)}})">${{esc(nb ? nb.label : nid)}}</span>`;
  }}).join('');
  document.getElementById('info-content').innerHTML = `
    <div class="field"><b>${{esc(n.label)}}</b></div>
    <div class="field">Group: ${{esc(n._group || 'unknown')}}</div>
    <div class="field">Source: ${{esc(n._source_file || '-')}}</div>
    <div class="field">Degree: ${{n._degree}}</div>
    ${{neighborIds.length ? `<div class="field" style="margin-top:8px;color:#6b7280;font-size:11px">Neighbors (${{neighborIds.length}})</div><div id="neighbors-list">${{neighborItems}}</div>` : ''}}
  `;
}}

function focusNode(nodeId) {{
  network.focus(nodeId, {{ scale: 1.4, animation: true }});
  network.selectNodes([nodeId]);
  showInfo(nodeId);
}}

let hoveredNodeId = null;
network.on('hoverNode', params => {{ hoveredNodeId = params.node; container.style.cursor = 'pointer'; }});
network.on('blurNode', () => {{ hoveredNodeId = null; container.style.cursor = 'default'; }});
container.addEventListener('click', () => {{
  if (hoveredNodeId !== null) {{ showInfo(hoveredNodeId); network.selectNodes([hoveredNodeId]); }}
}});
network.on('click', params => {{
  if (params.nodes.length > 0) {{ showInfo(params.nodes[0]); }}
  else if (hoveredNodeId === null) {{ document.getElementById('info-content').innerHTML = '<span class="empty">Click a node to inspect it</span>'; }}
}});

const searchInput = document.getElementById('search');
const searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', () => {{
  const q = searchInput.value.toLowerCase().trim();
  searchResults.innerHTML = '';
  if (!q) {{ searchResults.style.display = 'none'; return; }}
  const matches = RAW_NODES.filter(n => n.label.toLowerCase().includes(q)).slice(0, 20);
  if (!matches.length) {{ searchResults.style.display = 'none'; return; }}
  searchResults.style.display = 'block';
  matches.forEach(n => {{
    const el = document.createElement('div');
    el.className = 'search-item';
    el.textContent = n.label;
    el.style.borderLeft = `3px solid ${{n.color.background}}`;
    el.style.paddingLeft = '8px';
    el.onclick = () => {{
      network.focus(n.id, {{ scale: 1.5, animation: true }});
      network.selectNodes([n.id]);
      showInfo(n.id);
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
const selectAllCb = document.getElementById('select-all-cb');

function updateSelectAllState() {{
  const total = LEGEND.length;
  const hidden = hiddenGroups.size;
  selectAllCb.checked = hidden === 0;
  selectAllCb.indeterminate = hidden > 0 && hidden < total;
}}

function toggleAllGroups(hide) {{
  document.querySelectorAll('.legend-item').forEach(item => {{ hide ? item.classList.add('dimmed') : item.classList.remove('dimmed'); }});
  document.querySelectorAll('.legend-cb').forEach(cb => {{ cb.checked = !hide; }});
  LEGEND.forEach(g => {{ if (hide) hiddenGroups.add(g.group); else hiddenGroups.delete(g.group); }});
  const updates = RAW_NODES.map(n => ({{ id: n.id, hidden: hide }}));
  nodesDS.update(updates);
  updateSelectAllState();
}}

const legendEl = document.getElementById('legend');
LEGEND.forEach(g => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.className = 'legend-cb';
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
    nodesDS.update(updates);
    updateSelectAllState();
  }});
  item.innerHTML = `<div class="legend-dot" style="background:${{g.color}}"></div>
    <span class="legend-label">${{g.label}}</span>
    <span class="legend-count">${{g.count}}</span>`;
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

def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def load_rag_graph(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"RAG graph 必须是 dict，实际为 {type(data).__name__}")
    return data


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


def rag_to_vis(rag: Dict[str, Dict[str, Any]]) -> Tuple[List[dict], List[dict], List[dict], str]:
    """把 RAG parse_res 转换成 vis.js 需要的 nodes/edges/legend/stats。"""

    vis_nodes: List[dict] = []
    node_ids: set = set()
    label_to_ids: Dict[str, List[str]] = defaultdict(list)
    module_to_package: Dict[str, str] = {}

    group_counter: Counter = Counter()

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

    def ensure_external(target_name: str) -> str:
        eid = f"external::{_sanitize(target_name, max_len=80)}"
        if eid not in node_ids:
            add_node(eid, label=target_name, source_file="", group="External", meta={"type": "External"})
            external_ids.add(eid)
        return eid

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
        vis_edges.append({
            "from": src,
            "to": tgt,
            "label": rel,
            "title": _html.escape(f"{rel}"),
            "dashes": False,
            "width": 2,
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

def generate_html(vis_nodes: List[dict], vis_edges: List[dict], legend_data: List[dict],
                  stats: str, title: str = "RAG Knowledge Graph") -> str:
    nodes_json = _js_safe(vis_nodes)
    edges_json = _js_safe(vis_edges)
    legend_json = _js_safe(legend_data)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"
        integrity="sha384-Ux6phic9PEHJ38YtrijhkzyJ8yQlH8i/+buBR8s3mAZOJrP1gwyvAcIYl3GWtpX1"
        crossorigin="anonymous"></script>
{_html_styles()}
</head>
<body>
<div id="graph"></div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search nodes..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <h3>Node Info</h3>
    <div id="info-content"><span class="empty">Click a node to inspect it</span></div>
  </div>
  <div id="legend-wrap">
    <h3>Node Types</h3>
    <div id="legend-controls">
      <label><input type="checkbox" id="select-all-cb" checked onchange="toggleAllGroups(!this.checked)">Select All</label>
    </div>
    <div id="legend"></div>
  </div>
  <div id="stats">{stats}</div>
</div>
{_html_script(nodes_json, edges_json, legend_json)}
</body>
</html>"""


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

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
