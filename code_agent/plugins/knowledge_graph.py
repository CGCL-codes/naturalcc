from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from code_agent.plugins.base import (
    ConfigField,
    ConfigFieldType,
    ExecutionContext,
    ExecutionMode,
    FeatureMetadata,
    FeaturePlugin,
    PluginResult,
)
from code_agent.plugins.registry import register_plugin
from code_agent.rag.visualize import generate_html, rag_to_vis
from code_agent.rag.visualize.generate_graph import parse_c_project, parse_java_project


# [maint-note:ca1772f4980e]
# 数据与行为导航：KnowledgeGraphPlugin。
# 继承接口：FeaturePlugin。
# [/maint-note:ca1772f4980e]
@register_plugin
class KnowledgeGraphPlugin(FeaturePlugin):
    """Generate an interactive vis.js knowledge graph for a C/C++ or Java project."""

    # [maint-note:019ba5feb702]
    # 维护导航：metadata；下列为本函数源码中的接口与路径索引。
    # 返回契约：FeatureMetadata。
    # 返回路径：FeatureMetadata(name='knowledge_graph', label='Knowledge Graph', description='Parse a C/C++ or Java project into an interactive vis.js knowledge graph.', execution_mode=ExecutionMode.DIRECT)。
    # 直接调用：FeatureMetadata。
    # [/maint-note:019ba5feb702]
    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="knowledge_graph",
            label="Knowledge Graph",
            description="Parse a C/C++ or Java project into an interactive vis.js knowledge graph.",
            execution_mode=ExecutionMode.DIRECT,
        )

    # [maint-note:483510360ba5]
    # 维护导航：config_schema；下列为本函数源码中的接口与路径索引。
    # 返回契约：List[ConfigField]。
    # 返回路径：[ConfigField(name='language', label='Language', type=ConfigFieldType.SELECT, required=True, default='c', options=[{'value': 'c', 'label': 'C / C++'}, {'value': 'java', 'label': 'Java'}], help_text='Project language to parse.'), ConfigField(。
    # 直接调用：ConfigField。
    # [/maint-note:483510360ba5]
    @property
    def config_schema(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="language",
                label="Language",
                type=ConfigFieldType.SELECT,
                required=True,
                default="c",
                options=[
                    {"value": "c", "label": "C / C++"},
                    {"value": "java", "label": "Java"},
                ],
                help_text="Project language to parse.",
            ),
            ConfigField(
                name="output_name",
                label="Output Filename",
                type=ConfigFieldType.TEXT,
                required=False,
                default="knowledge_graph.html",
                placeholder="knowledge_graph.html",
                help_text="Name of the generated HTML file (relative to project directory).",
            ),
            ConfigField(
                name="title",
                label="Page Title",
                type=ConfigFieldType.TEXT,
                required=False,
                default="Project Knowledge Graph",
                help_text="Title shown in the generated HTML page.",
            ),
            ConfigField(
                name="save_json",
                label="Save Intermediate JSON",
                type=ConfigFieldType.SWITCH,
                required=False,
                default=False,
                help_text="Also save the intermediate RAG JSON to disk.",
            ),
            ConfigField(
                name="json_name",
                label="JSON Filename",
                type=ConfigFieldType.TEXT,
                required=False,
                default="knowledge_graph.json",
                help_text="Name of the intermediate JSON file when save_json is enabled.",
            ),
        ]

    # [maint-note:d00cc61e69f1]
    # 维护导航：validate；下列为本函数源码中的接口与路径索引。
    # 输入 config：Dict[str, Any]。
    # 输入 files：Optional[Dict[str, Any]]；缺省表达式 None。
    # 返回契约：Optional[str]。
    # 分支条件：base。
    # 分支条件：language not in {'c', 'java'}。
    # 分支条件：not output_name。
    # 分支条件：not output_name.lower().endswith(('.html', '.htm'))。
    # 分支条件：save_json。
    # 分支条件：not json_name。
    # 分支条件：not json_name.lower().endswith('.json')。
    # 返回路径：base。
    # 返回路径：'language must be one of: c, java'。
    # 返回路径：'output_name cannot be empty'。
    # 返回路径：'output_name must end with .html or .htm'。
    # 返回路径：'json_name cannot be empty when save_json is enabled'。
    # 返回路径：'json_name must end with .json'。
    # 返回路径：None。
    # 直接调用：super().validate。
    # 直接调用：super。
    # 直接调用：str(config.get('language', 'c')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('output_name', 'knowledge_graph.html')).strip。
    # 直接调用：output_name.lower().endswith。
    # 直接调用：output_name.lower。
    # 直接调用：bool。
    # 直接调用：str(config.get('json_name', 'knowledge_graph.json')).strip。
    # 直接调用：json_name.lower().endswith。
    # 直接调用：json_name.lower。
    # [/maint-note:d00cc61e69f1]
    def validate(self, config: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> Optional[str]:
        base = super().validate(config, files)
        if base:
            return base

        language = str(config.get("language", "c")).strip()
        if language not in {"c", "java"}:
            return "language must be one of: c, java"

        output_name = str(config.get("output_name", "knowledge_graph.html")).strip()
        if not output_name:
            return "output_name cannot be empty"
        if not output_name.lower().endswith((".html", ".htm")):
            return "output_name must end with .html or .htm"

        save_json = bool(config.get("save_json", False))
        if save_json:
            json_name = str(config.get("json_name", "knowledge_graph.json")).strip()
            if not json_name:
                return "json_name cannot be empty when save_json is enabled"
            if not json_name.lower().endswith(".json"):
                return "json_name must end with .json"

        return None

    # [maint-note:734cba5ad204]
    # 维护导航：execute；下列为本函数源码中的接口与路径索引。
    # 输入 context：ExecutionContext。
    # 返回契约：Generator[Any, None, None]。
    # 分支条件：not project_dir.is_dir()。
    # 分支条件：language == 'c'。
    # 分支条件：save_json。
    # 分支条件：save_json and json_path。
    # 返回路径：无返回值。
    # 捕获异常：Exception。
    # 直接调用：str(config.get('language', 'c')).strip。
    # 直接调用：str。
    # 直接调用：config.get。
    # 直接调用：str(config.get('output_name', 'knowledge_graph.html')).strip。
    # 直接调用：str(config.get('title', 'Project Knowledge Graph')).strip。
    # 直接调用：bool。
    # 直接调用：str(config.get('json_name', 'knowledge_graph.json')).strip。
    # 直接调用：Path(context.project_dir).expanduser().resolve。
    # 直接调用：Path(context.project_dir).expanduser。
    # 直接调用：Path。
    # 直接调用：project_dir.is_dir。
    # 直接调用：PluginResult。
    # 直接调用：parse_c_project。
    # 直接调用：parse_java_project。
    # 直接调用：self._format_parsing_error。
    # 直接调用：len。
    # 直接调用：json_path.write_text。
    # 直接调用：json.dumps。
    # 直接调用：files_modified.append。
    # 直接调用：rag_to_vis。
    # 直接调用：generate_html。
    # 直接调用：traceback.format_exc。
    # 直接调用：html_path.write_text。
    # 直接调用：log_lines.append。
    # 直接调用：'\n'.join。
    # 生成器交付：PluginResult(success=False, message=f'Project directory does not exist: {project_dir}', log=f'[KnowledgeGraph] Error: Directory not found: {project_dir}\n')。
    # 生成器交付：f'[KnowledgeGraph] Starting knowledge graph generation for {language} project: {project_dir}\n'。
    # 生成器交付：'[KnowledgeGraph] Parsing source files...\n'。
    # 生成器交付：PluginResult(success=False, message=f'Parsing failed: {exc}', log=error_log)。
    # 生成器交付：f'[KnowledgeGraph] Parsed {len(rag)} module(s).\n'。
    # 生成器交付：f'[KnowledgeGraph] Saved intermediate JSON to {json_path}\n'。
    # 生成器交付：PluginResult(success=False, message=f'Failed to write JSON: {exc}', log=f'[KnowledgeGraph] Error writing JSON: {exc}\n')。
    # 生成器交付：'[KnowledgeGraph] Building vis.js graph...\n'。
    # 生成器交付：PluginResult(success=False, message=f'Visualization generation failed: {exc}', log=f'[KnowledgeGraph] Error generating visualization: {exc}\n{traceback.format_exc()}')。
    # 生成器交付：f'[KnowledgeGraph] Generated HTML with {len(nodes)} node(s), {len(edges)} edge(s).\n'。
    # 生成器交付：PluginResult(success=False, message=f'Failed to write HTML: {exc}', log=f'[KnowledgeGraph] Error writing HTML: {exc}\n')。
    # 生成器交付：f'[KnowledgeGraph] HTML saved to {html_path}\n'。
    # 生成器交付：PluginResult(success=True, message=f'Knowledge graph saved to {output_name}', log='\n'.join(log_lines) + '\n', files_modified=files_modified, report=f'Generated knowledge graph for {language} project\n{stats}', artifacts={'html': html, 'htm。
    # [/maint-note:734cba5ad204]
    def execute(self, context: ExecutionContext) -> Generator[Any, None, None]:
        config = context.feature_config or {}
        language = str(config.get("language", "c")).strip()
        output_name = str(config.get("output_name", "knowledge_graph.html")).strip()
        title = str(config.get("title", "Project Knowledge Graph")).strip()
        save_json = bool(config.get("save_json", False))
        json_name = str(config.get("json_name", "knowledge_graph.json")).strip()

        project_dir = Path(context.project_dir).expanduser().resolve()
        if not project_dir.is_dir():
            yield PluginResult(
                success=False,
                message=f"Project directory does not exist: {project_dir}",
                log=f"[KnowledgeGraph] Error: Directory not found: {project_dir}\n",
            )
            return

        yield f"[KnowledgeGraph] Starting knowledge graph generation for {language} project: {project_dir}\n"
        yield "[KnowledgeGraph] Parsing source files...\n"

        try:
            if language == "c":
                rag = parse_c_project(str(project_dir))
            else:
                rag = parse_java_project(str(project_dir))
        except Exception as exc:
            error_log = self._format_parsing_error(language, exc)
            yield PluginResult(
                success=False,
                message=f"Parsing failed: {exc}",
                log=error_log,
            )
            return

        yield f"[KnowledgeGraph] Parsed {len(rag)} module(s).\n"

        files_modified: List[str] = []
        json_path: Optional[Path] = None

        if save_json:
            json_path = project_dir / json_name
            try:
                json_path.write_text(
                    json.dumps(rag, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                files_modified.append(str(json_path))
                yield f"[KnowledgeGraph] Saved intermediate JSON to {json_path}\n"
            except Exception as exc:
                yield PluginResult(
                    success=False,
                    message=f"Failed to write JSON: {exc}",
                    log=f"[KnowledgeGraph] Error writing JSON: {exc}\n",
                )
                return

        yield "[KnowledgeGraph] Building vis.js graph...\n"
        try:
            nodes, edges, legend, stats = rag_to_vis(rag)
            html = generate_html(nodes, edges, legend, stats, title=title)
        except Exception as exc:
            yield PluginResult(
                success=False,
                message=f"Visualization generation failed: {exc}",
                log=f"[KnowledgeGraph] Error generating visualization: {exc}\n{traceback.format_exc()}",
            )
            return

        yield f"[KnowledgeGraph] Generated HTML with {len(nodes)} node(s), {len(edges)} edge(s).\n"

        html_path = project_dir / output_name
        try:
            html_path.write_text(html, encoding="utf-8")
            files_modified.append(str(html_path))
        except Exception as exc:
            yield PluginResult(
                success=False,
                message=f"Failed to write HTML: {exc}",
                log=f"[KnowledgeGraph] Error writing HTML: {exc}\n",
            )
            return

        yield f"[KnowledgeGraph] HTML saved to {html_path}\n"

        log_lines = [
            "[KnowledgeGraph] Knowledge graph generated successfully.",
            f"- Language: {language}",
            f"- Modules: {len(rag)}",
            f"- Nodes: {len(nodes)}",
            f"- Edges: {len(edges)}",
            f"- HTML: {html_path}",
        ]
        if save_json and json_path:
            log_lines.append(f"- JSON: {json_path}")

        yield PluginResult(
            success=True,
            message=f"Knowledge graph saved to {output_name}",
            log="\n".join(log_lines) + "\n",
            files_modified=files_modified,
            report=f"Generated knowledge graph for {language} project\n{stats}",
            artifacts={
                "html": html,
                "html_path": str(html_path),
                "nodes": len(nodes),
                "edges": len(edges),
                "modules": len(rag),
            },
        )

    # [maint-note:af5353e8531a]
    # 维护导航：_format_parsing_error；下列为本函数源码中的接口与路径索引。
    # 输入 language：str。
    # 输入 exc：Exception。
    # 返回契约：str。
    # 分支条件：'libclang' in str(exc).lower() or 'clang' in str(exc).lower()。
    # 分支条件：language == 'java' and ('tree' in str(exc).lower() or 'sitter' in str(exc).lower())。
    # 返回路径：'\n'.join(lines) + '\n'。
    # 直接调用：str(exc).lower。
    # 直接调用：str。
    # 直接调用：lines.append。
    # 直接调用：traceback.format_exc。
    # 直接调用：'\n'.join。
    # [/maint-note:af5353e8531a]
    def _format_parsing_error(self, language: str, exc: Exception) -> str:
        lines = [f"[KnowledgeGraph] Error during parsing: {exc}"]
        if "libclang" in str(exc).lower() or "clang" in str(exc).lower():
            lines.append(
                "[KnowledgeGraph] Hint: C/C++ parsing requires libclang. "
                "Install it with: sudo apt install libclang1-18"
            )
        if language == "java" and ("tree" in str(exc).lower() or "sitter" in str(exc).lower()):
            lines.append(
                "[KnowledgeGraph] Hint: Java parsing requires tree-sitter-java. "
                "Install it with: uv pip install tree-sitter-java"
            )
        lines.append(traceback.format_exc())
        return "\n".join(lines) + "\n"
