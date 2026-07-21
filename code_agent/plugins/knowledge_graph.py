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


@register_plugin
class KnowledgeGraphPlugin(FeaturePlugin):
    """Generate an interactive vis.js knowledge graph for a C/C++ or Java project."""

    @property
    def metadata(self) -> FeatureMetadata:
        return FeatureMetadata(
            name="knowledge_graph",
            label="Knowledge Graph",
            description="Parse a C/C++ or Java project into an interactive vis.js knowledge graph.",
            execution_mode=ExecutionMode.DIRECT,
        )

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
