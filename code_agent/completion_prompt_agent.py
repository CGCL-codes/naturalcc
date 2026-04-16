from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CompletionPromptAgent:
    """
    基于 CProjectParser 的上层 Prompt Agent：
    1. 接收用户指令
    2. 识别补全类型
    3. 从 parse_res 中抽取相关信息
    4. 输出适合 LLM / Aider 使用的 prompt
    """

    def __init__(self):
        self.project_parser = None
        self.language: str = "c"
        self.proj_dir: Optional[str] = None
        self.parse_res: Optional[Dict] = None
        self.searcher = None

    @staticmethod
    def _infer_language(file_path: Optional[str], explicit_language: Optional[str] = None) -> str:
        if explicit_language:
            lang = explicit_language.strip().lower()
            if lang in {"c", "java"}:
                return lang

        suffix = Path(file_path or "").suffix.lower()
        if suffix == ".java":
            return "java"
        return "c"

    # ---------------------------
    # 项目加载
    # ---------------------------
    def load_project(self, proj_dir: str, language: str = "c"):
        if not proj_dir:
            raise ValueError("proj_dir 不能为空")

        if language not in {"c", "java"}:
            raise ValueError(f"不支持的语言: {language}")

        if language == "java":
            from .rag.java.java_project_parser_ts import JavaProjectParserTS

            self.project_parser = JavaProjectParserTS()
            self.searcher = getattr(self.project_parser, "searcher", None)
        else:
            from .rag.c.preprocess import CProjectParser

            self.project_parser = CProjectParser()
            self.searcher = None

        self.proj_dir = proj_dir
        self.language = language
        self.parse_res = self.project_parser.parse_dir(proj_dir)
        if self.language == "c":
            self.searcher = getattr(self.project_parser, "proj_searcher", None)

    # ---------------------------
    # 用户请求入口
    # ---------------------------
    def build_prompt(self, request: Dict) -> str:
        """
        request 示例：
        {
            "project_dir": "/path/to/project",
            "user_instruction": "补全 node 的成员",
            "file_path": "src/list.c",
            "symbol": "node",
            "completion_type": "member",   # 可选
            "prefix": "ne"                 # 可选
        }
        """
        proj_dir = request.get("project_dir")
        file_path = request.get("file_path")
        language = self._infer_language(file_path, request.get("language"))
        if proj_dir and (
            self.proj_dir != proj_dir
            or self.parse_res is None
            or self.language != language
        ):
            self.load_project(proj_dir, language=language)

        if self.parse_res is None:
            raise ValueError("项目尚未加载，请提供有效的 project_dir")

        user_instruction = request.get("user_instruction", "").strip()
        symbol = request.get("symbol")
        prefix = request.get("prefix", "") or ""
        completion_type = request.get("completion_type")

        if self.language == "java":
            return self._build_java_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                completion_type=completion_type,
                prefix=prefix,
            )

        # 如果用户没有显式提供 symbol，则从自然语言中自动提取
        if not symbol:
            symbol = self._infer_symbol_from_instruction(user_instruction)

        # 如果用户没有显式提供 completion_type，则自动推断
        if not completion_type:
            completion_type = self._infer_completion_type(user_instruction, symbol)
            
        if completion_type == "member":
            return self._build_member_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix,
            )

        if completion_type == "variable":
            return self._build_variable_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                prefix=prefix,
            )

        if completion_type == "function":
            return self._build_function_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix,
            )

        if completion_type == "function_body":
            return self._build_function_body_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
            )

        if completion_type == "type":
            return self._build_type_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix,
            )

        return self._build_fallback_prompt(
            user_instruction=user_instruction,
            file_path=file_path,
            symbol=symbol,
            prefix=prefix,
        )

    def _build_java_completion_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
        completion_type: Optional[str],
        prefix: str = "",
    ) -> str:
        file_info = self._get_file_info(file_path) if file_path else None

        symbol_candidates: List[str] = []
        if file_info:
            for name, info in file_info.items():
                if not name or name == "":
                    continue
                ntype = info.get("type")
                if ntype in {
                    "Class",
                    "Interface",
                    "Enum",
                    "Annotation",
                    "Record",
                    "Method",
                    "Constructor",
                    "Field",
                }:
                    symbol_candidates.append(name)

        if symbol:
            symbol_candidates = [x for x in symbol_candidates if symbol in x]
        if prefix:
            symbol_candidates = [x for x in symbol_candidates if x.split(".")[-1].startswith(prefix)]

        symbol_candidates = sorted(symbol_candidates)[:40]

        related_defs: List[str] = []
        for fpath, finfo in (self.parse_res or {}).items():
            for name, info in finfo.items():
                if not name or info.get("type") not in {"Class", "Interface", "Enum", "Record", "Method"}:
                    continue
                if symbol and symbol not in name:
                    continue
                if prefix and not name.split(".")[-1].startswith(prefix):
                    continue
                snippet = info.get("def") or ""
                if not snippet:
                    continue
                related_defs.append(f"/* {fpath} */\n{snippet}")
                if len(related_defs) >= 20:
                    break
            if len(related_defs) >= 20:
                break

        completion_hint = completion_type or "auto"
        return f"""你是一个 Java 代码补全助手。
任务：基于项目内真实符号与定义，在目标文件中直接完成补全或修正。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【补全类型】
{completion_hint}

【目标符号】
{symbol}

【前缀】
{prefix}

【当前文件候选符号】
{chr(10).join(f'- {x}' for x in symbol_candidates) if symbol_candidates else '- 无'}

【项目内相关定义】
{chr(10).join(related_defs) if related_defs else '无匹配定义'}

【执行要求】
1. 不要只输出解释或候选列表；请直接修改目标文件中的代码。
2. 优先复用项目中已有类、方法、字段与导入，不要臆造不存在的 API。
3. 保持现有代码风格，做最小必要修改。
4. 如需补充 import、类型声明或异常处理，可一并最小修改。
5. 如果代码存在未完成表达式、占位符或 TODO，请直接补全为可工作的实现。
"""

    def _infer_symbol_from_instruction(self, instruction: str) -> Optional[str]:
        """
        从用户自然语言里自动提取目标函数/符号名。
        例如：
        - 补全 parse_flags 函数
        - 完善 parse_flags 实现
        - 实现 parse_flags
        """
        if not instruction:
            return None

        import re

        patterns = [
            r"补全\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*函数",
            r"完善\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:函数|实现)?",
            r"实现\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:函数)?",
            r"补全\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"函数\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*函数",
        ]

        for pattern in patterns:
            m = re.search(pattern, instruction)
            if m:
                return m.group(1)

        return None
    # ---------------------------
    # 识别补全类型
    # ---------------------------
    def _infer_completion_type(self, instruction: str, symbol: Optional[str]) -> str:
        instruction = instruction or ""

        # 优先识别“函数体/实现补全”
        if any(
            x in instruction
            for x in [
                "函数体",
                "函数实现",
                "补全函数内容",
                "完善函数实现",
                "实现函数",
                "补全整个函数",
                "补全完整函数",
                "补全",
                "完善",
                "实现",
            ]
        ):
            if symbol:
                return "function_body"

        if any(x in instruction for x in ["成员", "字段", "struct字段", "成员补全"]) or "->" in instruction or "." in instruction:
            return "member"

        if any(x in instruction for x in ["局部变量", "全局变量", "变量补全", "变量名", "变量"]):
            return "variable"

        if any(x in instruction for x in ["函数签名", "函数声明", "函数名"]):
            return "function"

        if any(x in instruction for x in ["类型", "typedef", "struct", "union", "enum"]):
            return "type"

        # 只要能识别到 symbol，默认按“函数体补全”处理
        if symbol:
            return "function_body"

        # 默认也走函数体补全，比函数签名更实用
        return "function_body"

    # ---------------------------
    # 工具函数
    # ---------------------------
    def _get_file_info(self, file_path: str) -> Optional[Dict]:
        if not file_path or not self.parse_res:
            return None
        normalized_file_path = str(file_path).replace("\\", "/")
        return self.parse_res.get(normalized_file_path)

    def _iter_symbols_in_file(self, file_info: Dict):
        for name, info in file_info.items():
            if not name:
                continue
            yield name, info

    def _find_symbol_info(self, file_path: str, symbol: str) -> Optional[Tuple[str, Dict]]:
        file_info = self._get_file_info(file_path)
        if not file_info or not symbol:
            return None

        if symbol in file_info:
            return symbol, file_info[symbol]

        for name, info in self._iter_symbols_in_file(file_info):
            if name.endswith(f".{symbol}"):
                return name, info

        return None

    def _find_type_of_symbol(self, file_path: str, symbol: str) -> Optional[str]:
        res = self._find_symbol_info(file_path, symbol)
        if not res:
            return None

        _, info = res
        rels = info.get("rels", [])
        for item in rels:
            if len(item) >= 2 and item[-1] == "Typeof":
                return item[0]
        return None

    def _find_struct_like_by_type(self, file_path: str, type_name: str) -> Optional[str]:
        file_info = self._get_file_info(file_path)
        if not type_name:
            return None

        candidates = [type_name]
        clean = type_name.replace("*", "").replace("&", "").replace("const", "").strip()
        if clean != type_name:
            candidates.append(clean)

        if file_info:
            for cand in candidates:
                if cand in file_info and file_info[cand].get("type") in ("Struct", "Union", "Enum"):
                    return cand

        for _fpath, finfo in self.parse_res.items():
            for cand in candidates:
                if cand in finfo and finfo[cand].get("type") in ("Struct", "Union", "Enum"):
                    return cand

        return None

    def _collect_struct_members(self, struct_name: str) -> List[str]:
        if not struct_name:
            return []

        members = []
        for _fpath, finfo in self.parse_res.items():
            for name, info in finfo.items():
                if info.get("in_struct") == struct_name:
                    members.append(name)
        return sorted(members)

    def _filter_prefix(self, names: List[str], prefix: str) -> List[str]:
        if not prefix:
            return names
        return [x for x in names if x.split(".")[-1].startswith(prefix)]

    def _safe_get_prompt4names(
        self,
        file_path: str,
        names: set,
        only_def: bool = False,
        enable_docstring: bool = True,
    ) -> str:
        if not self.searcher or not file_path:
            return ""
        try:
            return self.searcher.get_prompt4names(
                file_path,
                names,
                only_def=only_def,
                enable_docstring=enable_docstring,
            )
        except Exception as exc:
            return f"/* 获取上下文失败: {exc} */"

    def _find_owner_file(self, symbol_name: str) -> Optional[str]:
        for fpath, finfo in self.parse_res.items():
            if symbol_name in finfo:
                return fpath
        return None

    def _build_aider_edit_requirements(self, extra_rules: Optional[List[str]] = None) -> str:
        rules = [
            "1. 不要只输出候选列表，也不要停留在解释层面；你的任务是直接修改目标文件中的代码。",
            "2. 优先做最小必要修改，保持当前项目代码风格、命名习惯和错误处理方式一致。",
            "3. 优先复用项目中真实存在的符号、类型、函数和成员，不要凭空捏造 API。",
            "4. 若目标位置存在占位符、未完成表达式、TODO、注释掉的旧实现或明显错误代码，请直接补全或修正它。",
            "5. 如需补充必要的声明、类型、头文件或辅助代码，可以一并修改，但避免无关重构。",
            "6. 通过直接编辑文件完成任务，不要返回“候选列表式答案”。",
        ]
        if extra_rules:
            rules.extend(extra_rules)
        return "【执行要求】\n" + "\n".join(rules)

    # ---------------------------
    # 1. 成员补全
    # ---------------------------
    def _build_member_completion_prompt(
        self,
        user_instruction: str,
        file_path: str,
        symbol: Optional[str],
        prefix: str = "",
    ) -> str:
        file_info = self._get_file_info(file_path)
        if not file_info:
            return f"无法找到文件 {file_path} 的解析信息。"

        type_name = self._find_type_of_symbol(file_path, symbol) if symbol else None
        struct_name = self._find_struct_like_by_type(file_path, type_name) if type_name else None
        member_names = self._collect_struct_members(struct_name) if struct_name else []
        member_names = self._filter_prefix(member_names, prefix)

        context_parts: List[str] = []

        if symbol and symbol in file_info:
            context_parts.append("【当前变量定义】")
            context_parts.append(
                self._safe_get_prompt4names(
                    file_path, {symbol}, only_def=False, enable_docstring=True
                )
            )

        if struct_name:
            context_parts.append("【相关类型定义】")
            owner_file = self._find_owner_file(struct_name)
            if owner_file:
                context_parts.append(
                    self._safe_get_prompt4names(
                        owner_file,
                        {struct_name} | set(member_names),
                        only_def=False,
                        enable_docstring=True,
                    )
                )

        context_parts.append("【候选成员】")
        if member_names:
            context_parts.extend([f"- {m.split('.')[-1]}" for m in member_names])
        else:
            context_parts.append("- 无法找到明确成员，请基于上下文谨慎推断")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据给定上下文，在目标文件中直接完成“成员访问”相关代码的补全或修正。

【用户指令】
{user_instruction}

【补全目标】
变量名：{symbol}
文件：{file_path}
推断类型：{type_name}
对应结构体/联合体：{struct_name}

{chr(10).join(context_parts)}

{self._build_aider_edit_requirements([
    '7. 如果当前代码中存在不完整的 `obj->...`、`obj....`、占位字段名或错误字段访问，请将其改成最合适的真实成员访问。',
    f'8. 优先使用项目中真实存在的字段；若给定前缀 "{prefix}"，优先选择此前缀匹配的成员。',
    '9. 若成员访问需要连带补充判空、类型转换或相关上下文代码，可做最小必要修改以保证代码可工作。',
])}
"""
        return prompt

    # ---------------------------
    # 2. 变量补全
    # ---------------------------
    def _build_variable_completion_prompt(
        self,
        user_instruction: str,
        file_path: str,
        prefix: str = "",
    ) -> str:
        file_info = self._get_file_info(file_path)
        if not file_info:
            return f"无法找到文件 {file_path} 的解析信息。"

        local_vars = []
        global_vars = []

        for name, info in self._iter_symbols_in_file(file_info):
            if info.get("type") != "Variable":
                continue
            if info.get("in_struct"):
                continue
            if info.get("in_function"):
                local_vars.append((name, info))
            else:
                global_vars.append((name, info))

        if prefix:
            local_vars = [(n, i) for n, i in local_vars if n.startswith(prefix)]
            global_vars = [(n, i) for n, i in global_vars if n.startswith(prefix)]

        local_vars.sort(key=lambda x: x[1].get("sline", 10**9))
        global_vars.sort(key=lambda x: x[1].get("sline", 10**9))

        context = self._safe_get_prompt4names(
            file_path, {""}, only_def=True, enable_docstring=False
        )

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据给定文件中的可见变量，在目标文件中直接完成变量引用、变量名补全或相关小范围修正。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【当前文件摘要】
{context}

【局部变量候选】
{chr(10).join(f"- {n}" for n, _ in local_vars) if local_vars else "- 无"}

【全局变量候选】
{chr(10).join(f"- {n}" for n, _ in global_vars) if global_vars else "- 无"}

{self._build_aider_edit_requirements([
    '7. 如果当前代码存在未完成变量名、错误变量引用或缺失的局部上下文，请直接改成合适的变量使用方式。',
    '8. 优先使用当前作用域内的局部变量，其次再考虑全局变量。',
    f'9. 若指定前缀 "{prefix}"，优先选择此前缀匹配的变量；若无精确匹配，再结合上下文做最合理的小范围修正。',
])}
"""
        return prompt

    # ---------------------------
    # 3. 函数签名补全
    # ---------------------------
    def _build_function_completion_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
        prefix: str = "",
    ) -> str:
        matched = []

        for fpath, finfo in self.parse_res.items():
            for name, info in finfo.items():
                if info.get("type") != "Function":
                    continue
                if symbol and name == symbol:
                    matched.append((fpath, name, info))
                elif prefix and name.startswith(prefix):
                    matched.append((fpath, name, info))
                elif not symbol and not prefix:
                    matched.append((fpath, name, info))

        matched.sort(key=lambda x: (x[0] != file_path, x[2].get("sline", 10**9), x[1]))
        matched = matched[:30]

        func_blocks = []
        for fpath, _name, info in matched:
            func_blocks.append(f"/* {fpath} */\n{info.get('def', '')}")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据项目中已有函数定义/声明，在目标文件中直接完成函数调用、函数声明或函数签名相关代码的补全与修正。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【目标函数名/前缀】
symbol={symbol}
prefix={prefix}

【候选函数签名】
{chr(10).join(func_blocks) if func_blocks else "无匹配函数"}

{self._build_aider_edit_requirements([
    '7. 如果当前代码中存在未完成函数调用、错误函数名、缺失声明或不完整签名，请直接修改为可工作的最终代码。',
    '8. 优先复用目标文件内已有函数风格和项目内现有函数签名；若 symbol 精确给定，优先围绕该函数完成修改。',
    '9. 如需新增或补齐函数声明，请保持与项目中的声明/定义风格一致。',
])}
"""
        return prompt

    # ---------------------------
    # 4. 函数体补全
    # ---------------------------
    def _build_function_body_completion_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
    ) -> str:
        file_info = self._get_file_info(file_path)
        if not file_info:
            return f"无法找到文件 {file_path} 的解析信息。"

        context_parts: List[str] = []

        if symbol and symbol in file_info:
            context_parts.append("【目标函数当前定义】")
            context_parts.append(
                self._safe_get_prompt4names(
                    file_path, {symbol}, only_def=False, enable_docstring=True
                )
            )

        same_file_funcs = []
        for name, info in self._iter_symbols_in_file(file_info):
            if info.get("type") == "Function" and name != symbol:
                same_file_funcs.append((name, info))

        same_file_funcs = same_file_funcs[:10]
        if same_file_funcs:
            context_parts.append("【同文件其他函数参考】")
            ref_names = {name for name, _ in same_file_funcs}
            context_parts.append(
                self._safe_get_prompt4names(
                    file_path, ref_names, only_def=False, enable_docstring=True
                )
            )

        related_blocks = []
        for fpath, finfo in self.parse_res.items():
            if symbol and symbol in finfo and fpath != file_path:
                info = finfo[symbol]
                if info.get("type") == "Function":
                    related_blocks.append(f"/* {fpath} */\n{info.get('def', '')}")

        if related_blocks:
            context_parts.append("【项目内相关声明/定义】")
            context_parts.append("\n".join(related_blocks[:10]))

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据目标函数当前上下文、同文件风格和项目内相关定义，补全目标函数的完整实现内容。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【目标函数】
{symbol}

{chr(10).join(context_parts)}

{self._build_aider_edit_requirements([
    '7. 目标是让目标函数在文件中成为完整、可工作的实现，而不是只给一个片段候选。',
    '8. 若函数中已有部分注释或被注释掉的旧实现，请优先恢复和完善该实现。',
    '9. 如果实现函数需要同步补齐相关调用、局部变量或错误处理逻辑，可做最小必要修改。',
])}
"""
        return prompt

    # ---------------------------
    # 5. 类型相关补全
    # ---------------------------
    def _build_type_completion_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
        prefix: str = "",
    ) -> str:
        matched = []

        for fpath, finfo in self.parse_res.items():
            for name, info in finfo.items():
                if info.get("type") not in ("Struct", "Union", "Enum", "Variable"):
                    continue

                if symbol and name == symbol:
                    matched.append((fpath, name, info))
                elif prefix and name.startswith(prefix):
                    matched.append((fpath, name, info))
                elif not symbol and not prefix:
                    matched.append((fpath, name, info))

        matched.sort(key=lambda x: (x[0] != file_path, x[2].get("sline", 10**9), x[1]))
        matched = matched[:30]

        blocks = []
        for fpath, name, info in matched:
            if info.get("type") in ("Struct", "Union", "Enum"):
                blocks.append(
                    self._safe_get_prompt4names(
                        fpath, {name}, only_def=False, enable_docstring=True
                    )
                )
            else:
                blocks.append(f"/* {fpath} */\n{info.get('def', '')}")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据项目中的类型定义，在目标文件中直接完成类型相关代码的补全、修正或最小必要改写。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【目标类型名/前缀】
symbol={symbol}
prefix={prefix}

【候选类型上下文】
{chr(10).join(blocks) if blocks else "无匹配类型"}

{self._build_aider_edit_requirements([
    '7. 如果当前代码中存在缺失类型名、错误 typedef/struct/enum 引用或不完整类型声明，请直接修正目标文件中的代码。',
    '8. 若上下文明显属于字段/成员访问场景，可结合该类型的关键字段和真实定义完成必要修改。',
    f'9. 若给定 symbol 或前缀（prefix={prefix}），优先使用与其最匹配的真实类型定义。',
])}
"""
        return prompt

    # ---------------------------
    # fallback
    # ---------------------------
    def _build_fallback_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
        prefix: str = "",
    ) -> str:
        context = ""
        if file_path and file_path in self.parse_res:
            context = self._safe_get_prompt4names(
                file_path, {""}, only_def=True, enable_docstring=False
            )

        return f"""你是一个 C/C++ 代码补全助手。
请基于以下项目上下文，直接修改目标文件中的相关代码并完成任务。

【用户指令】
{user_instruction}

【文件】
{file_path}

【符号】
{symbol}

【前缀】
{prefix}

【文件上下文】
{context}

{self._build_aider_edit_requirements()}
"""


if __name__ == "__main__":
    agent = CompletionPromptAgent()

    req = {
        "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/examples/json-c",
        "user_instruction": "补全 parse_flags 函数的完整实现，若已有被注释掉的旧实现，请优先恢复并完善。",
        "file_path": "tests/parse_flags.c",
        "symbol": "parse_flags",
        "completion_type": "function_body",
    }

    try:
        print(agent.build_prompt(req))
    except Exception as e:
        print(f"运行失败: {e}")
