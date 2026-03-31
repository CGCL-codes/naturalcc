import os
import re
import json
from typing import Dict, List, Optional, Tuple

from .preprocess import CProjectParser


class CompletionPromptAgent:
    """
    不修改原有工具，只在其上层封装一个：
    1. 接收用户指令
    2. 识别补全类型
    3. 从 parse_res 中抽取相关信息
    4. 输出不同风格的 prompt
    """

    def __init__(self):
        self.project_parser = CProjectParser()
        self.proj_dir = None
        self.parse_res = None
        self.searcher = None

    # ---------------------------
    # 项目加载
    # ---------------------------
    def load_project(self, proj_dir: str):
        self.proj_dir = proj_dir
        self.parse_res = self.project_parser.parse_dir(proj_dir)
        self.searcher = self.project_parser.proj_searcher

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
        if proj_dir and (self.proj_dir != proj_dir or self.parse_res is None):
            self.load_project(proj_dir)

        if self.parse_res is None:
            raise ValueError("项目尚未加载，请提供 project_dir")

        user_instruction = request.get("user_instruction", "").strip()
        file_path = request.get("file_path")
        symbol = request.get("symbol")
        prefix = request.get("prefix", "")
        completion_type = request.get("completion_type")

        if not completion_type:
            completion_type = self._infer_completion_type(user_instruction, symbol)

        if completion_type == "member":
            return self._build_member_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix
            )
        elif completion_type == "variable":
            return self._build_variable_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                prefix=prefix
            )
        elif completion_type == "function":
            return self._build_function_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix
            )
        elif completion_type == "type":
            return self._build_type_completion_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix
            )
        else:
            return self._build_fallback_prompt(
                user_instruction=user_instruction,
                file_path=file_path,
                symbol=symbol,
                prefix=prefix
            )

    # ---------------------------
    # 识别补全类型
    # ---------------------------
    def _infer_completion_type(self, instruction: str, symbol: Optional[str]) -> str:
        text = instruction.lower()

        if any(x in instruction for x in ["成员", "字段", "struct字段", "成员补全"]) \
           or "->" in instruction or "." in instruction:
            return "member"

        if any(x in instruction for x in ["局部变量", "全局变量", "变量补全", "变量名"]):
            return "variable"

        if any(x in instruction for x in ["函数签名", "函数声明", "函数补全", "函数名"]):
            return "function"

        if any(x in instruction for x in ["类型", "typedef", "struct", "union", "enum"]):
            return "type"

        # 如果显式给了 symbol，但没说清楚，先猜成员/类型
        if symbol:
            return "member"

        return "function"

    # ---------------------------
    # 工具函数
    # ---------------------------
    def _get_file_info(self, file_path: str) -> Optional[Dict]:
        if not file_path:
            return None
        return self.parse_res.get(file_path)

    def _iter_symbols_in_file(self, file_info: Dict):
        for name, info in file_info.items():
            if not name:
                continue
            yield name, info

    def _find_symbol_info(self, file_path: str, symbol: str) -> Optional[Tuple[str, Dict]]:
        file_info = self._get_file_info(file_path)
        if not file_info or not symbol:
            return None

        # 先精确匹配
        if symbol in file_info:
            return symbol, file_info[symbol]

        # 再尝试匹配结构体成员尾部
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
            if len(item) >= 2:
                # 兼容旧格式 [type_name, "Typeof"] 或清洗后 [type_name, extra, "Typeof"]
                if item[-1] == "Typeof":
                    return item[0]
        return None

    def _find_struct_like_by_type(self, file_path: str, type_name: str) -> Optional[str]:
        file_info = self._get_file_info(file_path)
        if not file_info or not type_name:
            return None

        candidates = [type_name]
        # 简单去指针、const
        clean = type_name.replace("*", "").replace("&", "").replace("const", "").strip()
        if clean != type_name:
            candidates.append(clean)

        for cand in candidates:
            if cand in file_info and file_info[cand]["type"] in ("Struct", "Union", "Enum"):
                return cand

        # 项目内跨文件找
        for fpath, finfo in self.parse_res.items():
            if clean in finfo and finfo[clean]["type"] in ("Struct", "Union", "Enum"):
                return clean

        return None

    def _collect_struct_members(self, struct_name: str) -> List[str]:
        members = []
        for fpath, finfo in self.parse_res.items():
            for name, info in finfo.items():
                if info.get("in_struct") == struct_name:
                    members.append(name)
        return sorted(members)

    def _filter_prefix(self, names: List[str], prefix: str) -> List[str]:
        if not prefix:
            return names
        return [x for x in names if x.split(".")[-1].startswith(prefix)]

    # ---------------------------
    # 1. 成员补全
    # ---------------------------
    def _build_member_completion_prompt(
        self,
        user_instruction: str,
        file_path: str,
        symbol: Optional[str],
        prefix: str = ""
    ) -> str:
        file_info = self._get_file_info(file_path)
        if not file_info:
            return f"无法找到文件 {file_path} 的解析信息。"

        type_name = self._find_type_of_symbol(file_path, symbol) if symbol else None
        struct_name = self._find_struct_like_by_type(file_path, type_name) if type_name else None
        member_names = self._collect_struct_members(struct_name) if struct_name else []
        member_names = self._filter_prefix(member_names, prefix)

        context_parts = []

        if symbol and symbol in file_info:
            context_parts.append("【当前变量定义】")
            context_parts.append(self.searcher.get_prompt4names(file_path, {symbol}, only_def=False, enable_docstring=True))

        if struct_name:
            context_parts.append("【相关类型定义】")
            # 用旧工具拼 struct 上下文
            owner_file = self._find_owner_file(struct_name)
            if owner_file:
                context_parts.append(
                    self.searcher.get_prompt4names(owner_file, {struct_name} | set(member_names), only_def=False, enable_docstring=True)
                )

        context_parts.append("【候选成员】")
        if member_names:
            context_parts.extend([f"- {m.split('.')[-1]}" for m in member_names])
        else:
            context_parts.append("- 无法找到明确成员，请基于上下文谨慎推断")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据给定上下文，为“成员访问”场景生成补全候选。

【用户指令】
{user_instruction}

【补全目标】
变量名：{symbol}
文件：{file_path}
推断类型：{type_name}
对应结构体/联合体：{struct_name}

{chr(10).join(context_parts)}

【输出要求】
1. 只输出成员候选，不输出解释。
2. 每行一个候选。
3. 优先输出项目中真实存在的字段。
4. 若有前缀 "{prefix}"，优先只输出此前缀匹配项。
"""
        return prompt

    # ---------------------------
    # 2. 变量补全
    # ---------------------------
    def _build_variable_completion_prompt(
        self,
        user_instruction: str,
        file_path: str,
        prefix: str = ""
    ) -> str:
        file_info = self._get_file_info(file_path)
        if not file_info:
            return f"无法找到文件 {file_path} 的解析信息。"

        local_vars = []
        global_vars = []

        for name, info in self._iter_symbols_in_file(file_info):
            if info["type"] != "Variable":
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

        context = self.searcher.get_prompt4names(file_path, {""}, only_def=True, enable_docstring=False)

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据给定文件中的可见变量，生成变量补全候选。

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

【输出要求】
1. 只输出变量名，不输出解释。
2. 每行一个候选。
3. 优先局部变量，再输出全局变量。
4. 若指定前缀 "{prefix}"，优先只输出此前缀匹配项。
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
        prefix: str = ""
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
        for fpath, name, info in matched:
            func_blocks.append(f"/* {fpath} */\n{info['def']}")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据项目中已有函数定义/声明，生成函数签名补全候选。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【目标函数名/前缀】
symbol={symbol}
prefix={prefix}

【候选函数签名】
{chr(10).join(func_blocks) if func_blocks else "无匹配函数"}

【输出要求】
1. 只输出函数签名候选，不输出解释。
2. 每行一个候选。
3. 优先输出目标文件内的函数，再输出项目内其他文件的函数。
4. 若 symbol 精确给定，优先输出该函数签名。
"""
        return prompt

    # ---------------------------
    # 4. 类型相关补全
    # ---------------------------
    def _build_type_completion_prompt(
        self,
        user_instruction: str,
        file_path: Optional[str],
        symbol: Optional[str],
        prefix: str = ""
    ) -> str:
        matched = []

        for fpath, finfo in self.parse_res.items():
            for name, info in finfo.items():
                if info.get("type") not in ("Struct", "Union", "Enum", "Variable"):
                    continue

                # typedef 在旧系统里也可能被记成 Variable
                if symbol and name == symbol:
                    matched.append((fpath, name, info))
                elif prefix and name.startswith(prefix):
                    matched.append((fpath, name, info))

        matched.sort(key=lambda x: (x[0] != file_path, x[2].get("sline", 10**9), x[1]))
        matched = matched[:30]

        blocks = []
        for fpath, name, info in matched:
            if info["type"] in ("Struct", "Union", "Enum"):
                blocks.append(self.searcher.get_prompt4names(fpath, {name}, only_def=False, enable_docstring=True))
            else:
                blocks.append(f"/* {fpath} */\n{info['def']}")

        prompt = f"""你是一个 C/C++ 代码补全助手。
任务：根据项目中的类型定义，为类型相关场景生成补全候选。

【用户指令】
{user_instruction}

【目标文件】
{file_path}

【目标类型名/前缀】
symbol={symbol}
prefix={prefix}

【候选类型上下文】
{chr(10).join(blocks) if blocks else "无匹配类型"}

【输出要求】
1. 若是类型名补全，只输出类型名候选。
2. 若上下文明显在字段/成员场景，可输出该类型的关键字段名。
3. 不输出多余解释。
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
        prefix: str = ""
    ) -> str:
        context = ""
        if file_path and file_path in self.parse_res:
            context = self.searcher.get_prompt4names(file_path, {""}, only_def=True, enable_docstring=False)

        return f"""你是一个 C/C++ 代码补全助手。
请基于以下项目上下文完成补全。

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

【输出要求】
只输出补全候选，不输出解释。
"""

    # ---------------------------
    # 辅助：找到符号归属文件
    # ---------------------------
    def _find_owner_file(self, symbol_name: str) -> Optional[str]:
        for fpath, finfo in self.parse_res.items():
            if symbol_name in finfo:
                return fpath
        return None


if __name__ == "__main__":
    # 示例
    agent = CompletionPromptAgent()

#     req1 = {
#     "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/awtk",
#     "user_instruction": "补全 factory 的成员",
#     "file_path": "src/base/children_layouter_factory.c",
#     "symbol": "factory",
#     "completion_type": "member"
# }
#     req1 = {
#         "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/awtk",
#         "user_instruction": "补全当前函数里的变量",
#         "file_path": "src/base/children_layouter_factory.c",
#         "completion_type": "variable"
#     }
#     req1 = {
#         "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/awtk",
#         "user_instruction": "补全 children_layouter_factory_ 相关函数",
#         "file_path": "src/base/children_layouter_factory.c",
#         "completion_type": "function",
#         "prefix": "children_layouter_factory_"
#     }
    req1 = {
        "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/curl",
        "user_instruction": "补全 get_terminal_ 开头的函数",
        "file_path": "src/terminal.c",
        "completion_type": "function",
        "prefix": "get_terminal_"
    }
    # req1 = {
    # "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/curl",
    # "user_instruction": "补全当前函数里的变量",
    # "file_path": "src/terminal.c",
    # "completion_type": "variable",
    # "prefix": "co"
    # }
    req1 = {
    "project_dir": "/home/sub4-wy/wangchen/NLPLAB/car/ccoder/CEval/call_repo/curl",
    "user_instruction": "补全 winsize 类型",
    "file_path": "src/terminal.c",
    "completion_type": "type",
    "prefix": "win"
    }
    # req2 = {
    #     "project_dir": "/path/to/your/c_project",
    #     "user_instruction": "补全 parse_ 开头的函数签名",
    #     "file_path": "src/parser.c",
    #     "prefix": "parse_",
    #     "completion_type": "function"
    # }

    print(agent.build_prompt(req1))
    # print(agent.build_prompt(req2))