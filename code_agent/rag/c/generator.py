import os
import json
import re

from .tokenizer import CModelTokenizer
from .utils import MAX_HOP, ONLY_DEF, ENABLE_DOCSTRING, LAST_K_LINES


# [maint-note:b5fd16c3a200]
# 数据与行为导航：CGenerator。
# 继承接口：object。
# [/maint-note:b5fd16c3a200]
class CGenerator(object):
    # [maint-note:f3da696d5c27]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 输入 proj_dir：类型由调用方与函数体约束。
    # 输入 info_dir：类型由调用方与函数体约束。
    # 输入 model：类型由调用方与函数体约束。
    # 写入目标：self.proj_dir。
    # 写入目标：self.info_dir。
    # 写入目标：self.tokenizer。
    # 写入目标：self.project。
    # 写入目标：self.proj_info。
    # 直接调用：os.path.abspath。
    # 直接调用：CModelTokenizer。
    # [/maint-note:f3da696d5c27]
    def __init__(self, proj_dir, info_dir, model):
        self.proj_dir = os.path.abspath(proj_dir)
        self.info_dir = os.path.abspath(info_dir)
        self.tokenizer = CModelTokenizer(model)
        
        self.project = None
        self.proj_info = None
    
    # [maint-note:09d80c3b9963]
    # 维护导航：_set_project；下列为本函数源码中的接口与路径索引。
    # 输入 project：类型由调用方与函数体约束。
    # 分支条件：project == self.project。
    # 分支条件：not os.path.isfile(info_file)。
    # 返回路径：无返回值。
    # 写入目标：self.project。
    # 写入目标：self.proj_info。
    # 直接调用：os.path.join。
    # 直接调用：os.path.isfile。
    # 直接调用：print。
    # 直接调用：open。
    # 直接调用：json.load。
    # 资源作用域：open(info_file, 'r')；退出该 with 时调用对应上下文清理。
    # [/maint-note:09d80c3b9963]
    def _set_project(self, project):
        if project == self.project:
            return

        info_file = os.path.join(self.info_dir, f'{project}.json')
        if not os.path.isfile(info_file):
            print(f'未知项目 {project} 在 {self.info_dir}')
            return
        
        self.project = project
        with open(info_file, 'r') as f:
            self.proj_info = json.load(f)
    
    # [maint-note:41a06a1829dd]
    # 维护导航：_extract_include_headers；下列为本函数源码中的接口与路径索引。
    # 输入 source_code：类型由调用方与函数体约束。
    # 返回路径：user_includes。
    # 直接调用：re.findall。
    # [/maint-note:41a06a1829dd]
    def _extract_include_headers(self, source_code):
        user_includes = re.findall(r'#include\s+"([^"]+)"', source_code)
        return user_includes
    
    # [maint-note:0bb492175133]
    # 维护导航：_find_header_info；下列为本函数源码中的接口与路径索引。
    # 输入 header_name：类型由调用方与函数体约束。
    # 分支条件：path.endswith(header_name)。
    # 分支条件：not header_paths。
    # 分支条件：entity_info.get('type') == 'Variable' and entity_info.get('def', '').find(header_name) >= 0。
    # 分支条件：path not in header_paths。
    # 分支条件：path in self.proj_info。
    # 分支条件：entity_info.get('type') == 'Function'。
    # 遍历边界：path ← self.proj_info。
    # 遍历边界：(path, info) ← self.proj_info.items()。
    # 遍历边界：(entity_name, entity_info) ← info.items()。
    # 遍历边界：path ← header_paths。
    # 遍历边界：(entity_name, entity_info) ← self.proj_info[path].items()。
    # 返回路径：(header_paths, functions_info)。
    # 写入目标：functions_info[entity_name]。
    # 直接调用：path.endswith。
    # 直接调用：header_paths.append。
    # 直接调用：self.proj_info.items。
    # 直接调用：info.items。
    # 直接调用：entity_info.get。
    # 直接调用：entity_info.get('def', '').find。
    # 直接调用：self.proj_info[path].items。
    # [/maint-note:0bb492175133]
    def _find_header_info(self, header_name):
        header_paths = []
        for path in self.proj_info:
            if path.endswith(header_name):
                header_paths.append(path)
        
        if not header_paths:
            for path, info in self.proj_info.items():
                for entity_name, entity_info in info.items():
                    if entity_info.get('type') == 'Variable' and entity_info.get('def', '').find(header_name) >= 0:
                        if path not in header_paths:
                            header_paths.append(path)
        
        functions_info = {}
        for path in header_paths:
            if path in self.proj_info:
                for entity_name, entity_info in self.proj_info[path].items():
                    if entity_info.get('type') == 'Function':
                        functions_info[entity_name] = entity_info
        
        return header_paths, functions_info
    
    # [maint-note:15e364cd5ad2]
    # 维护导航：_format_function_defs；下列为本函数源码中的接口与路径索引。
    # 输入 header_path：类型由调用方与函数体约束。
    # 输入 functions_info：类型由调用方与函数体约束。
    # 分支条件：not functions_info。
    # 分支条件：'def' in func_info。
    # 遍历边界：(func_name, func_info) ← sorted_functions。
    # 返回路径：''。
    # 返回路径：result。
    # 直接调用：sorted。
    # 直接调用：functions_info.items。
    # [/maint-note:15e364cd5ad2]
    def _format_function_defs(self, header_path, functions_info):
        if not functions_info:
            return ""
        
        sorted_functions = sorted(functions_info.items(), key=lambda x: x[1].get('sline', 0))
        
        result = f"// {header_path}\n"
        for func_name, func_info in sorted_functions:
            if 'def' in func_info:
                result += f"{func_info['def']}\n"
        
        return result
    
    # [maint-note:a784f21b9209]
    # 维护导航：get_suffix；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 返回路径：f'// path: {fpath}\n'。
    # [/maint-note:a784f21b9209]
    def get_suffix(self, fpath):
        return f"// path: {fpath}\n"
    
    # [maint-note:56977d6bc871]
    # 维护导航：retrieve_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 project：类型由调用方与函数体约束。
    # 输入 fpath：类型由调用方与函数体约束。
    # 输入 source_code：类型由调用方与函数体约束。
    # 分支条件：not prompt.strip()。
    # 分支条件：self.tokenizer.cal_token_nums(source_code) > half_length。
    # 分支条件：not self.tokenizer.judge_prompt(prompt, max_prompt_length)。
    # 分支条件：current_length + line_length <= max_prompt_length。
    # 遍历边界：header ← user_headers。
    # 遍历边界：header_path ← header_paths。
    # 遍历边界：line ← lines。
    # 返回路径：self.tokenizer.truncate_concat(source_code, '', suffix)。
    # 返回路径：self.tokenizer.truncate_concat(source_code, prompt, suffix)。
    # 直接调用：self._set_project。
    # 直接调用：self._extract_include_headers。
    # 直接调用：self._find_header_info。
    # 直接调用：self._format_function_defs。
    # 直接调用：prompt.strip。
    # 直接调用：self.get_suffix。
    # 直接调用：self.tokenizer.truncate_concat。
    # 直接调用：self.tokenizer.cal_prompt_max_length。
    # 直接调用：int。
    # 直接调用：self.tokenizer.cal_token_nums。
    # 直接调用：self.tokenizer.judge_prompt。
    # 直接调用：prompt.split。
    # 直接调用：len。
    # 直接调用：self.tokenizer.tokenizer.encode(line, return_tensors='pt').flatten。
    # 直接调用：self.tokenizer.tokenizer.encode。
    # 直接调用：truncated_lines.append。
    # 直接调用：'\n'.join。
    # [/maint-note:56977d6bc871]
    def retrieve_prompt(self, project, fpath, source_code):
        self._set_project(project)
        
        user_headers = self._extract_include_headers(source_code)
        
        prompt = ""
        for header in user_headers:
            header_paths, functions_info = self._find_header_info(header)
            
            for header_path in header_paths:
                prompt += self._format_function_defs(header_path, functions_info)
                prompt += "\n"
        
        if not prompt.strip():
            suffix = self.get_suffix(fpath)
            return self.tokenizer.truncate_concat(source_code, "", suffix)
        
        suffix = self.get_suffix(fpath)

        max_prompt_length = self.tokenizer.cal_prompt_max_length(source_code, suffix)
        
        half_length = int(0.5 * self.tokenizer.max_input_length)
        if self.tokenizer.cal_token_nums(source_code) > half_length:
            source_code = source_code[-half_length:]
        
        if not self.tokenizer.judge_prompt(prompt, max_prompt_length):
            lines = prompt.split('\n')
            truncated_lines = []
            current_length = 0
            
            for line in lines:
                line_length = len(self.tokenizer.tokenizer.encode(line, return_tensors="pt").flatten())
                if current_length + line_length <= max_prompt_length:
                    truncated_lines.append(line)
                    current_length += line_length
                else:
                    break
            
            prompt = '\n'.join(truncated_lines)
        
        return self.tokenizer.truncate_concat(source_code, prompt, suffix)