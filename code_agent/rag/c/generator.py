import os
import json
import re

from .tokenizer import CModelTokenizer
from .utils import MAX_HOP, ONLY_DEF, ENABLE_DOCSTRING, LAST_K_LINES


class CGenerator(object):
    def __init__(self, proj_dir, info_dir, model):
        self.proj_dir = os.path.abspath(proj_dir)
        self.info_dir = os.path.abspath(info_dir)
        self.tokenizer = CModelTokenizer(model)
        
        self.project = None
        self.proj_info = None
    
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
    
    def _extract_include_headers(self, source_code):
        user_includes = re.findall(r'#include\s+"([^"]+)"', source_code)
        return user_includes
    
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
    
    def _format_function_defs(self, header_path, functions_info):
        if not functions_info:
            return ""
        
        sorted_functions = sorted(functions_info.items(), key=lambda x: x[1].get('sline', 0))
        
        result = f"// {header_path}\n"
        for func_name, func_info in sorted_functions:
            if 'def' in func_info:
                result += f"{func_info['def']}\n"
        
        return result
    
    def get_suffix(self, fpath):
        return f"// path: {fpath}\n"
    
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