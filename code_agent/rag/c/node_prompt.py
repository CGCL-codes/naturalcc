import os
import json
from itertools import groupby
# 根据一个已经解析好的 C 项目信息，按名字查找相关定义，并把它们整理成适合展示或拼接成上下文提示的代码片段。

# [maint-note:3419af7f8ce4]
# 数据与行为导航：CProjectSearcher。
# 继承接口：object。
# [/maint-note:3419af7f8ce4]
class CProjectSearcher(object):
    # [maint-note:16eb74af58d6]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 写入目标：self.proj_dir。
    # 写入目标：self.proj_info。
    # 写入目标：self.standard_libraries。
    # [/maint-note:16eb74af58d6]
    def __init__(self):
        self.proj_dir = None
        self.proj_info = None

        self.standard_libraries = {
            "stdio", "stdlib", "string", "math", "time", "ctype", "assert", 
            "limits", "float", "stddef", "stdarg", "stdbool", "stdint", "errno",
            "signal", "locale", "setjmp", "wchar", "wctype"
        }
    
    # [maint-note:e753d97b74df]
    # 维护导航：set_proj；下列为本函数源码中的接口与路径索引。
    # 输入 proj_dir：类型由调用方与函数体约束。
    # 输入 proj_info：类型由调用方与函数体约束。
    # 分支条件：'' in file_info and normalized_module not in file_info。
    # 遍历边界：(module, file_info) ← proj_info.items()。
    # 写入目标：self.proj_dir。
    # 写入目标：file_info[normalized_module]。
    # 写入目标：normalized_info[normalized_module]。
    # 写入目标：self.proj_info。
    # 直接调用：os.path.abspath。
    # 直接调用：proj_info.items。
    # 直接调用：self._normalize_path。
    # 直接调用：file_info.pop。
    # [/maint-note:e753d97b74df]
    def set_proj(self, proj_dir, proj_info):
        self.proj_dir = os.path.abspath(proj_dir)
            
        normalized_info = {}
        for module, file_info in proj_info.items():
            normalized_module = self._normalize_path(module)
            if "" in file_info and normalized_module not in file_info:
                module_info = file_info.pop("")
                file_info[normalized_module] = module_info
                
            normalized_info[normalized_module] = file_info
        
        self.proj_info = normalized_info

    # [maint-note:a0c9ecaaaa84]
    # 维护导航：_normalize_path；下列为本函数源码中的接口与路径索引。
    # 输入 path：类型由调用方与函数体约束。
    # 返回路径：str(path).replace('\\', '/')。
    # 直接调用：str(path).replace。
    # 直接调用：str。
    # [/maint-note:a0c9ecaaaa84]
    def _normalize_path(self, path):
        return str(path).replace("\\", "/")
    
    # [maint-note:aab2cd204e4a]
    # 维护导航：name_in_file；下列为本函数源码中的接口与路径索引。
    # 输入 name：类型由调用方与函数体约束。
    # 输入 avail_list：类型由调用方与函数体约束。
    # 输入 src_name：类型由调用方与函数体约束；缺省表达式 None。
    # 输入 struct_name：类型由调用方与函数体约束；缺省表达式 None。
    # 分支条件：name.count('.') > 0 and struct_name。
    # 分支条件：parts[0] == struct_name。
    # 分支条件：src_name is None or item != src_name。
    # 分支条件：name == item。
    # 分支条件：name.startswith(f'{item}.')。
    # 遍历边界：item ← sorted(avail_list, key=lambda x: len(x.split('.')), reverse=True)。
    # 返回路径：(struct_name, parts[1])。
    # 返回路径：(item, None)。
    # 返回路径：(item, name[len(item) + 1:])。
    # 返回路径：None。
    # 直接调用：name.count。
    # 直接调用：name.split。
    # 直接调用：sorted。
    # 直接调用：name.startswith。
    # 直接调用：len。
    # [/maint-note:aab2cd204e4a]
    def name_in_file(self, name, avail_list, src_name=None, struct_name=None):
        if name.count('.') > 0 and struct_name:
            parts = name.split('.')
            if parts[0] == struct_name:
                return struct_name, parts[1]

        for item in sorted(avail_list, key=lambda x:len(x.split('.')), reverse=True):
            if src_name is None or item != src_name:
                if name == item:
                    return item, None
                
                elif name.startswith(f'{item}.'):
                    return item, name[len(item)+1:]
        
        return None
    
    # [maint-note:4515ab11851f]
    # 维护导航：is_local_include；下列为本函数源码中的接口与路径索引。
    # 输入 file_path：类型由调用方与函数体约束。
    # 输入 include_info：类型由调用方与函数体约束。
    # 分支条件：isinstance(include_info, list) and len(include_info) > 0。
    # 分支条件：header_base in self.standard_libraries。
    # 分支条件：path.endswith(normalized_header_name) or path.endswith(normalized_include_suffix)。
    # 分支条件：not candidates。
    # 分支条件：candidates。
    # 分支条件：len(candidates) > 1。
    # 遍历边界：path ← self.proj_info。
    # 返回路径：None。
    # 返回路径：[candidates[0], None]。
    # 直接调用：isinstance。
    # 直接调用：len。
    # 直接调用：os.path.splitext。
    # 直接调用：os.path.basename。
    # 直接调用：self._normalize_path。
    # 直接调用：os.path.join。
    # 直接调用：path.endswith。
    # 直接调用：candidates.append。
    # 直接调用：x.replace。
    # 直接调用：candidates.sort。
    # 集合派生：x 来自 self.proj_info；过滤 os.path.basename(x.replace('/', os.sep)) == os.path.basename(header_name)。
    # [/maint-note:4515ab11851f]
    def is_local_include(self, file_path, include_info):
        if isinstance(include_info, list) and len(include_info) > 0:
            header_name = include_info[0]
        else:
            header_name = include_info
        
        header_base = os.path.splitext(os.path.basename(header_name))[0]
        if header_base in self.standard_libraries:
            return None
        
        normalized_header_name = self._normalize_path(header_name)
        normalized_include_suffix = self._normalize_path(os.path.join("include", header_name))

        candidates = []
        for path in self.proj_info:
            if path.endswith(normalized_header_name) or path.endswith(normalized_include_suffix):
                candidates.append(path)
        
        if not candidates:
            candidates = [
                x for x in self.proj_info
                if os.path.basename(x.replace("/", os.sep)) == os.path.basename(header_name)
            ]
        
        if candidates:
            if len(candidates) > 1:
                candidates.sort(key=lambda x: self.get_distance_paths(file_path, x))
            
            return [candidates[0], None]
        
        return None
    
    # [maint-note:f599becfd866]
    # 维护导航：get_distance_paths；下列为本函数源码中的接口与路径索引。
    # 输入 src_path：类型由调用方与函数体约束。
    # 输入 target_path：类型由调用方与函数体约束。
    # 分支条件：src_parts[i] == target_parts[i]。
    # 遍历边界：i ← range(min(len(src_parts), len(target_parts)))。
    # 返回路径：len(src_parts) + len(target_parts) - 2 * common_len。
    # 直接调用：self._normalize_path(src_path).split。
    # 直接调用：self._normalize_path。
    # 直接调用：self._normalize_path(target_path).split。
    # 直接调用：range。
    # 直接调用：min。
    # 直接调用：len。
    # 集合派生：part 来自 self._normalize_path(src_path).split('/')；过滤 part。
    # 集合派生：part 来自 self._normalize_path(target_path).split('/')；过滤 part。
    # [/maint-note:f599becfd866]
    def get_distance_paths(self, src_path, target_path):
        src_parts = [part for part in self._normalize_path(src_path).split("/") if part]
        target_parts = [part for part in self._normalize_path(target_path).split("/") if part]
        
        common_len = 0
        for i in range(min(len(src_parts), len(target_parts))):
            if src_parts[i] == target_parts[i]:
                common_len += 1
            else:
                break
        
        return len(src_parts) + len(target_parts) - 2 * common_len
    
    # [maint-note:42fc7654a345]
    # 维护导航：_get_indent；下列为本函数源码中的接口与路径索引。
    # 输入 def_stat：类型由调用方与函数体约束。
    # 分支条件：len(lines) <= 1。
    # 分支条件：line.strip()。
    # 遍历边界：line ← reversed(lines)。
    # 返回路径：''。
    # 返回路径：line[:len(line) - len(line.lstrip())]。
    # 直接调用：def_stat.split。
    # 直接调用：len。
    # 直接调用：reversed。
    # 直接调用：line.strip。
    # 直接调用：line.lstrip。
    # [/maint-note:42fc7654a345]
    def _get_indent(self, def_stat):
        lines = def_stat.split('\n')
        if len(lines) <= 1:
            return ''
        
        for line in reversed(lines):
            if line.strip():
                return line[:len(line) - len(line.lstrip())]
        
        return ''
    
    # [maint-note:d239f91b7462]
    # 维护导航：_get_file_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 file_info：类型由调用方与函数体约束。
    # 输入 name_set：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：enable_docstring。
    # 分支条件：'' in file_info and 'docstring' in file_info['']。
    # 分支条件：file_info.get(file_info.get('module_name', ''), {}).get('docstring')。
    # 分支条件：doc。
    # 分支条件：sline == -1。
    # 分支条件：file_info[name].get('include')。
    # 分支条件：isinstance(include_info, list) and len(include_info) > 0。
    # 分支条件：'/' in header or '\\' in header。
    # 分支条件：includes。
    # 分支条件：len(name_list) > 1。
    # 分支条件：file_info[name]['type'] != 'Variable'。
    # 分支条件：name_type == 'Variable'。
    # 分支条件：any((x in name_set for x in name_list))。
    # 分支条件：name_type == 'Function'。
    # 分支条件：name_type == 'Struct'。
    # 分支条件：name_type == 'Union'。
    # 分支条件：name_type == 'Enum'。
    # 遍历边界：(sline, name_list) ← groupby(global_names, key=lambda x: file_info[x].get('sline', -1))。
    # 遍历边界：name ← name_list。
    # 返回路径：'\n'.join(prompt_list)。
    # 直接调用：file_info.get(file_info.get('module_name', ''), {}).get。
    # 直接调用：file_info.get。
    # 直接调用：file_info[module_name].get。
    # 直接调用：prompt_list.append。
    # 直接调用：file_info.items。
    # 直接调用：v.get。
    # 直接调用：global_names.sort。
    # 直接调用：groupby。
    # 直接调用：list。
    # 直接调用：file_info[name].get。
    # 直接调用：isinstance。
    # 直接调用：len。
    # 直接调用：includes.append。
    # 直接调用：'\n'.join。
    # 直接调用：any。
    # 直接调用：self._get_variable_prompt。
    # 直接调用：self._get_function_prompt。
    # 直接调用：self._get_struct_prompt。
    # 直接调用：self._get_union_prompt。
    # 直接调用：self._get_enum_prompt。
    # 直接调用：x.rstrip。
    # 集合派生：(k, v) 来自 file_info.items()；过滤 k and (not v.get('in_struct', False)) and (not v.get('in_function', False))。
    # 集合派生：x 来自 name_list。
    # 集合派生：x 来自 prompt_list。
    # [/maint-note:d239f91b7462]
    def _get_file_prompt(self, file_info, name_set, only_def=True, enable_docstring=True):

        prompt_list = []

        if enable_docstring:
            doc = None
            if '' in file_info and 'docstring' in file_info['']:
                doc = file_info['']['docstring']
            elif file_info.get(file_info.get('module_name', ''), {}).get('docstring'):
                module_name = file_info.get('module_name', '')
                doc = file_info[module_name].get('docstring')
                
            if doc:
                prompt_list.append(doc)
        
        global_names = [k for k, v in file_info.items() 
                       if k and not v.get('in_struct', False) and not v.get('in_function', False)]
        
        global_names.sort(key=lambda x: file_info[x].get('sline', -1))
        
        for sline, name_list in groupby(global_names, key=lambda x: file_info[x].get('sline', -1)):
            name_list = list(name_list)
            
            if sline == -1:
                includes = []
                for name in name_list:
                    if file_info[name].get('include'):
                        include_info = file_info[name]['include']
                        if isinstance(include_info, list) and len(include_info) > 0:
                            header = include_info[0]
                        else:
                            header = include_info
                            
                        if '/' in header or '\\' in header:
                            includes.append(f'#include "{header}"')
                        else:
                            includes.append(f'#include <{header}>')
                
                if includes:
                    prompt_list.append('\n'.join(includes))
                continue
            
            if len(name_list) > 1:
                for name in name_list:
                    if file_info[name]['type'] != 'Variable':
                        name_list = [name]
                        break
            
            name = name_list[0]
            name_info = file_info[name]
            name_type = name_info['type']
            
            if name_type == 'Variable':
                if any(x in name_set for x in name_list):
                    prompt_list.append(self._get_variable_prompt(file_info, name_list, False))
                else:
                    prompt_list.append(self._get_variable_prompt(file_info, name_list, only_def))
            
            elif name_type == 'Function':
                prompt_list.append(self._get_function_prompt(name_info, only_def, enable_docstring))
            
            elif name_type == 'Struct':
                tmp_set = name_set | {name}
                prompt_list.append(self._get_struct_prompt(file_info, name, {}, tmp_set, only_def, enable_docstring))
            
            elif name_type == 'Union':
                tmp_set = name_set | {name}
                prompt_list.append(self._get_union_prompt(file_info, name, {}, tmp_set, only_def, enable_docstring))
            
            elif name_type == 'Enum':
                prompt_list.append(self._get_enum_prompt(name_info, only_def, enable_docstring))
        
        prompt_list = [x.rstrip() for x in prompt_list]
        return '\n'.join(prompt_list)
    
    # [maint-note:29a84dc8faa8]
    # 维护导航：_get_struct_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 file_info：类型由调用方与函数体约束。
    # 输入 struct_name：类型由调用方与函数体约束。
    # 输入 struct_dict：类型由调用方与函数体约束。
    # 输入 name_set：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：enable_docstring and 'docstring' in file_info[struct_name]。
    # 分支条件：struct_name in name_set。
    # 分支条件：file_info[members[0]]['type'] == 'Variable'。
    # 分支条件：any((x in name_set for x in members))。
    # 分支条件：file_info[members[0]]['type'] in ('Struct', 'Union')。
    # 遍历边界：(sline, members) ← groupby(member_names, key=lambda x: file_info[x]['sline'])。
    # 返回路径：f'\n{struct_indent}'.join(prompt_list)。
    # 直接调用：self._get_indent。
    # 直接调用：prompt_list.append。
    # 直接调用：file_info.items。
    # 直接调用：v.get。
    # 直接调用：member_names.sort。
    # 直接调用：groupby。
    # 直接调用：list。
    # 直接调用：any。
    # 直接调用：self._get_variable_prompt。
    # 直接调用：self._get_struct_prompt。
    # 直接调用：struct_dict.get。
    # 直接调用：x.rstrip。
    # 直接调用：f'\n{struct_indent}'.join。
    # 集合派生：(k, v) 来自 file_info.items()；过滤 v.get('in_struct', None) == struct_name。
    # 集合派生：x 来自 members。
    # 集合派生：x 来自 prompt_list。
    # [/maint-note:29a84dc8faa8]
    def _get_struct_prompt(self, file_info, struct_name, struct_dict, name_set, only_def=True, enable_docstring=True):
        def_content = file_info[struct_name]['def']
        struct_indent = self._get_indent(def_content)
        
        prompt_list = [def_content]
        
        if enable_docstring and 'docstring' in file_info[struct_name]:
            prompt_list.append(file_info[struct_name]['docstring'])
        
        if struct_name in name_set:
            member_names = [k for k, v in file_info.items() 
                           if v.get('in_struct', None) == struct_name]
            
            member_names.sort(key=lambda x: file_info[x]['sline'])
            
            for sline, members in groupby(member_names, key=lambda x: file_info[x]['sline']):
                members = list(members)
                
                if file_info[members[0]]['type'] == 'Variable':
                    if any(x in name_set for x in members):
                        prompt_list.append(self._get_variable_prompt(file_info, members, False))
                    else:
                        prompt_list.append(self._get_variable_prompt(file_info, members, only_def))
                
                elif file_info[members[0]]['type'] in ('Struct', 'Union'):
                    inner_name = members[0]
                    prompt_list.append(self._get_struct_prompt(
                        file_info, inner_name, struct_dict, name_set | {inner_name}, 
                        only_def, enable_docstring
                    ))
        
        else:
            member_names = struct_dict.get(struct_name, [])
            member_names.sort(key=lambda x: file_info[x]['sline'])
            
            for sline, members in groupby(member_names, key=lambda x: file_info[x]['sline']):
                members = list(members)
                
                if file_info[members[0]]['type'] == 'Variable':
                    if any(x in name_set for x in members):
                        prompt_list.append(self._get_variable_prompt(file_info, members, False))
                    else:
                        prompt_list.append(self._get_variable_prompt(file_info, members, only_def))
                
                elif file_info[members[0]]['type'] in ('Struct', 'Union'):
                    inner_name = members[0]
                    prompt_list.append(self._get_struct_prompt(
                        file_info, inner_name, struct_dict, name_set,
                        only_def, enable_docstring
                    ))
        
        prompt_list = [x.rstrip() for x in prompt_list]
        return f'\n{struct_indent}'.join(prompt_list)
    
    # [maint-note:ea7fe0336eaa]
    # 维护导航：_get_union_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 file_info：类型由调用方与函数体约束。
    # 输入 union_name：类型由调用方与函数体约束。
    # 输入 union_dict：类型由调用方与函数体约束。
    # 输入 name_set：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 返回路径：self._get_struct_prompt(file_info, union_name, union_dict, name_set, only_def, enable_docstring)。
    # 直接调用：self._get_struct_prompt。
    # [/maint-note:ea7fe0336eaa]
    def _get_union_prompt(self, file_info, union_name, union_dict, name_set, only_def=True, enable_docstring=True):
        return self._get_struct_prompt(file_info, union_name, union_dict, name_set, only_def, enable_docstring)
    
    # [maint-note:0ade9d42d533]
    # 维护导航：_get_enum_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 node_info：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：only_def and enable_docstring and ('docstring' in node_info)。
    # 分支条件：not only_def and 'body' in node_info。
    # 返回路径：prompt。
    # [/maint-note:0ade9d42d533]
    def _get_enum_prompt(self, node_info, only_def=True, enable_docstring=True):
        prompt = node_info['def']
        
        if only_def and enable_docstring and 'docstring' in node_info:
            prompt += node_info['docstring']
        elif not only_def and 'body' in node_info:
            prompt += node_info['body']
        
        return prompt
    
    # [maint-note:70a90491c6d6]
    # 维护导航：_get_function_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 node_info：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：only_def。
    # 分支条件：enable_docstring and 'docstring' in node_info。
    # 分支条件：not prompt.rstrip().endswith(';')。
    # 分支条件：'body' in node_info。
    # 返回路径：prompt。
    # 直接调用：prompt.rstrip().endswith。
    # 直接调用：prompt.rstrip。
    # [/maint-note:70a90491c6d6]
    def _get_function_prompt(self, node_info, only_def=True, enable_docstring=True):
        prompt = node_info['def']
        
        if only_def:
            if enable_docstring and 'docstring' in node_info:
                prompt += node_info['docstring']
            if not prompt.rstrip().endswith(';'):
                prompt += ';'
        elif 'body' in node_info:
            prompt += node_info['body']
        
        return prompt
    
    # [maint-note:26c7edcaea9f]
    # 维护导航：_get_variable_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 file_info：类型由调用方与函数体约束。
    # 输入 name_list：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：not only_def or len(name_list) == 1。
    # 分支条件：struct_name。
    # 分支条件：func_name。
    # 遍历边界：name ← name_list。
    # 返回路径：name_info['def']。
    # 返回路径：', '.join(ret)。
    # 直接调用：len。
    # 直接调用：name_info.get。
    # 直接调用：name.split。
    # 直接调用：ret.append。
    # 直接调用：', '.join。
    # [/maint-note:26c7edcaea9f]
    def _get_variable_prompt(self, file_info, name_list, only_def=True):
        if not only_def or len(name_list) == 1:
            name_info = file_info[name_list[0]]
            return name_info['def']
        
        ret = []
        for name in name_list:
            name_info = file_info[name]
            
            struct_name = name_info.get('in_struct', None)
            if struct_name:
                name = name.split('.')[-1]  
            
            func_name = name_info.get('in_function', None)
            if func_name:
                name = f'    {name}'  
            
            ret.append(name)
        
        return ', '.join(ret)
    
    # [maint-note:db9598ee3bc1]
    # 维护导航：get_path_comment；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 返回路径：f'/* {fpath} */\n'。
    # [/maint-note:db9598ee3bc1]
    def get_path_comment(self, fpath):
        return f"/* {fpath} */\n"
    
    # [maint-note:301717d270a3]
    # 维护导航：get_prompt4names；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 输入 name_set：类型由调用方与函数体约束。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：file_info is None。
    # 分支条件：'' in name_set or None in name_set。
    # 分支条件：name not in file_info。
    # 分支条件：struct_name not in struct_dict。
    # 分支条件：sline == -1。
    # 分支条件：file_info[name].get('include')。
    # 分支条件：'/' in header or '\\' in header。
    # 分支条件：includes。
    # 分支条件：len(names) > 1。
    # 分支条件：file_info[name]['type'] != 'Variable'。
    # 分支条件：name_type == 'Variable'。
    # 分支条件：name_type == 'Function'。
    # 分支条件：name_type == 'Struct'。
    # 分支条件：name_type == 'Union'。
    # 分支条件：name_type == 'Enum'。
    # 遍历边界：name ← name_set。
    # 遍历边界：struct_name is not None。
    # 遍历边界：(sline, names) ← groupby(global_names, key=lambda x: file_info[x].get('sline', -1))。
    # 遍历边界：name ← names。
    # 返回路径：None。
    # 返回路径：path_comment + self._get_file_prompt(file_info, name_set, only_def, enable_docstring)。
    # 返回路径：path_comment + '\n'.join(prompt_list)。
    # 写入目标：struct_dict[struct_name]。
    # 直接调用：self.proj_info.get。
    # 直接调用：self.get_path_comment。
    # 直接调用：self._get_file_prompt。
    # 直接调用：set。
    # 直接调用：file_info[name].get。
    # 直接调用：struct_dict[struct_name].add。
    # 直接调用：global_names.add。
    # 直接调用：sorted。
    # 直接调用：groupby。
    # 直接调用：list。
    # 直接调用：includes.append。
    # 直接调用：prompt_list.append。
    # 直接调用：'\n'.join。
    # 直接调用：len。
    # 直接调用：self._get_variable_prompt。
    # 直接调用：self._get_function_prompt。
    # 直接调用：self._get_struct_prompt。
    # 直接调用：self._get_union_prompt。
    # 直接调用：self._get_enum_prompt。
    # 直接调用：x.rstrip。
    # 集合派生：x 来自 prompt_list。
    # [/maint-note:301717d270a3]
    def get_prompt4names(self, fpath, name_set, only_def=True, enable_docstring=True):
        file_info = self.proj_info.get(fpath, None)
        if file_info is None:
            return None
        
        path_comment = self.get_path_comment(fpath)
        
        if '' in name_set or None in name_set:
            return path_comment + self._get_file_prompt(file_info, name_set, only_def, enable_docstring)
        
        struct_dict = {}
        global_names = set()
        
        for name in name_set:
            if name not in file_info:
                continue
                
            struct_name = file_info[name].get('in_struct', None)
            while struct_name is not None:
                if struct_name not in struct_dict:
                    struct_dict[struct_name] = {name}
                else:
                    struct_dict[struct_name].add(name)
                
                name = struct_name
                struct_name = file_info[name].get('in_struct', None)
            
            global_names.add(name)
        
        prompt_list = []
        
        global_names = sorted(global_names, key=lambda x: file_info[x].get('sline', -1))
        
        for sline, names in groupby(global_names, key=lambda x: file_info[x].get('sline', -1)):
            names = list(names)
            
            if sline == -1:
                includes = []
                for name in names:
                    if file_info[name].get('include'):
                        header = file_info[name]['include']
                        if '/' in header or '\\' in header:
                            includes.append(f'#include "{header}"')
                        else:
                            includes.append(f'#include <{header}>')
                
                if includes:
                    prompt_list.append('\n'.join(includes))
                continue

            if len(names) > 1:
                for name in names:
                    if file_info[name]['type'] != 'Variable':
                        names = [name]
                        break
            
            name = names[0]
            name_info = file_info[name]
            name_type = name_info['type']
            
            if name_type == 'Variable':
                prompt_list.append(self._get_variable_prompt(file_info, names, False))
            
            elif name_type == 'Function':
                prompt_list.append(self._get_function_prompt(name_info, only_def, enable_docstring))
            
            elif name_type == 'Struct':
                prompt_list.append(self._get_struct_prompt(
                    file_info, name, struct_dict, name_set, 
                    only_def, enable_docstring
                ))
            
            elif name_type == 'Union':
                prompt_list.append(self._get_union_prompt(
                    file_info, name, struct_dict, name_set,
                    only_def, enable_docstring
                ))
            
            elif name_type == 'Enum':
                prompt_list.append(self._get_enum_prompt(name_info, only_def, enable_docstring))
        
        prompt_list = [x.rstrip() for x in prompt_list]
        return path_comment + '\n'.join(prompt_list)
    
    # [maint-note:34e5552f9608]
    # 维护导航：pseudo_topo_sort；下列为本函数源码中的接口与路径索引。
    # 输入 fpath_set：类型由调用方与函数体约束。
    # 输入 file_edges：类型由调用方与函数体约束。
    # 输入 fpath_order：类型由调用方与函数体约束。
    # 分支条件：item not in in_table。
    # 分支条件：item not in out_table。
    # 分支条件：x not in fpath_set。
    # 分支条件：x not in in_table。
    # 分支条件：in_degree < min_degree。
    # 分支条件：in_degree == min_degree。
    # 分支条件：node_list[min_index] in fpath_order and item in fpath_order。
    # 分支条件：fpath_order.index(item) < fpath_order.index(node_list[min_index])。
    # 分支条件：node_list[min_index] not in fpath_order and item not in fpath_order。
    # 分支条件：item < node_list[min_index]。
    # 分支条件：item not in fpath_order。
    # 分支条件：x in in_table。
    # 遍历边界：item ← fpath_set。
    # 遍历边界：x ← file_edges.get(item, [])。
    # 遍历边界：in_table。
    # 遍历边界：i ← range(1, len(node_list))。
    # 遍历边界：x ← out_table.pop(item, [])。
    # 返回路径：list(reversed(sort_list))。
    # 写入目标：in_table[item]。
    # 写入目标：out_table[item]。
    # 写入目标：in_table[x]。
    # 直接调用：file_edges.get。
    # 直接调用：out_table[item].append。
    # 直接调用：in_table[x].append。
    # 直接调用：list。
    # 直接调用：len。
    # 直接调用：range。
    # 直接调用：fpath_order.index。
    # 直接调用：sort_list.append。
    # 直接调用：out_table.pop。
    # 直接调用：in_table[x].remove。
    # 直接调用：in_table.pop。
    # 直接调用：reversed。
    # [/maint-note:34e5552f9608]
    def pseudo_topo_sort(self, fpath_set, file_edges, fpath_order):
        in_table = {}  
        out_table = {} 
        
        for item in fpath_set:
            if item not in in_table:
                in_table[item] = []
            if item not in out_table:
                out_table[item] = []
            
            for x in file_edges.get(item, []):
                if x not in fpath_set:
                    continue
                
                out_table[item].append(x)
                if x not in in_table:
                    in_table[x] = [item]
                else:
                    in_table[x].append(item)
        
        sort_list = []
        while in_table:
            node_list = list(in_table)
            
            min_index = 0
            min_degree = len(in_table[node_list[min_index]])
            
            for i in range(1, len(node_list)):
                item = node_list[i]
                in_degree = len(in_table[item])
                
                if in_degree < min_degree:
                    min_index = i
                    min_degree = in_degree
                elif in_degree == min_degree:
                    if node_list[min_index] in fpath_order and item in fpath_order:
                        if fpath_order.index(item) < fpath_order.index(node_list[min_index]):
                            min_index = i
                    elif node_list[min_index] not in fpath_order and item not in fpath_order:
                        if item < node_list[min_index]:
                            min_index = i
                    else:
                        if item not in fpath_order:
                            min_index = i
            
            item = node_list[min_index]
            sort_list.append(item)
            
            for x in out_table.pop(item, []):
                if x in in_table:
                    in_table[x].remove(item)
            
            in_table.pop(item, None)
        
        return list(reversed(sort_list))
    
    # [maint-note:e3a449231662]
    # 维护导航：depthFirstSearch；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 输入 name：类型由调用方与函数体约束。
    # 输入 max_hop：类型由调用方与函数体约束；缺省表达式 None。
    # 返回路径：(node_dict, file_edges)。
    # 直接调用：self.dfs。
    # [/maint-note:e3a449231662]
    def depthFirstSearch(self, fpath, name, max_hop=None):
        node_dict = {}  # {fpath: set(name)}
        file_edges = {} # {fpath: [fpath]}
        
        self.dfs(fpath, name, 0, node_dict, file_edges, max_hop)
        
        return node_dict, file_edges
    
    # [maint-note:8d1bd0d3ef48]
    # 维护导航：dfs；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 输入 name：类型由调用方与函数体约束。
    # 输入 depth：类型由调用方与函数体约束。
    # 输入 node_dict：类型由调用方与函数体约束。
    # 输入 file_edges：类型由调用方与函数体约束。
    # 输入 max_hop：类型由调用方与函数体约束。
    # 分支条件：fpath not in self.proj_info or name not in self.proj_info[fpath]。
    # 分支条件：fpath in node_dict and name in node_dict[fpath]。
    # 分支条件：fpath not in node_dict。
    # 分支条件：max_hop is not None and depth + 1 > max_hop。
    # 分支条件：'include' in node_info。
    # 分支条件：isinstance(include_info, list) and len(include_info) > 0。
    # 分支条件：len(include_info) == 2。
    # 分支条件：t_fpath not in file_edges。
    # 分支条件：fpath not in file_edges。
    # 分支条件：t_name。
    # 分支条件：'rels' in node_info。
    # 遍历边界：item ← node_info['rels']。
    # 返回路径：无返回值。
    # 写入目标：node_dict[fpath]。
    # 写入目标：file_edges[t_fpath]。
    # 写入目标：file_edges[fpath]。
    # 直接调用：node_dict[fpath].add。
    # 直接调用：isinstance。
    # 直接调用：len。
    # 直接调用：file_edges[fpath].append。
    # 直接调用：self.dfs。
    # [/maint-note:8d1bd0d3ef48]
    def dfs(self, fpath, name, depth, node_dict, file_edges, max_hop):

        if fpath not in self.proj_info or name not in self.proj_info[fpath]:
            return

        if fpath in node_dict and name in node_dict[fpath]:
            return

        node_info = self.proj_info[fpath][name]
        if fpath not in node_dict:
            node_dict[fpath] = {name}
        else:
            node_dict[fpath].add(name)
        
        if max_hop is not None and depth+1 > max_hop:
            return
        
        if 'include' in node_info:
            include_info = node_info['include']
            
            if isinstance(include_info, list) and len(include_info) > 0:
                if len(include_info) == 2:
                    t_fpath, t_name = include_info
                else:
                    t_fpath, t_name = include_info[0], None
            else:
                t_fpath, t_name = include_info, None
            
            if t_fpath not in file_edges:
                file_edges[t_fpath] = []
            
            if fpath not in file_edges:
                file_edges[fpath] = [t_fpath]
            else:
                file_edges[fpath].append(t_fpath)
            
            if t_name:
                self.dfs(t_fpath, t_name, depth+1, node_dict, file_edges, max_hop)
       
        if 'rels' in node_info:
            for item in node_info['rels']:
                t_name = item[0]
                self.dfs(fpath, t_name, depth+1, node_dict, file_edges, max_hop)
    
    # [maint-note:4bb16db4d5c8]
    # 维护导航：get_prompt；下列为本函数源码中的接口与路径索引。
    # 输入 node_list：类型由调用方与函数体约束。
    # 输入 max_hop：类型由调用方与函数体约束；缺省表达式 None。
    # 输入 only_def：类型由调用方与函数体约束；缺省表达式 True。
    # 输入 enable_docstring：类型由调用方与函数体约束；缺省表达式 True。
    # 分支条件：fpath not in fpath_order。
    # 分支条件：k not in node_dict。
    # 分支条件：k not in file_edges。
    # 分支条件：fpath in node_dict。
    # 分支条件：prompt。
    # 遍历边界：(fpath, name) ← node_list。
    # 遍历边界：(k, v) ← tmp_nodes.items()。
    # 遍历边界：(k, v) ← tmp_edges.items()。
    # 遍历边界：fpath ← sorted_files。
    # 返回路径：'\n\n'.join(prompt_list)。
    # 写入目标：node_dict[k]。
    # 写入目标：file_edges[k]。
    # 直接调用：fpath_order.append。
    # 直接调用：self.depthFirstSearch。
    # 直接调用：tmp_nodes.items。
    # 直接调用：node_dict[k].update。
    # 直接调用：tmp_edges.items。
    # 直接调用：file_edges[k].extend。
    # 直接调用：self.pseudo_topo_sort。
    # 直接调用：set。
    # 直接调用：self.get_prompt4names。
    # 直接调用：prompt_list.append。
    # 直接调用：'\n\n'.join。
    # [/maint-note:4bb16db4d5c8]
    def get_prompt(self, node_list, max_hop=None, only_def=True, enable_docstring=True):
        
        node_dict = {}  # {fpath: set(name)}
        file_edges = {} # {fpath: [fpath]}
        fpath_order = []
        
        for fpath, name in node_list:
            if fpath not in fpath_order:
                fpath_order.append(fpath)
            
            tmp_nodes, tmp_edges = self.depthFirstSearch(fpath, name, max_hop)
            
            for k, v in tmp_nodes.items():
                if k not in node_dict:
                    node_dict[k] = v
                else:
                    node_dict[k].update(v)
            
            for k, v in tmp_edges.items():
                if k not in file_edges:
                    file_edges[k] = v
                else:
                    file_edges[k].extend(v)
        
        sorted_files = self.pseudo_topo_sort(set(node_dict), file_edges, fpath_order)
        
        prompt_list = []
        for fpath in sorted_files:
            if fpath in node_dict:
                prompt = self.get_prompt4names(fpath, node_dict[fpath], only_def, enable_docstring)
                if prompt:
                    prompt_list.append(prompt)
        
        return '\n\n'.join(prompt_list)
