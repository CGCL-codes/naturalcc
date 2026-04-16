import os
import json
from itertools import groupby
# 根据一个已经解析好的 C 项目信息，按名字查找相关定义，并把它们整理成适合展示或拼接成上下文提示的代码片段。

class CProjectSearcher(object):
    def __init__(self):
        self.proj_dir = None
        self.proj_info = None

        self.standard_libraries = {
            "stdio", "stdlib", "string", "math", "time", "ctype", "assert", 
            "limits", "float", "stddef", "stdarg", "stdbool", "stdint", "errno",
            "signal", "locale", "setjmp", "wchar", "wctype"
        }
    
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

    def _normalize_path(self, path):
        return str(path).replace("\\", "/")
    
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
    
    def _get_indent(self, def_stat):
        lines = def_stat.split('\n')
        if len(lines) <= 1:
            return ''
        
        for line in reversed(lines):
            if line.strip():
                return line[:len(line) - len(line.lstrip())]
        
        return ''
    
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
    
    def _get_union_prompt(self, file_info, union_name, union_dict, name_set, only_def=True, enable_docstring=True):
        return self._get_struct_prompt(file_info, union_name, union_dict, name_set, only_def, enable_docstring)
    
    def _get_enum_prompt(self, node_info, only_def=True, enable_docstring=True):
        prompt = node_info['def']
        
        if only_def and enable_docstring and 'docstring' in node_info:
            prompt += node_info['docstring']
        elif not only_def and 'body' in node_info:
            prompt += node_info['body']
        
        return prompt
    
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
    
    def get_path_comment(self, fpath):
        return f"/* {fpath} */\n"
    
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
    
    def depthFirstSearch(self, fpath, name, max_hop=None):
        node_dict = {}  # {fpath: set(name)}
        file_edges = {} # {fpath: [fpath]}
        
        self.dfs(fpath, name, 0, node_dict, file_edges, max_hop)
        
        return node_dict, file_edges
    
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
