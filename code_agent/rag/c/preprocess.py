import os
import re
import time
import json
from .cfile_parse import CParser
from .node_prompt import CProjectSearcher
from .utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR


class CProjectParser(object):
    def __init__(self): 
        self.c_parser = CParser()
        self.file_pattern = re.compile(r'[^\w\-]')
        self.header_pattern = re.compile(r'\.(h|hpp)$')
        self.source_pattern = re.compile(r'\.(c|cpp)$')
        
        self.proj_searcher = CProjectSearcher()
        
        self.proj_dir = None
        self.parse_res = None
    
    def set_proj_dir(self, dir_path):
        self.proj_dir = os.path.abspath(dir_path)

        print(f"\n📁 设置项目目录: {self.proj_dir}")

    def _normalize_module_path(self, path: str) -> str:
        return path.replace("\\", "/")

    def retain_project_rels(self):
        print("🔧 清洗项目内引用关系中 ...")
        for module, file_info in self.parse_res.items():
            for name, info_dict in file_info.items():
                struct_name = info_dict.get("in_struct", None)
                
                rels = info_dict.get("rels", None)
                if rels is not None:
                    del_index = []
                    for i, item in enumerate(rels):
                        find_info = self.proj_searcher.name_in_file(item[0], list(file_info), name, struct_name)

                        if find_info is None:
                            del_index.append(i)
                        else:
                            if len(item) == 2:
                                info_dict["rels"][i] = [find_info[0], find_info[1], item[1]]

                    # 移除无效关系
                    for index in reversed(del_index):
                        info_dict["rels"].pop(index)
                    
                    if len(info_dict["rels"]) == 0:
                        info_dict.pop("rels")

                include_info = info_dict.get("include", None)
                if info_dict["type"] == 'Variable' and include_info is not None:
                    judge_res = self.proj_searcher.is_local_include(module, include_info)
                    if judge_res is None:
                        info_dict.pop("include")
                    else:
                        info_dict["include"] = judge_res
    
    def _get_all_c_file_paths(self, target_path):
        print(f"🔍 扫描 C/C++ 文件: {target_path}")
        if not os.path.isdir(target_path):
            print("❌ 目录不存在！")
            return {}
        
        dir_list = [target_path]
        c_dict = {} 
        
        while len(dir_list) > 0:
            c_dir = dir_list.pop()
            c_dict[c_dir] = set()
            
            for item in os.listdir(c_dir):
                if item.startswith('.'): 
                    continue
                    
                fpath = os.path.join(c_dir, item)
                if os.path.isdir(fpath):
                    if re.search(self.file_pattern, item) is None:
                        dir_list.append(fpath)
                        c_dict[c_dir].add(fpath)
                elif os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath)):
                    if re.search(self.file_pattern, os.path.splitext(item)[0]) is None:
                        c_dict[c_dir].add(fpath)
        
        return c_dict
    
    def _get_module_name(self, fpath):
        rel_path = os.path.relpath(os.path.abspath(fpath), self.proj_dir)
        return self._normalize_module_path(rel_path)
    
    def parse_dir(self, c_proj_dir):
        self.set_proj_dir(c_proj_dir)

        c_dict = self._get_all_c_file_paths(c_proj_dir)

        # 收集文件
        c_files = set()
        for dir_path, file_set in c_dict.items():
            for fpath in file_set:
                if os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath)):
                    c_files.add(fpath)

        print(f"📄 共找到 {len(c_files)} 个 C/C++ 源文件")

        self.parse_res = {}
        for fpath in sorted(c_files):
            module = self._get_module_name(fpath)
            # print(f"  ➤ 解析文件: {module}")

            try:
                self.c_parser.set_file_path(fpath)
                info_dict = self.c_parser.parse(fpath)
                
                if info_dict and len(info_dict) > 0:
                    self.parse_res[module] = info_dict
                    # print(f"     ✓ 成功解析")
                else:
                    print("     ⚠️ 空文件或无有效信息")
            except Exception as e:
                print(f"     ❌ 解析失败: {e}")
        
        # 设置工程上下文
        self.proj_searcher.set_proj(c_proj_dir, self.parse_res)

        # 清洗关系
        self.retain_project_rels()
        
        print(f"✅ 目录解析完成: {c_proj_dir}")
        print(f"📦 共解析 {len(self.parse_res)} 个文件节点\n")

        return self.parse_res


# ---------------- MAIN ----------------
if __name__ == '__main__':
    t_all0 = time.perf_counter()

    with open(DS_FILE, 'r') as f:
        ds = [json.loads(line) for line in f.readlines()]

    pkg_set = set([x['pkg'] for x in ds])
    print(f"📚 数据集包含 {len(pkg_set)} 个仓库")

    project_parser = CProjectParser()

    if not os.path.isdir(DS_GRAPH_DIR):
        os.mkdir(DS_GRAPH_DIR)
        print(f"📁 创建输出目录: {DS_GRAPH_DIR}")

    print(f"\n🚀 开始解析仓库: {DS_REPO_DIR}\n")

    total = 0
    skipped = 0
    success = 0
    failed = 0

    for item in os.listdir(DS_REPO_DIR):
        if item not in pkg_set:
            continue

        dir_path = os.path.join(DS_REPO_DIR, item)
        if not os.path.isdir(dir_path):
            continue

        total += 1

        # ✅ 如果输出文件已存在，跳过
        output_path = os.path.join(DS_GRAPH_DIR, f'{item}.json')
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            skipped += 1
            print(f"⏭️ 跳过已处理仓库: {item} （已存在: {output_path}）")
            continue

        print(f"\n==============================")
        print(f"🔷 解析仓库: {item}")
        print(f"📂 路径: {dir_path}")
        print(f"==============================\n")

        t0 = time.perf_counter()
        try:
            info = project_parser.parse_dir(dir_path)

            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2)

            dt = time.perf_counter() - t0
            success += 1
            print(f"💾 已写出图文件: {output_path}")
            print(f"⏱️ 仓库耗时: {dt:.2f}s")
        except Exception as e:
            dt = time.perf_counter() - t0
            failed += 1
            print(f"❌ 处理仓库 {item} 时出错: {e}")
            print(f"⏱️ 出错前耗时: {dt:.2f}s")

    # 显示输出目录内容
    visible_files = [
        f for f in os.listdir(DS_GRAPH_DIR)
        if not f.startswith('.')
        and os.path.isfile(os.path.join(DS_GRAPH_DIR, f))
    ]

    dt_all = time.perf_counter() - t_all0

    print(f"\n🎉 完成！共生成 {len(visible_files)} 个仓库语义图文件")
    print(f"📁 输出目录: {DS_GRAPH_DIR}")
    print(f"📄 文件列表: {visible_files}")

    print("\n📊 统计：")
    print(f"  - 总候选仓库数: {total}")
    print(f"  - 跳过（已处理）: {skipped}")
    print(f"  - 成功解析: {success}")
    print(f"  - 失败: {failed}")
    print(f"⏱️ 总耗时: {dt_all:.2f}s")
