import os
import re
import time
import json
from .cfile_parse import CParser
from .node_prompt import CProjectSearcher
from .utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR


# [maint-note:66013b8efed7]
# 数据与行为导航：CProjectParser。
# 继承接口：object。
# [/maint-note:66013b8efed7]
class CProjectParser(object):
    # [maint-note:0e390529ba95]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 写入目标：self.c_parser。
    # 写入目标：self.file_pattern。
    # 写入目标：self.header_pattern。
    # 写入目标：self.source_pattern。
    # 写入目标：self.proj_searcher。
    # 写入目标：self.proj_dir。
    # 写入目标：self.parse_res。
    # 直接调用：CParser。
    # 直接调用：re.compile。
    # 直接调用：CProjectSearcher。
    # [/maint-note:0e390529ba95]
    def __init__(self): 
        self.c_parser = CParser()
        self.file_pattern = re.compile(r'[^\w\-]')
        self.header_pattern = re.compile(r'\.(h|hpp)$')
        self.source_pattern = re.compile(r'\.(c|cpp)$')
        
        self.proj_searcher = CProjectSearcher()
        
        self.proj_dir = None
        self.parse_res = None
    
    # [maint-note:4cfcf3b2819d]
    # 维护导航：set_proj_dir；下列为本函数源码中的接口与路径索引。
    # 输入 dir_path：类型由调用方与函数体约束。
    # 写入目标：self.proj_dir。
    # 直接调用：os.path.abspath。
    # 直接调用：print。
    # [/maint-note:4cfcf3b2819d]
    def set_proj_dir(self, dir_path):
        self.proj_dir = os.path.abspath(dir_path)

        print(f"\n📁 设置项目目录: {self.proj_dir}")

    # [maint-note:15a240ab698c]
    # 维护导航：_normalize_module_path；下列为本函数源码中的接口与路径索引。
    # 输入 path：str。
    # 返回契约：str。
    # 返回路径：path.replace('\\', '/')。
    # 直接调用：path.replace。
    # [/maint-note:15a240ab698c]
    def _normalize_module_path(self, path: str) -> str:
        return path.replace("\\", "/")

    # [maint-note:caba3f0f5e4c]
    # 维护导航：retain_project_rels；下列为本函数源码中的接口与路径索引。
    # 分支条件：rels is not None。
    # 分支条件：find_info is None。
    # 分支条件：len(item) == 2。
    # 分支条件：len(info_dict['rels']) == 0。
    # 分支条件：info_dict['type'] == 'Variable' and include_info is not None。
    # 分支条件：judge_res is None。
    # 遍历边界：(module, file_info) ← self.parse_res.items()。
    # 遍历边界：(name, info_dict) ← file_info.items()。
    # 遍历边界：(i, item) ← enumerate(rels)。
    # 遍历边界：index ← reversed(del_index)。
    # 写入目标：info_dict['rels'][i]。
    # 写入目标：info_dict['include']。
    # 直接调用：print。
    # 直接调用：self.parse_res.items。
    # 直接调用：file_info.items。
    # 直接调用：info_dict.get。
    # 直接调用：enumerate。
    # 直接调用：self.proj_searcher.name_in_file。
    # 直接调用：list。
    # 直接调用：del_index.append。
    # 直接调用：len。
    # 直接调用：reversed。
    # 直接调用：info_dict['rels'].pop。
    # 直接调用：info_dict.pop。
    # 直接调用：self.proj_searcher.is_local_include。
    # [/maint-note:caba3f0f5e4c]
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
    
    # [maint-note:64caa90d5bc1]
    # 维护导航：_get_all_c_file_paths；下列为本函数源码中的接口与路径索引。
    # 输入 target_path：类型由调用方与函数体约束。
    # 分支条件：not os.path.isdir(target_path)。
    # 分支条件：item.startswith('.')。
    # 分支条件：os.path.isdir(fpath)。
    # 分支条件：re.search(self.file_pattern, item) is None。
    # 分支条件：os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath))。
    # 分支条件：re.search(self.file_pattern, os.path.splitext(item)[0]) is None。
    # 遍历边界：len(dir_list) > 0。
    # 遍历边界：item ← os.listdir(c_dir)。
    # 返回路径：{}。
    # 返回路径：c_dict。
    # 写入目标：c_dict[c_dir]。
    # 直接调用：print。
    # 直接调用：os.path.isdir。
    # 直接调用：len。
    # 直接调用：dir_list.pop。
    # 直接调用：set。
    # 直接调用：os.listdir。
    # 直接调用：item.startswith。
    # 直接调用：os.path.join。
    # 直接调用：re.search。
    # 直接调用：dir_list.append。
    # 直接调用：c_dict[c_dir].add。
    # 直接调用：os.path.isfile。
    # 直接调用：self.header_pattern.search。
    # 直接调用：self.source_pattern.search。
    # 直接调用：os.path.splitext。
    # [/maint-note:64caa90d5bc1]
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
    
    # [maint-note:3b6a303752f6]
    # 维护导航：_get_module_name；下列为本函数源码中的接口与路径索引。
    # 输入 fpath：类型由调用方与函数体约束。
    # 返回路径：self._normalize_module_path(rel_path)。
    # 直接调用：os.path.relpath。
    # 直接调用：os.path.abspath。
    # 直接调用：self._normalize_module_path。
    # [/maint-note:3b6a303752f6]
    def _get_module_name(self, fpath):
        rel_path = os.path.relpath(os.path.abspath(fpath), self.proj_dir)
        return self._normalize_module_path(rel_path)
    
    # [maint-note:db2df2dd1028]
    # 维护导航：parse_dir；下列为本函数源码中的接口与路径索引。
    # 输入 c_proj_dir：类型由调用方与函数体约束。
    # 分支条件：os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath))。
    # 分支条件：info_dict and len(info_dict) > 0。
    # 遍历边界：(dir_path, file_set) ← c_dict.items()。
    # 遍历边界：fpath ← file_set。
    # 遍历边界：fpath ← sorted(c_files)。
    # 返回路径：self.parse_res。
    # 捕获异常：Exception。
    # 写入目标：self.parse_res。
    # 写入目标：self.parse_res[module]。
    # 直接调用：self.set_proj_dir。
    # 直接调用：self._get_all_c_file_paths。
    # 直接调用：set。
    # 直接调用：c_dict.items。
    # 直接调用：os.path.isfile。
    # 直接调用：self.header_pattern.search。
    # 直接调用：self.source_pattern.search。
    # 直接调用：c_files.add。
    # 直接调用：print。
    # 直接调用：len。
    # 直接调用：sorted。
    # 直接调用：self._get_module_name。
    # 直接调用：self.c_parser.set_file_path。
    # 直接调用：self.c_parser.parse。
    # 直接调用：self.proj_searcher.set_proj。
    # 直接调用：self.retain_project_rels。
    # [/maint-note:db2df2dd1028]
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
