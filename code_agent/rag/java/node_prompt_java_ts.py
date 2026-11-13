import os
from typing import Dict, List, Optional, Tuple, Set

_PRIMITIVES = {
    "byte","short","int","long","float","double","boolean","char","void"
}

# [maint-note:5d98704386a4]
# 维护导航：_strip_type；下列为本函数源码中的接口与路径索引。
# 输入 t：str。
# 返回契约：str。
# 返回路径：t。
# 直接调用：t.strip。
# 直接调用：re.sub。
# 直接调用：t.replace。
# 直接调用：' '.join。
# 直接调用：t.split。
# [/maint-note:5d98704386a4]
def _strip_type(t: str) -> str:
    
    import re
    t = t.strip()
    t = re.sub(r"<.*?>", "", t)
    t = t.replace("...", "[]")
    t = t.replace("@", " ")
    t = " ".join(t.split())
    return t

# [maint-note:a6a5a8d1bd05]
# 维护导航：_simple_name；下列为本函数源码中的接口与路径索引。
# 输入 t：str。
# 返回契约：str。
# 返回路径：t.split('.')[-1].strip()。
# 直接调用：_strip_type。
# 直接调用：t.replace。
# 直接调用：t.split('.')[-1].strip。
# 直接调用：t.split。
# [/maint-note:a6a5a8d1bd05]
def _simple_name(t: str) -> str:
    t = _strip_type(t)
    t = t.replace("[]", "")
    return t.split(".")[-1].strip()

class JavaProjectSearcherTS:
    """
    Project-level resolver for Java types (project-internal).
    """
    # [maint-note:fec8823f0498]
    # 维护导航：__init__；下列为本函数源码中的接口与路径索引。
    # 写入目标：self.proj_dir。
    # 写入目标：self.proj_info。
    # 写入目标：self.fqcn_to_file。
    # 写入目标：self.pkg_simple_to_fqcn。
    # 写入目标：self.simple_to_fqcns。
    # 写入目标：self.std_prefix。
    # 写入目标：self.std_packages。
    # [/maint-note:fec8823f0498]
    def __init__(self):
        self.proj_dir: Optional[str] = None
        self.proj_info: Optional[Dict[str, Dict]] = None

        # indexes
        self.fqcn_to_file: Dict[str, str] = {}
        self.pkg_simple_to_fqcn: Dict[Tuple[str, str], str] = {}
        self.simple_to_fqcns: Dict[str, List[str]] = {}

        # standards
        self.std_prefix = ("java.", "javax.", "jakarta.", "kotlin.", "scala.", "android.")
        self.std_packages = {"java.lang"}  # implicit import

    # [maint-note:fa5b871820cb]
    # 维护导航：set_proj；下列为本函数源码中的接口与路径索引。
    # 输入 proj_dir：str。
    # 输入 proj_info：Dict[str, Dict]。
    # 写入目标：self.proj_dir。
    # 写入目标：self.proj_info。
    # 直接调用：proj_dir.endswith。
    # 直接调用：self._build_indexes。
    # 表达式分流：proj_dir.endswith(os.sep)；真值与假值分支分别计算，不同时执行。
    # [/maint-note:fa5b871820cb]
    def set_proj(self, proj_dir: str, proj_info: Dict[str, Dict]):
        self.proj_dir = proj_dir if proj_dir.endswith(os.sep) else proj_dir + os.sep
        self.proj_info = proj_info
        self._build_indexes()

    # [maint-note:f6e679f764bb]
    # 维护导航：_build_indexes；下列为本函数源码中的接口与路径索引。
    # 分支条件：pkg is None。
    # 分支条件：name == ''。
    # 分支条件：'#' in name。
    # 分支条件：'.' in name。
    # 分支条件：'$' in name。
    # 分支条件：node.get('type') in ('Class', 'Interface', 'Enum', 'Annotation', 'Record')。
    # 遍历边界：(fpath, file_info) ← self.proj_info.items()。
    # 遍历边界：(name, node) ← file_info.items()。
    # 写入目标：self.fqcn_to_file[fqcn]。
    # 写入目标：self.pkg_simple_to_fqcn[pkg, name]。
    # 直接调用：self.fqcn_to_file.clear。
    # 直接调用：self.pkg_simple_to_fqcn.clear。
    # 直接调用：self.simple_to_fqcns.clear。
    # 直接调用：self.proj_info.items。
    # 直接调用：file_info.get。
    # 直接调用：mod.get。
    # 直接调用：file_info.items。
    # 直接调用：node.get。
    # 直接调用：self.simple_to_fqcns.setdefault(name, []).append。
    # 直接调用：self.simple_to_fqcns.setdefault。
    # 表达式分流：pkg；真值与假值分支分别计算，不同时执行。
    # [/maint-note:f6e679f764bb]
    def _build_indexes(self):
        self.fqcn_to_file.clear()
        self.pkg_simple_to_fqcn.clear()
        self.simple_to_fqcns.clear()

        for fpath, file_info in self.proj_info.items():
            mod = file_info.get("", {})
            pkg = mod.get("package")  # may be None
            if pkg is None:
                pkg = ""  # default package

            # Find top-level type declarations in this file:
            # We store them by scanning nodes whose key doesn't contain '.' and doesn't contain '#' and isn't ""
            # (top-level class name = SimpleName, inner = Outer$Inner)
            for name, node in file_info.items():
                if name == "":
                    continue
                if "#" in name:
                    continue
                if "." in name:
                    continue
                # top-level: SimpleName or Outer$Inner; choose only top-level SimpleName (no $)
                if "$" in name:
                    continue
                if node.get("type") in ("Class","Interface","Enum","Annotation","Record"):
                    fqcn = f"{pkg}.{name}" if pkg else name
                    self.fqcn_to_file[fqcn] = fpath
                    self.pkg_simple_to_fqcn[(pkg, name)] = fqcn
                    self.simple_to_fqcns.setdefault(name, []).append(fqcn)

    # [maint-note:4a7b08ed1f31]
    # 维护导航：resolve_type；下列为本函数源码中的接口与路径索引。
    # 输入 cur_file：str。
    # 输入 type_name：str。
    # 返回契约：Optional[str]。
    # 分支条件：not self.proj_info。
    # 分支条件：not t or t in _PRIMITIVES or t == 'void'。
    # 分支条件：'.' in t_no_arr。
    # 分支条件：t_no_arr in self.fqcn_to_file。
    # 分支条件：cand in self.fqcn_to_file。
    # 分支条件：imp.endswith('.*')。
    # 分支条件：cand2 in self.fqcn_to_file。
    # 分支条件：k in self.pkg_simple_to_fqcn。
    # 分支条件：imp.endswith('.' + simple) and imp in self.fqcn_to_file。
    # 分支条件：not imp.endswith('.*')。
    # 分支条件：len(cands) == 1。
    # 遍历边界：imp ← imports。
    # 返回路径：None。
    # 返回路径：t_no_arr。
    # 返回路径：cand。
    # 返回路径：cand2。
    # 返回路径：self.pkg_simple_to_fqcn[k]。
    # 返回路径：imp。
    # 返回路径：cands[0]。
    # 直接调用：_strip_type。
    # 直接调用：t.replace('[]', '').strip。
    # 直接调用：t.replace。
    # 直接调用：self.proj_info[cur_file].get。
    # 直接调用：mod.get。
    # 直接调用：imp.endswith。
    # 直接调用：self.simple_to_fqcns.get。
    # 直接调用：len。
    # 表达式分流：pkg；真值与假值分支分别计算，不同时执行。
    # [/maint-note:4a7b08ed1f31]
    def resolve_type(self, cur_file: str, type_name: str) -> Optional[str]:
        """
        Resolve a type string (possibly generic/array) to a project-internal FQCN if possible.
        """
        if not self.proj_info:
            return None

        t = _strip_type(type_name)
        if not t or t in _PRIMITIVES or t == "void":
            return None

        # remove array brackets
        t_no_arr = t.replace("[]", "").strip()

        mod = self.proj_info[cur_file].get("", {})
        pkg = mod.get("package") or ""
        imports: List[str] = mod.get("imports", []) or []

        # 1) If looks like FQCN
        if "." in t_no_arr:
            # direct match
            if t_no_arr in self.fqcn_to_file:
                return t_no_arr
            # could be Outer.Inner in same package or imported package:
            # try same package prefix
            cand = f"{pkg}.{t_no_arr}" if pkg else t_no_arr
            if cand in self.fqcn_to_file:
                return cand
            # try imported packages (wildcards)
            for imp in imports:
                if imp.endswith(".*"):
                    base = imp[:-2]
                    cand2 = f"{base}.{t_no_arr}"
                    if cand2 in self.fqcn_to_file:
                        return cand2
            # unknown external
            return None

        simple = t_no_arr

        # 2) Same package (implicit)
        k = (pkg, simple)
        if k in self.pkg_simple_to_fqcn:
            return self.pkg_simple_to_fqcn[k]

        # 3) Explicit imports
        for imp in imports:
            if imp.endswith(".*"):
                continue
            if imp.endswith("." + simple) and imp in self.fqcn_to_file:
                return imp

        # 4) Wildcard imports
        for imp in imports:
            if not imp.endswith(".*"):
                continue
            base = imp[:-2]
            cand = f"{base}.{simple}"
            if cand in self.fqcn_to_file:
                return cand

        # 5) java.lang implicit (treat as external unless you indexed JDK sources)
        # If you do index java.lang sources, it will resolve via fqcn_to_file
        cand = f"java.lang.{simple}"
        if cand in self.fqcn_to_file:
            return cand

        # 6) Unique simple name in project
        cands = self.simple_to_fqcns.get(simple, [])
        if len(cands) == 1:
            return cands[0]

        return None

    # [maint-note:dcd560e56dd2]
    # 维护导航：type_to_file；下列为本函数源码中的接口与路径索引。
    # 输入 fqcn：str。
    # 返回契约：Optional[str]。
    # 返回路径：self.fqcn_to_file.get(fqcn)。
    # 直接调用：self.fqcn_to_file.get。
    # [/maint-note:dcd560e56dd2]
    def type_to_file(self, fqcn: str) -> Optional[str]:
        return self.fqcn_to_file.get(fqcn)

    # [maint-note:4d151eaa62a1]
    # 维护导航：resolve_import_to_file；下列为本函数源码中的接口与路径索引。
    # 输入 cur_file：str。
    # 输入 import_path：str。
    # 返回契约：Optional[str]。
    # 分支条件：import_path.startswith(self.std_prefix)。
    # 分支条件：import_path.endswith('.*')。
    # 分支条件：fqcn.startswith(base + '.')。
    # 遍历边界：(fqcn, fpath) ← self.fqcn_to_file.items()。
    # 返回路径：None。
    # 返回路径：fpath。
    # 返回路径：self.fqcn_to_file.get(import_path)。
    # 直接调用：import_path.startswith。
    # 直接调用：import_path.endswith。
    # 直接调用：self.fqcn_to_file.items。
    # 直接调用：fqcn.startswith。
    # 直接调用：self.fqcn_to_file.get。
    # [/maint-note:4d151eaa62a1]
    def resolve_import_to_file(self, cur_file: str, import_path: str) -> Optional[str]:
        """
        import_path like com.a.Foo or com.a.*
        Return a representative file path if resolvable (for include edges).
        """
        if import_path.startswith(self.std_prefix):
            return None

        if import_path.endswith(".*"):
            base = import_path[:-2]
            # pick any one type from that package
            for fqcn, fpath in self.fqcn_to_file.items():
                if fqcn.startswith(base + "."):
                    return fpath
            return None

        return self.fqcn_to_file.get(import_path)

    # [maint-note:0dd2762d5912]
    # 维护导航：resolve_static_import_to_file；下列为本函数源码中的接口与路径索引。
    # 输入 import_path：str。
    # 返回契约：Optional[str]。
    # 分支条件：not import_path or import_path.startswith(self.std_prefix)。
    # 分支条件：not owner。
    # 分支条件：owner in self.fqcn_to_file。
    # 返回路径：None。
    # 返回路径：self.fqcn_to_file[owner]。
    # 直接调用：import_path.startswith。
    # 直接调用：import_path.endswith。
    # 直接调用：import_path.rsplit。
    # 表达式分流：import_path.endswith('.*')；真值与假值分支分别计算，不同时执行。
    # [/maint-note:0dd2762d5912]
    def resolve_static_import_to_file(self, import_path: str) -> Optional[str]:
        """
        Resolve static import owner class to a project file.
        Examples:
        - com.a.Util.CONST  -> file of com.a.Util
        - com.a.Util.*      -> file of com.a.Util
        """
        if not import_path or import_path.startswith(self.std_prefix):
            return None

        owner = import_path[:-2] if import_path.endswith(".*") else import_path.rsplit(".", 1)[0]
        if not owner:
            return None

        if owner in self.fqcn_to_file:
            return self.fqcn_to_file[owner]

        return None
