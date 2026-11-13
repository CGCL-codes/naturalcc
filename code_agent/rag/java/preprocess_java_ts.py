import os
import time
import json

from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR
from java_project_parser_ts import JavaProjectParserTS


# [maint-note:059eed57a89a]
# 维护导航：load_dataset_pkgs；下列为本函数源码中的接口与路径索引。
# 输入 ds_file：str。
# 返回契约：set。
# 返回路径：pkg_set。
# 直接调用：open。
# 直接调用：json.loads。
# 直接调用：line.strip。
# 直接调用：set。
# 资源作用域：open(ds_file, 'r', encoding='utf-8', errors='ignore')；退出该 with 时调用对应上下文清理。
# 集合派生：line 来自 f；过滤 line.strip()。
# 集合派生：x 来自 ds；过滤 'pkg' in x。
# [/maint-note:059eed57a89a]
def load_dataset_pkgs(ds_file: str) -> set:
    with open(ds_file, "r", encoding="utf-8", errors="ignore") as f:
        ds = [json.loads(line) for line in f if line.strip()]
    pkg_set = set(x["pkg"] for x in ds if "pkg" in x)
    return pkg_set


# [maint-note:36a92ce71924]
# 维护导航：ensure_dir；下列为本函数源码中的接口与路径索引。
# 输入 path：str。
# 分支条件：not os.path.isdir(path)。
# 直接调用：os.path.isdir。
# 直接调用：os.makedirs。
# 直接调用：print。
# [/maint-note:36a92ce71924]
def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        print(f"📁 创建输出目录: {path}")


# [maint-note:aca4502e3f57]
# 维护导航：main；下列为本函数源码中的接口与路径索引。
# 分支条件：item not in pkg_set。
# 分支条件：not os.path.isdir(dir_path)。
# 分支条件：os.path.exists(output_path) and os.path.getsize(output_path) > 0。
# 遍历边界：item ← os.listdir(DS_REPO_DIR)。
# 捕获异常：Exception。
# 直接调用：time.perf_counter。
# 直接调用：load_dataset_pkgs。
# 直接调用：print。
# 直接调用：len。
# 直接调用：JavaProjectParserTS。
# 直接调用：ensure_dir。
# 直接调用：os.listdir。
# 直接调用：os.path.join。
# 直接调用：os.path.isdir。
# 直接调用：os.path.exists。
# 直接调用：os.path.getsize。
# 直接调用：project_parser.parse_dir。
# 直接调用：open。
# 直接调用：json.dump。
# 直接调用：f.startswith。
# 直接调用：os.path.isfile。
# 资源作用域：open(output_path, 'w', encoding='utf-8')；退出该 with 时调用对应上下文清理。
# 集合派生：f 来自 os.listdir(DS_GRAPH_DIR)；过滤 not f.startswith('.') and os.path.isfile(os.path.join(DS_GRAPH_DIR, f)) and (os.path.getsize(os.path.join(DS_GRAPH_DIR, f)) > 0)。
# [/maint-note:aca4502e3f57]
def main():
    t_all0 = time.perf_counter()

    pkg_set = load_dataset_pkgs(DS_FILE)
    print(f"📚 数据集包含 {len(pkg_set)} 个仓库")

    project_parser = JavaProjectParserTS()

    ensure_dir(DS_GRAPH_DIR)

    print(f"\n🚀 开始解析仓库: {DS_REPO_DIR}\n")

    total = 0
    skipped = 0
    success = 0
    failed = 0

    # 遍历 DS_REPO_DIR 下的仓库目录
    for item in os.listdir(DS_REPO_DIR):
        if item not in pkg_set:
            continue

        dir_path = os.path.join(DS_REPO_DIR, item)
        if not os.path.isdir(dir_path):
            continue

        total += 1

        # ✅ 如果输出文件已存在，跳过
        output_path = os.path.join(DS_GRAPH_DIR, f"{item}.json")
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

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

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
        if not f.startswith(".")
        and os.path.isfile(os.path.join(DS_GRAPH_DIR, f))
        and os.path.getsize(os.path.join(DS_GRAPH_DIR, f)) > 0
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


if __name__ == "__main__":
    main()
