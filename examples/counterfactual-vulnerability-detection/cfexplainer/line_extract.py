import os
import pickle as pkl
from pathlib import Path
from helpers import utils
from data_pre import bigvul


def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """Get lines that are dependent on added lines.

    Example:
    df = bigvul()
    filepath_before = "storage/processed/bigvul/before/177775.c"
    filepath_after = "storage/processed/bigvul/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    """

    before_cache_name = "_".join(str(filepath_before).split("/")[-3:])
    before_cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(before_cache_name).stem
    before_graph = pkl.load(open(before_cachefp, "rb"))[0]
    
    after_cache_name = "_".join(str(filepath_after).split("/")[-3:])
    after_cachefp = utils.get_dir(utils.cache_dir() / "vul_graph_feat") / Path(after_cache_name).stem
    after_graph = pkl.load(open(after_cachefp, "rb"))[0]

    # Get nodes in graph corresponding to added lines
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # Get lines dependent on added lines in added graph
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # Filter by lines in before graph
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def helper(row):
    """Run get_dep_add_lines from dict.

    Example:
    df = bigvul()
    added = df[df.id==177775].added.item()
    removed = df[df.id==177775].removed.item()
    helper({"id":177775, "removed": removed, "added": added})
    """
    before_path = str(utils.processed_dir() / f"bigvul/before/{row['id']}.c")
    after_path = str(utils.processed_dir() / f"bigvul/after/{row['id']}.c")
    try:
        dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
    except Exception:
        dep_add_lines = []
    return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]


def get_dep_add_lines_bigvul(cache=True):
    """Cache dependent added lines for bigvul."""
    saved = utils.get_dir(utils.processed_dir() / "bigvul/eval") / "statement_labels.pkl"
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = bigvul()
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = utils.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
    return lines_dict


if __name__ == "__main__":
    get_dep_add_lines_bigvul()
