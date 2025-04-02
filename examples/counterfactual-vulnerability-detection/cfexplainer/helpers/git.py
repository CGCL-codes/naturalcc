import os
import pickle as pkl
import uuid
from multiprocessing import Pool

from . import utils
from tqdm import tqdm
from unidiff import PatchSet


def gitdiff(old: str, new: str):
    """Git diff between two strings."""
    cachedir = utils.cache_dir()
    oldfile = cachedir / uuid.uuid4().hex
    newfile = cachedir / uuid.uuid4().hex
    with open(oldfile, "w") as f:
        f.write(old)
    with open(newfile, "w") as f:
        f.write(new)
    cmd = " ".join(
        [
            "git",
            "diff",
            "--no-index",
            "--no-prefix",
            f"-U{len(old.splitlines()) + len(new.splitlines())}",
            str(oldfile),
            str(newfile),
        ]
    )
    process = utils.subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    r"""Get modified and deleted lines from Git patch.

    old = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
       asn1_push_tag(data, ASN1_GENERAL_STRING);\n\
       asn1_write_LDAPString(data, s);\n\
       asn1_pop_tag(data);\n\
       return !data->has_error;\n\
    }\n\
    \n\
    "

    new = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
        if (!asn1_push_tag(data, ASN1_GENERAL_STRING)) return false;\n\
        if (!asn1_write_LDAPString(data, s)) return false;\n\
        return asn1_pop_tag(data);\n\
    }\n\
    \n\
    int test() {\n\
        return 1;\n\
    }\n\
    "

    patch = gitdiff(old, new)
    """
    parsed_patch = PatchSet(patch)
    ret = {"added": [], "removed": [], "diff": ""}
    if len(parsed_patch) == 0:
        return ret
    parsed_file = parsed_patch[0]
    hunks = list(parsed_file)
    assert len(hunks) == 1
    hunk = hunks[0]
    ret["diff"] = str(hunk).split("\n", 1)[1]
    for idx, ad in enumerate([i for i in ret["diff"].splitlines()], start=1):
        if len(ad) > 0:
            ad = ad[0]
            if ad == "+" or ad == "-":
                ret["added" if ad == "+" else "removed"].append(idx)
    return ret


def code2diff(old: str, new: str):
    """Get added and removed lines from old and new string."""
    patch = gitdiff(old, new)
    return md_lines(patch)


def _c2dhelper(item):
    """Given item with func_before, func_after, id, and dataset, save gitdiff."""
    savedir = utils.get_dir(utils.cache_dir() / item["dataset"] / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if os.path.exists(savepath):
        return
    if item["func_before"] == item["func_after"]:
        return
    ret = code2diff(item["func_before"], item["func_after"])
    with open(savepath, "wb") as f:
        pkl.dump(ret, f)


def mp_code2diff(df):
    """Parallelise code2diff.

    Input DF must have columns: func_before, func_after, id, dataset
    """
    items = df[["func_before", "func_after", "id", "dataset"]].to_dict("records")
    with Pool(processes=6) as pool:
        for _ in tqdm(pool.imap_unordered(_c2dhelper, items), total=len(items)):
            pass


def get_codediff(dataset, iid):
    """Get codediff from file."""
    savedir = utils.get_dir(utils.cache_dir() / dataset / "gitdiff")
    savepath = savedir / f"{iid}.git.pkl"
    try:
        with open(savepath, "rb") as f:
            return pkl.load(f)
    except:
        return []


def allfunc(row):
    """Return a combined function (before + after commit) given the diff.

    diff = return raw diff of combined function
    added = return added line numbers relative to the combined function (start at 1)
    removed = return removed line numbers relative to the combined function (start at 1)
    before = return combined function, commented out added lines
    after = return combined function, commented out removed lines
    """
    readfile = get_codediff(row["dataset"], row["id"])

    ret = dict()
    ret["diff"] = "" if len(readfile) == 0 else readfile["diff"]
    ret["added"] = [] if len(readfile) == 0 else readfile["added"]
    ret["removed"] = [] if len(readfile) == 0 else readfile["removed"]
    ret["before"] = row["func_before"]
    ret["after"] = row["func_before"]

    if len(readfile) > 0:
        lines_before = []
        lines_after = []
        for li in ret["diff"].splitlines():
            if len(li) == 0:
                continue
            li_before = li
            li_after = li
            if li[0] == "-":
                li_before = li[1:]
                li_after = "// " + li[1:]
            if li[0] == "+":
                li_before = "// " + li[1:]
                li_after = li[1:]
            lines_before.append(li_before)
            lines_after.append(li_after)
        ret["before"] = "\n".join(lines_before)
        ret["after"] = "\n".join(lines_after)

    return ret
