import os
import re
import random
import numpy as np
import pandas as pd
from helpers import utils
from helpers import git
from sklearn.model_selection import train_test_split


def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def bigvul(minimal=True, sample=False, return_raw=False, splits="default"):
    """Read BigVul Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = utils.get_dir(utils.cache_dir() / "minimal_datasets")
    if minimal:
        try:
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            md = pd.read_csv(utils.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            default_splits = utils.external_dir() / "bigvul_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            return df
        except Exception as E:
            print(E)
            pass
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(utils.external_dir() / filename)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = utils.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = utils.dfmp(df, remove_comments, "func_after", cs=500)

    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    utils.dfmp(df, git._c2dhelper, columns=cols, ordr=False, cs=300)

    # Assign info and save
    df["info"] = utils.dfmp(df, git.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # POST PROCESSING
    dfv = df[df.vul == 1]
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # Remove functions with abnormal ending (no } or ;)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # Remove samples with mod_prop > 0.5
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # Filter by post-processing filtering
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # Make splits
    df = train_val_test_split_df(df, "id", "vul")

    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(utils.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df


if __name__ == "__main__":
    """Run preperation scripts for BigVul dataset."""
    bigvul()
