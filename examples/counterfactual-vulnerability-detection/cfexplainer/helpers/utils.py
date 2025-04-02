"""Set up project paths."""
import hashlib
import inspect
import os
import random
import string
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def project_dir() -> Path:
    """Get project path."""
    return Path(__file__).parent.parent


def storage_dir() -> Path:
    """Get storage path."""
    storage = os.getenv("SINGSTORAGE")
    if storage:
        return Path(storage) / "storage"
    return Path(__file__).parent.parent / "storage"


def external_dir() -> Path:
    """Get storage external path."""
    path = storage_dir() / "external"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def interim_dir() -> Path:
    """Get storage interim path."""
    path = storage_dir() / "interim"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def processed_dir() -> Path:
    """Get storage processed path."""
    path = storage_dir() / "processed"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def outputs_dir() -> Path:
    """Get output path."""
    path = storage_dir() / "outputs"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def cache_dir() -> Path:
    """Get storage cache path."""
    path = storage_dir() / "cache"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )


def gitsha():
    """Get current git commit sha for reproducibility."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )


def gitmessage():
    """Get current git commit sha for reproducibility."""
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())


def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = True):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    singularity = os.getenv("SINGULARITY")
    if singularity != "true" and not force_shell:
        command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output


def watch_subprocess_cmd(command: str, force_shell: bool = True):
    """Run subprocess and monitor output. Used for debugging purposes."""
    singularity = os.getenv("SINGULARITY")
    if singularity != "true" and not force_shell:
        command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Poll process for new output until finished
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True


def genid():
    """Generate random string."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID


def hashstr(s):
    """Hash a string."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
