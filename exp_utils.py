from datetime import datetime
from pathlib import Path
import json
import os
import zipfile
import sys
import __main__

default_ignore_dirs = [
    "runs", "__pycache__", "chkpts", ".vscode", "figs", "wandb", "legacy",
    ".git", "data", "legacy"
]

default_ignore_exts = [".pth", ".png", ".pkl", ".npz", ".gz"]


def is_path_valid(path, ignoreDir, ignoreExt):
    splited = None
    if os.path.isfile(path):
        if ignoreExt:
            _, ext = os.path.splitext(path)
            if ext in ignoreExt:
                return False

        splited = os.path.dirname(path).split('\\/')
    else:
        if not ignoreDir:
            return True
        splited = path.split('\\/')

    if ignoreDir:
        for s in splited:
            if s in ignoreDir:  # You can also use set.intersection or [x for],
                return False

    return True


def zipDirHelper(path, rootDir, zf, ignoreDir=None, ignoreExt=None):
    # zf is zipfile handle
    if os.path.isfile(path):
        if is_path_valid(path, ignoreDir, ignoreExt):
            relative = os.path.relpath(path, rootDir)
            zf.write(path, relative)
        return

    ls = os.listdir(path)
    for subFileOrDir in ls:
        if not is_path_valid(subFileOrDir, ignoreDir, ignoreExt):
            continue
        joinedPath = os.path.join(path, subFileOrDir)
        zipDirHelper(joinedPath, rootDir, zf, ignoreDir, ignoreExt)


def ZipDir(path, zf, ignoreDir=None, ignoreExt=None, close=False):
    rootDir = path if os.path.isdir(path) else os.path.dirname(path)

    try:
        zipDirHelper(path, rootDir, zf, ignoreDir, ignoreExt)
    finally:
        if close:
            zf.close()


def save_tmp_src(results_dir: Path, ignorefile=".gitignore"):
    exts = []
    dirs = []
    if Path(ignorefile).is_file():
        lines = []
        with open('.gitignore', 'r') as f:
            for line in f:
                if len(line) <= 1:
                    continue
                if line[0] == "#":
                    continue
                elif line[-1] == '\n':
                    lines.append(line[:-1])
                else:
                    lines.append(line)

        for line in lines:
            if line[:2] == "*.":
                exts.append(line[2:])
            elif line[0] == "/":
                dirs.append(line[1:])
            else:
                dirs.append(line)
    else:
        if ignorefile != ".gitignore":
            raise FileExistsError("No such ignorefile.")
        exts = default_ignore_exts
        dirs = default_ignore_dirs

    with zipfile.ZipFile(results_dir / "src.zip", "w") as zf:
        ZipDir(Path("./").resolve(),
               zf,
               ignoreDir=[results_dir.parent.stem] + default_ignore_dirs,
               ignoreExt=exts,
               close=False)


def save_cmd(results_dir: Path):
    with open(results_dir / "cmd.txt", "w") as f:
        f.write("python " + " ".join(sys.argv))


def save_exp_config(results_dir: Path,
                    config: dict,
                    save_src=True,
                    save_command=True):
    """
    Save experiment config.

    Args:
        results_dir: Path. The path in which the experiment results and source are stored.
        config: dict. experiment configuration object
        save_src: bool. Pack and save source code in the results dir if True.
        save_command: bool. Save the calling command to the results dir if True.
    """
    if not results_dir.is_dir():
        raise FileExistsError("No dir for saving configs.")
    json.dump(config, open(results_dir / "config.json", "w"))

    if save_src:
        save_tmp_src(results_dir)
    if save_command:
        save_cmd(results_dir)


def gen_exp_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_result_dir(result_parent="results"):
    result_parent_dir = Path(result_parent)
    result_parent_dir.mkdir(exist_ok=True)
    exp_id = gen_exp_id()
    app_name = os.path.basename(__main__.__file__).strip(".py")
    result_dir = result_parent_dir / f"{exp_id}-{app_name}"
    result_dir.mkdir()
    return result_dir, exp_id
