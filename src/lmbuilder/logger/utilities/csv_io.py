import csv
import ast
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional, Mapping, Tuple
from lmbuilder.logger.utilities.fs_io import get_filesystem, _is_dir
from warnings import warn
from argparse import Namespace

def load_hparams_from_tags_csv(tags_csv: Path) -> Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = os.path.join('.', 'testing-hparams.csv')
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)

    """
    fs = get_filesystem(tags_csv)
    if not fs.exists(tags_csv):
        warn(f"Missing Tags: {tags_csv}.", category=RuntimeWarning)
        return {}

    with fs.open(tags_csv, "r", newline="") as fp:
        csv_reader = csv.reader(fp, delimiter=",")
        return {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}


def save_hparams_to_csv(csv_file: Path, hparams: Union[dict, Namespace]) -> None:
    fs = get_filesystem(csv_file)
    if not _is_dir(fs, os.path.dirname(csv_file)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(csv_file)}.")

    with fs.open(csv_file, "w", newline="") as fp:
        fieldnames = ["key", "value"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in hparams.items():
            writer.writerow({"key": k, "value": v})

def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        return val