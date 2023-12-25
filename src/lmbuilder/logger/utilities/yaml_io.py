import yaml
from pathlib import Path
import contextlib
import os
from warnings import warn
from typing import Union, Dict, Any, Optional, Mapping, Tuple
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from lmbuilder.logger.utilities.fs_io import get_filesystem, _is_dir
from lmbuilder.utils import is_package_available

_OMEGACONF_AVAILABLE = is_package_available("omegaconf")

def load_hparams_from_yaml(config_yaml: Path, use_omegaconf: bool = True) -> Dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
                the hparams will be converted to ``DictConfig`` if possible.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)

    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.full_load(fp)

    if _OMEGACONF_AVAILABLE and use_omegaconf:
        from omegaconf import OmegaConf
        from omegaconf.errors import UnsupportedValueType, ValidationError

        with contextlib.suppress(UnsupportedValueType, ValidationError):
            return OmegaConf.create(hparams)
    return hparams


def save_hparams_to_yaml(config_yaml: Path, hparams: Union[dict, Namespace], use_omegaconf: bool = True) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
        use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
            the hparams will be converted to ``DictConfig`` if possible.

    """
    fs = get_filesystem(config_yaml)
    if not _is_dir(fs, os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, Dict[str, Any]):
        hparams = dict(hparams)

    # saving with OmegaConf objects
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        from omegaconf import OmegaConf
        from omegaconf.dictconfig import DictConfig
        from omegaconf.errors import UnsupportedValueType, ValidationError

        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        #hparams = apply_to_collection(hparams, DictConfig, OmegaConf.to_container, resolve=True)
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except TypeError:
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)
