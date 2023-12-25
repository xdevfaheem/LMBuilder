import os
import csv
from lmbuilder.logger.base import LMBuilderLogger
from typing import Dict, Union, Any, Optional, List, Set
from argparse import Namespace
from pathlib import Path
from overrides import override
from lmbuilder.logger.utilities.fs_io import get_filesystem, _is_dir
from lmbuilder.logger.utilities.params import _add_prefix, _convert_params, _sanitize_callable_params
from lmbuilder.logger.utilities.csv_io import save_hparams_to_csv
from lmbuilder.logger.utilities.colorize import colorize_log_message
from warnings import warn
from torch import Tensor

class CSVLogger(LMBuilderLogger):

    LOGGER_JOIN_CHAR = "."

    def __init__(
            self,
            root_dir: Path,
            name: str = "lmbuilder_logs",
            version: Optional[Union[int, str]] = None,
            prefix: str = "", 
            n_steps_to_log=100
        ):

        super().__init__()
        self._fs = get_filesystem(root_dir)
        self._root_dir = root_dir
        self._prefix = prefix
        self._name = name
        self._version = version
        self.flush_interval = n_steps_to_log
        log_dir = self.log_dir
        
        if self._fs.exists(log_dir) and self._fs.listdir(log_dir):
            warn(
                f"Experiment logs directory {log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!",
                category=RuntimeWarning
            )
            self._fs.delete(log_dir, recursive=True)
        self._fs.makedirs(log_dir, exist_ok=True)

        # File for logging messages
        self.messages_log_file = os.path.join(log_dir, "messages.csv")
        self.msg_file_fieldnames = ["date", "time", "level", "message"]
        self._initialize_csv_file(self.messages_log_file, self.msg_file_fieldnames)
        
        # File for logging metrics
        self.metrics_file_path = os.path.join(log_dir, "metrics.csv")
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        # File for logging hyperparametres
        self.hparams_log_file = os.path.join(log_dir, "hyperparameters.csv")
        self._initialize_csv_file(self.hparams_log_file, ["key", "value"])
        
    @property
    @override
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, self.name, version)
    
    def log_msg(self, message: str, level="info") -> None:
        """Log message to the file as it gets"""
        date, time = self.get_curr_date_time()
        # msg, llevel = colorize_log_message(message, log_level=level, bg_color=bg_color, bold=bold, light_fore=light_fore, light_bg=light_bg)
        log_dict = {"date": date, "time": time, "level": level.upper(), "message": message}
        self._write_to_csv(self.messages_log_file, self.msg_file_fieldnames, log_dict)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        metrics_dict = _add_prefix(metrics_dict, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is None:
            step = len(self.metrics)

        metrics = {k: self._handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

        print(f"Metrics at Step {step}: {self.metrics}")

        if (step) % self.flush_interval == 0:
            self.save()

    def log_hparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """Save the hyperparameters to a file as soon as it gets"""
        clean_params = _sanitize_callable_params(_convert_params(params))
        save_hparams_to_csv(self.hparams_log_file, hparams=clean_params)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists: # only write the header if we're writing a fresh file
                writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)
        self.metrics = []  # reset

    def _initialize_csv_file(self, file_path: str, field_names: List[str]):
        with self._fs.open(file_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

    def _write_to_csv(self, file_path: str, fileldnames, content: dict):
        with self._fs.open(file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fileldnames)
            writer.writerow(content)

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                print(metric)
                writer.writerow(metric)
        
    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not _is_dir(self._fs, versions_root, strict=True):
            warn("Missing logger folder: %s", versions_root, category=RuntimeError)
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1