import os
import json
from typing import Dict, Union, Any, Optional, List
from pathlib import Path
from lmbuilder.logger.base import LMBuilderLogger
from overrides import override
from warnings import warn
from lmbuilder.logger.utilities.fs_io import get_filesystem
from lmbuilder.logger.utilities.params import _add_prefix, _convert_params, _sanitize_callable_params


class JSONLogger(LMBuilderLogger):
    
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
        self.messages_log_file = os.path.join(log_dir, "messages.json")
        self.message_dict = {}
        
        # weird way :)
        self.message_dict.setdefault("date", [])
        self.message_dict.setdefault("time", [])
        self.message_dict.setdefault("level", [])
        self.message_dict.setdefault("message", [])

        # File for logging metrics
        self.metrics_log_file = os.path.join(log_dir, "metrics.json")
        self.metrics: Dict[str, List[Union[str, int]]] = {}

        # File for logging hyperparameters
        self.hparams_log_file = os.path.join(log_dir, "hyperparameters.json")

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, self.name, version)

    def log_msg(self, message: str, level="info") -> None:
        date, time = self.get_curr_date_time()
        self.message_dict.get("date").append(date)
        self.message_dict.get("time").append(time)
        self.message_dict.get("level").append(level.upper())
        self.message_dict.get("message").append(message)
        self._write_to_json(self.messages_log_file, self.message_dict)

    def log_metrics(self, metrics_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        
        metrics_dict = _add_prefix(metrics_dict, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is None:
            step = len(self.metrics.get("steps", [])) #assuming that metrics are recorded sequentially.
        
        self.metrics.setdefault("steps", []).append(int(step))
        for key, value in metrics_dict.items():
            self.metrics.setdefault(key, {}).update({int(step): self._handle_value(value)})
        
        #print(f"Metrics at Step {step}: {self.metrics}")

        if step % self.flush_interval == 0:
            self.save()

    def log_hparams(self, params: Dict[str, Any]) -> None:
        params = _convert_params(params)
        clean_params = _sanitize_callable_params(params)
        self._write_to_json(self.hparams_log_file, clean_params)

    def save(self) -> None:
        
        file_exist = self._fs.isfile(self.metrics_log_file)

        if file_exist:
            with self._fs.open(self.metrics_log_file, "r") as file:
                prev_content = dict(json.load(file))
                print(f"\nMetric Dict Before: {prev_content}\n", flush=True)
                print(f"\nMetrics Now: {self.metrics}\n", flush=True)
                #print(list(self.metrics.values())[0])

            content_keys =  list(prev_content.keys())
            for k, v in self.metrics.items():
                if k in content_keys:
                    vl = prev_content[k]
                    if isinstance(vl, list):
                        prev_content.get(k).extend(v)
                    elif isinstance(vl, dict):
                        prev_content.get(k).update(v)
                else:
                    prev_content.setdefault(k, {}).update(v)
                
            print(f"Metric Dict After: {prev_content}", flush=True)

            self._write_to_json(self.metrics_log_file, prev_content)
        else:
            self._write_to_json(self.metrics_log_file, self.metrics)

        self.metrics = {}

    def _write_to_json(self, file_path: str, content: dict):
        with self._fs.open(file_path, mode="w") as json_file:
            json.dump(content, json_file, indent=4)

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not self._is_dir(versions_root, strict=True):
            warn("Missing logger folder: %s", versions_root, category=RuntimeError)
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if self._is_dir(full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
    
