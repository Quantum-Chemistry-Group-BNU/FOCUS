import uuid
import optree
import torch
import numpy as np

from typing import TypedDict
from pathlib import Path
from datetime import datetime
from loguru import logger

from camps.utils.typing import SaveInfo
from camps.utils.tools import random_str


class LoggerInfo(TypedDict):
    metadata: dict
    init_parameters: dict
    iterations: dict[int, SaveInfo]
    extra_tags: dict


class IterationSaver:
    def __init__(
        self,
        base_dir: str = "./tmp/",
        experiment_name: str = "exp",
        fixed_filename: str = None,
        no_random_subdir: bool = True  # 添加这个参数
    ):

        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.fixed_filename = fixed_filename or random_str()
        
        if no_random_subdir:
            # 直接使用 base_dir，不创建随机子文件夹
            self.save_dir = self.base_dir
            self.full_path = self.save_dir / f"{self.fixed_filename}.pth"
        else:
            # 原来的逻辑
            self.session_id = datetime.now().strftime("%y%m%d-%H%M") + f"-{uuid.uuid4().hex[:4]}"
            self.save_dir = self.base_dir / f"{self.experiment_name}-{self.session_id}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.full_path = (self.save_dir / self.fixed_filename).with_suffix(".pth")
        
        self.save_dir.mkdir(parents=True, exist_ok=True)  # 确保基础目录存在

        self.data = LoggerInfo(
            metadata=None,
            init_parameters=None,
            iterations={},
            extra_tags={},
        )
        self._initialize_metadata(no_random_subdir)

    def _initialize_metadata(self, no_random_subdir=False):
        if no_random_subdir:
            metadata = {
                "experiment_name": self.experiment_name,
                "fixed_filename": self.fixed_filename,
                "created_at": datetime.now().isoformat(),
            }
        else:
            metadata = {
                "experiment_name": self.experiment_name,
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
            }
        self.metadata = metadata
        self.data["metadata"] = metadata

    def save_initial_info(self, initial_data: dict):
        self.data["init_parameters"] = initial_data
        self._save_all_data(self.data)

    def _save_all_data(self, data: dict):
        self.use_memory = self.get_array_memory_optree(data)
        logger.info(f"Save data approximate memory: {self.use_memory:.3f} MiB")
        logger.info(f"Save mps/hamiltonian state: -> {self.full_path}")
        torch.save(data, self.full_path)

    def save_iteration_info(self, save_info: SaveInfo, iteration: int):
        self.data["iterations"][iteration] = {
            **save_info,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_all_data(self.data)

    def add_extra_tags(self, tags: dict):
        assert isinstance(tags, dict)
        for key, value in tags.items():
            self.data["extra_tags"][key] = value

    @staticmethod
    def get_array_memory_optree(data_dict: dict) -> float:
        leaves = optree.tree_flatten(data_dict)[0]
        return sum(p.nbytes / (1024**2) for p in leaves if isinstance(p, (np.ndarray, torch.Tensor)))


def load_logger(file: str, *args, **kwargs) -> LoggerInfo:
    """
    load file from logging-info
    """
    return torch.load(file, weights_only=False, *args, **kwargs)
