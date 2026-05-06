import torch
import numpy as np
from loguru import logger
from torch import Tensor

from typing import Tuple
from typing_extensions import Self

def _print_path_info(path_info):
    """打印路径信息"""
    # print("=" * 50)
    # print("路径信息分析")
    # print("=" * 50)
    
    # 检查所有可用的属性
    # print("可用的属性:")
    for attr in dir(path_info):
        if not attr.startswith('_'):
            value = getattr(path_info, attr)
            logger.info(f"  {attr}: {value}")
    
    # 特别检查常见的属性名
    common_attrs = ['largest_intermediate', 'opt_cost', 'contraction_list', 'path']
    for attr in common_attrs:
        if hasattr(path_info, attr):
            value = getattr(path_info, attr)
            logger.info(f"{attr}: {value}")
    
    # print("=" * 50)

def calculate_tensor_memory(tensor: Tensor):
    """计算单个张量的显存占用"""
    if tensor is None:
        return 0
    return (tensor.element_size() * tensor.numel())/2**30

# XXX: how to implement the MemoryTrack?
# ref: https://github.com/huangpan2507/Tools_Pytorch-Memory-Utils
class MemoryTrack:
    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device

        self.before_memory: float = 0.0
        self.after_memory: float = 0.0
        self.before_max_memory: float = 0.0
        self.after_max_memory: float = 0.0

    def __enter__(self) -> Self:
        self.clean_memory_cache(self.device)
        self.before_max_memory = self.get_max_memory(self.device)
        self.before_memory = self.get_current_memory(self.device)
        s = f"{self.device} memory allocated: {self.before_memory:.5f} GiB"
        # sys.stdout.write(s)
        logger.info(s, master=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for i in (exc_type, exc_val, exc_tb):
            if i is not None:
                raise RuntimeError
        self.after_max_memory = self.get_max_memory(self.device)
        self.clean_memory_cache(self.device)
        self.after_memory = self.get_current_memory(self.device)
        s = f"{self.device} memory allocated: {self.after_memory:.5f} GiB, "
        s += f"using memory: {(self.after_max_memory-self.before_memory):.5f} GiB"
        logger.info(s, master=True)
        # sys.stdout.write(s)

    def manually_clean_cache(self, objs: Tuple[Tensor] = None) -> None:
        if objs is not None:
            for obj in objs:
                if isinstance(obj, (Tensor,)):
                    del obj
        # gc.collect() # affect efficiency, worse or better?
        self.clean_memory_cache(self.device)

    @staticmethod
    def get_max_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.max_memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def get_current_memory(device: torch.device) -> float:
        n = 0.0
        if device.type == "cuda":
            n = torch.cuda.memory_allocated(device) / 2**30  # GiB
        return n

    @staticmethod
    def clean_memory_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()
