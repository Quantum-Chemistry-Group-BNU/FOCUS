# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
# motified by: Longfei Chang
import os
import sys
import tempfile
import h5py
from pathlib import Path
import numpy as np
import torch
import opt_einsum
from functools import lru_cache
from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor
from pyfocus.camps.utils.config import dtype_config
from opt_einsum import contract
from loguru import logger


STORAGE_BACKEND = 'h5py'

class Environ:
    def __init__(self, mps, mpo, domain=None, mps_save_mode: str = 'save'):
        # read的结果直接是在GPU的torch中
        # todo: real disk and other backend
        # todo: contract_one_site_multi_mpo could generalize contract_one_site,
        # we could unify them in the future.

        # idx indicates the exact position of L or R, like
        # L(idx-1) - mpo(idx) - R(idx+1)
        self.scratch_dir_path = dtype_config.scratch_dir
        self.mps_save_mode = mps_save_mode
        if self.mps_save_mode == 'save':
            length = mps.get_system_size()
        elif self.mps_save_mode == 'normal':
            length = len(mps)
        self.length = length
        self.storage_backend = STORAGE_BACKEND.lower()
        assert self.storage_backend in ('h5py', 'memory')
        self._setup_storage_backend()
        self._virtual_disk = {}
        self.sentinel = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
        self._construct(mps, mpo, domain)
        info = self.get_storage_info()
        size = info['size']
        s = f"Use {self.storage_backend}, size: {size} "
        if self.storage_backend == 'h5py':
            s += f"Temporary HDF5 file: {self._temp_file.name}"
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def _setup_storage_backend(self):
        if self.storage_backend == "h5py":
            if self.scratch_dir_path is None:
                scratch_dir = Path("scratch")
                scratch_dir.mkdir(exist_ok=True)
            else:
                scratch_dir = Path(self.scratch_dir_path)
            self._temp_file = tempfile.NamedTemporaryFile(delete=False,
                                                        dir=scratch_dir,
                                                        suffix='.h5')
            self._temp_file.close()
            self.h5_file = h5py.File(self._temp_file.name, 'w')
            self._storage = self.h5_file
        elif self.storage_backend == "memory":
            self._storage = {}

    def __del__(self):
        if self.storage_backend == "h5py":
            if hasattr(self, 'h5_file') and self.h5_file:
                self.h5_file.close()
            if hasattr(self, '_temp_file') and os.path.exists(self._temp_file.name):
                try:
                    os.unlink(self._temp_file.name)
                except:
                    pass
            sys.stdout.write(f"del temprary HDF5 file {self._temp_file.name}" + "\n")
            sys.stdout.flush()

    def _get_h5_dataset_path(self, domain, siteidx):
        return f"{domain}/{siteidx}"

    def _construct(self, mps, mpo, domain=None):
        assert domain in ["R", "L", None]
        if domain is None:
            self._construct(mps, mpo, "R")
            self._construct(mps, mpo, "L")
            return
        
        if domain == "L":
            start, end, inc = 0, self.length - 1, 1
        else:
            start, end, inc = self.length - 1, 0, -1
        self.write_l_sentinel(self.length)
        self.write_r_sentinel(self.length)

        tensor = self.sentinel
        for idx in range(start, end, inc):
            # one single mpo
            if self.mps_save_mode == 'save':
                mps_i = torch.from_numpy(mps.read(idx)).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo.read(idx)).to(dtype_config.default_dtype).to(dtype_config.device)
            elif self.mps_save_mode == 'normal':
                mps_i = torch.from_numpy(mps[idx]).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo[idx]).to(dtype_config.default_dtype).to(dtype_config.device)
            with torch.no_grad():
                tensor = contract_one_site(tensor, mps_i, mpo_i, domain)
                tensor = tensor.detach()
            del mps_i, mpo_i
            self.write(domain, idx, tensor)
        del tensor
        torch.cuda.empty_cache()

    def write_l_sentinel(self, length):
        self.write("L", -1, self.sentinel)

    def write_r_sentinel(self, length):
        self.write("R", length, self.sentinel)

    def GetLR(
        self, domain, siteidx, mps, mpo, itensor=None, method="Scratch"):
        """
        get the L/R Hamiltonian matrix at a random site(siteidx): 3d tensor
        S-     -S     mpsconj
        O- or  -O     mpo
        S-     -S     mps
        enviroment part from self.virtual_disk,  system part from one step calculation
        support from scratch calculation: from two open boundary np.ones((1,1,1))
        """

        assert domain in ["L", "R"]
        assert method in ["Enviro", "System", "Scratch"]
        
        if siteidx not in range(self.length):
            return self.sentinel

        if method == "Scratch":
            itensor = self.sentinel
            if domain == "L":
                sitelist = range(siteidx + 1)
            else:
                sitelist = range(self.length - 1, siteidx - 1, -1)
            for imps in sitelist:
                if self.mps_save_mode == 'save':
                    mps_i = torch.from_numpy(mps.read(imps)).to(dtype_config.default_dtype).to(dtype_config.device)
                    mpo_i = torch.from_numpy(mpo.read(imps)).to(dtype_config.default_dtype).to(dtype_config.device)
                elif self.mps_save_mode == 'normal':
                    mps_i = torch.from_numpy(mps[imps]).to(dtype_config.default_dtype).to(dtype_config.device)
                    mpo_i = torch.from_numpy(mpo[imps]).to(dtype_config.default_dtype).to(dtype_config.device)
                itensor = contract_one_site(itensor, mps_i, mpo_i, domain)
        elif method == "Enviro":
            itensor = self.read(domain, siteidx)
        elif method == "System":
            if itensor is None:
                offset = -1 if domain == "L" else 1
                itensor = self.read(domain, siteidx + offset)  # 需要优化
            if self.mps_save_mode == 'save':
                mps_i = torch.from_numpy(mps.read(siteidx)).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo.read(siteidx)).to(dtype_config.default_dtype).to(dtype_config.device)
            elif self.mps_save_mode == 'normal':
                mps_i = torch.from_numpy(mps[siteidx]).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo[siteidx]).to(dtype_config.default_dtype).to(dtype_config.device)
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     start_event = torch.cuda.Event(enable_timing=True)
            #     end_event = torch.cuda.Event(enable_timing=True)
            #     start_event.record()
            itensor = contract_one_site(itensor, mps_i, mpo_i, domain)
            # if torch.cuda.is_available():
            #     end_event.record()
            #     torch.cuda.synchronize()  # 等待CUDA操作完成
            #     elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            #     logger.info(f'[time needed for calculating env tensor]: {elapsed_time}.')
            self.write(domain, siteidx, itensor)
        return itensor

    def write(self, domain, siteidx, tensor):  # 注意类型转换,转到CPU上进行存储
        key = (domain, siteidx)
        if self.storage_backend == "h5py":
            dataset_path = self._get_h5_dataset_path(domain, siteidx)
            if dataset_path in self.h5_file:
                del self.h5_file[dataset_path]
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     start_event = torch.cuda.Event(enable_timing=True)
            #     end_event = torch.cuda.Event(enable_timing=True)
            #     start_event.record()
            self.h5_file.create_dataset(dataset_path, data=tensor.detach().to('cpu').numpy())
            # if torch.cuda.is_available():
            #     end_event.record()
            #     torch.cuda.synchronize()  # 等待CUDA操作完成
            #     elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            #     logger.info(f'[time needed for writting env tensor]: {elapsed_time}.')
        elif self.storage_backend == "memory": # 直接存储在GPU中
            self._storage[key] = tensor.detach()

    def read(self, domain, siteidx):
        key = (domain, siteidx)
        if self.storage_backend == "h5py":
            dataset_path = self._get_h5_dataset_path(domain, siteidx)
            if dataset_path not in self.h5_file:
                return self.sentinel
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     start_event = torch.cuda.Event(enable_timing=True)
            #     end_event = torch.cuda.Event(enable_timing=True)
            #     start_event.record()
            data = torch.from_numpy(self.h5_file[dataset_path][()]).to(dtype_config.device)
            # if torch.cuda.is_available():
            #     end_event.record()
            #     torch.cuda.synchronize()  # 等待CUDA操作完成
            #     elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            #     logger.info(f'[time needed for reading env tensor]: {elapsed_time}.')
            return data
        elif self.storage_backend == "memory":
            if key not in self._storage:
                return self.sentinel
            return self._storage[key]

    def keys(self):
        if self.storage_backend == "h5py":
            keys = []
            for domain in self.h5_file.keys():
                if domain in ['L', 'R']:
                    for siteidx in self.h5_file[domain].keys():
                        keys.append((domain, int(siteidx)))
            return keys
        elif self.storage_backend == "memory":
            return list(self._storage.keys())

    def clear(self):
        if self.storage_backend == "h5py":
            self.h5_file.close()
            os.unlink(self._temp_file.name)
            self._setup_storage_backend()
        elif self.storage_backend == "memory":
            self._storage.clear()

    def get_storage_info(self):
        if self.storage_backend == "h5py":
            size = os.path.getsize(self._temp_file.name) if os.path.exists(self._temp_file.name) else 0
            size = size / (1024 * 1024) # MiB
            return {
                "backend": "h5py",
                "file_path": self._temp_file.name,
                "num_items": len(self.keys()),
                "size": f"{size:.3f} MiB"
            }
        elif self.storage_backend == "memory":
            size = sum(arr.nbytes for arr in self._storage.values()) if self._storage else 0
            size = size / (1024 * 1024) # MiB
            return {
                "backend": "memory",
                "num_items": len(self._storage),
                "size": f"{size:.3f} MiB"
            }

# def env_L_i(mps_i, mpo_i, env_Li):
#     """
#         S-a-S-f
#             d  
#         O-b-O-g
#             e  
#         S-c-S-h
#     """
#     env_L_new = contract('abc, adf, bdeg, ceh -> fgh', 
#                         env_Li, mps_i.conj(), mpo_i, mps_i)
#     return env_L_new

# def env_R_i(mps_i, mpo_i, env_Ri):
#     """
#         -f-S-a-S
#            d    
#         -g-O-b-O
#            e    
#         -h-S-c-S
#     """
#     # print(mps_i.dtype, env_Ri.dtype, mpo_i.dtype)
#     env_R_new = contract('fda, abc, gdeb, hec -> fgh', 
#                         mps_i.conj(), env_Ri, mpo_i, mps_i)
#     return env_R_new

def _gib(numel: float, itemsize: int) -> float:
    return (float(numel) * itemsize) / (2 ** 30)


def _parse_eq(eq: str):
    lhs, out = eq.split("->")
    inputs = [s.strip() for s in lhs.split(",")]
    return inputs, out.strip()



@lru_cache(maxsize=512)
def _cached_expr(eq: str, shapes: Tuple[Tuple[int, ...], ...]):
    # build expression with shapes only; backend is torch at call-time
    return opt_einsum.contract_expression(eq, *shapes)


def _slice_on_char(t: torch.Tensor, subs: str, ch: str, start: int, end: int) -> torch.Tensor:
    if ch not in subs:
        return t
    sl = [slice(None)] * t.ndim
    for axis, c in enumerate(subs):
        if c == ch:
            sl[axis] = slice(start, end)
    return t[tuple(sl)]


def batched_contract_auto(
    eq: str,
    *arrays: torch.Tensor,
    max_memory_gib: float = 0.5,
    optimize="auto",
    safety: float = 1.05,
    batch_char: Optional[str] = None,   # None -> auto pick
):
    """
    Contract with batching to keep opt_einsum's largest_intermediate under budget.
    Returns: output tensor
    """
    eq = eq.replace(" ", "")
    arrays = tuple(arrays)
    shapes = tuple(tuple(int(x) for x in a.shape) for a in arrays)
    inputs, out_sub = _parse_eq(eq)

    # dtype / itemsize used for memory estimate
    dt = arrays[0].dtype
    for a in arrays[1:]:
        dt = torch.promote_types(dt, a.dtype)
    itemsize = torch.empty((), dtype=dt).element_size()

    # baseline path info
    _, info = opt_einsum.contract_path(eq, *shapes, shapes=True, optimize=optimize)
    base_gib = _gib(info.largest_intermediate, itemsize)

    # if already within budget -> direct contract
    if max_memory_gib == -1 or base_gib <= max_memory_gib:
        expr = _cached_expr(eq, shapes)
        return expr(*arrays, backend="torch")

    size_dict: Dict[str, int] = dict(info.size_dict)

    # candidates: indices NOT in output, appear in at least one input
    contracted = [ch for ch in size_dict.keys() if ch not in set(out_sub)]
    if not contracted:
        # nothing to batch on that's contracted -> fallback to output batching on first output dim
        contracted = [out_sub[0]]

    # auto pick batch_char: prefer large dim and appears multiple times
    if batch_char is None:
        def score(ch: str) -> int:
            occ = sum(ch in s for s in inputs)
            return size_dict[ch] * occ
        batch_char = max(contracted, key=score)

    if batch_char not in size_dict:
        raise ValueError(f"batch_char='{batch_char}' not in equation indices.")

    dim = int(size_dict[batch_char])

    # initial min_batch guess: assume roughly linear scaling wrt that dim
    n = int(np.ceil((base_gib / max_memory_gib) * safety))
    n = max(1, n)
    min_batch = int(np.ceil(dim / n))
    min_batch = max(1, min(min_batch, dim))

    # refine by recomputing largest_intermediate with modified shapes, shrink if needed
    def shapes_with(bs: int):
        sm = logger  # avoid lint
        shapes_mod = [list(s) for s in shapes]
        for i, subs in enumerate(inputs):
            for axis, c in enumerate(subs):
                if c == batch_char:
                    shapes_mod[i][axis] = bs
        return tuple(tuple(s) for s in shapes_mod)

    shapes_mod = shapes_with(min_batch)
    _, info_b = opt_einsum.contract_path(eq, *shapes_mod, shapes=True, optimize=optimize)
    batched_gib = _gib(info_b.largest_intermediate, itemsize)

    # shrink loop (fast, few iterations)
    while batched_gib > max_memory_gib and min_batch > 1:
        min_batch = max(1, min_batch // 2)
        shapes_mod = shapes_with(min_batch)
        _, info_b = opt_einsum.contract_path(eq, *shapes_mod, shapes=True, optimize=optimize)
        batched_gib = _gib(info_b.largest_intermediate, itemsize)
    logger.info(
        f"[compute env tensors] largest intermediate memory of env tensor: {base_gib:.3f} GiB -> {batched_gib:.3f} GiB."
    )

    # build expr for full batches and tail batches
    expr_full = _cached_expr(eq, shapes_mod)

    # output buffer
    out_shape = tuple(int(size_dict[c]) for c in out_sub)
    out = torch.zeros(out_shape, dtype=dt, device=arrays[0].device)

    # batching logic: if batch_char in output, write slice; else accumulate
    out_axis = out_sub.find(batch_char)
    idx = [0] + (np.arange(0, dim, min_batch) + min_batch).tolist()
    idx[-1] = dim

    for start, end in zip(idx[:-1], idx[1:]):
        bs = end - start
        sliced = tuple(_slice_on_char(a, inputs[i], batch_char, start, end) for i, a in enumerate(arrays))
        if bs == min_batch:
            val = expr_full(*sliced, backend="torch")
        else:
            shapes_tail = shapes_with(bs)
            expr_tail = _cached_expr(eq, shapes_tail)
            val = expr_tail(*sliced, backend="torch")

        if out_axis < 0:
            out += val
        else:
            sl = [slice(None)] * out.ndim
            sl[out_axis] = slice(start, end)
            out[tuple(sl)] = val

    return out


# --------- your two env builders (batched) ---------
def env_L_i(mps_i, mpo_i, env_Li, max_memory_gib=2, batch_char='g', optimize="auto"):
    eq = "abc, adf, bdeg, ceh -> fgh"
    return batched_contract_auto(
        eq,
        env_Li, mps_i.conj(), mpo_i, mps_i,
        max_memory_gib=max_memory_gib,
        batch_char=batch_char,   # None -> auto choose among contracted indices
        optimize=optimize,
    )

def env_R_i(mps_i, mpo_i, env_Ri, max_memory_gib=2, batch_char='g', optimize="auto"):
    eq = "fda, abc, gdeb, hec -> fgh"
    return batched_contract_auto(
        eq,
        mps_i.conj(), env_Ri, mpo_i, mps_i,
        max_memory_gib=max_memory_gib,
        batch_char=batch_char,  # 拆分mpo
        optimize=optimize,
    )

def get_envs_L(mps: List[Tensor], 
               mpo: Union[List[Tensor], List[List[Tensor]]], 
               method: str, 
               mpo_type: str = 'S'):
    if method == '1site':
        remain = 1
    elif method == '2site':
        remain = 2
    else:
        assert False
    nsites = len(mps)
    envs_L_list = []
    # env_L_tmp = np.ones([1,]*3)
    if mpo_type == 'S':
        env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
        envs_L_list.append(env_L_tmp)
        for i in range(nsites-remain):
            env_L_tmp = env_L_i(mps[i], mpo[i], env_L_tmp)
            envs_L_list.append(env_L_tmp)
    elif mpo_type == 'M':
        M_MPO_len = len(mpo)
        for L in range(M_MPO_len):
            single_env_list = []
            env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
            single_env_list.append(env_L_tmp)
            for i in range(nsites-remain):
                env_L_tmp = env_L_i(mps[i], mpo[L][i], env_L_tmp)
                single_env_list.append(env_L_tmp)
            envs_L_list.append(single_env_list)
    return envs_L_list

def get_envs_R(mps: List[Tensor], 
               mpo: Union[List[Tensor], List[List[Tensor]]], 
               method: str, 
               mpo_type: str = 'S'):
    if method == '1site':
        remain = 1
    elif method == '2site':
        remain = 2
    else:
        assert False
    nsites = len(mps)
    envs_R_list = []
    if mpo_type == 'S':
        env_R_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
        envs_R_list.append(env_R_tmp)
        for i in range(nsites-1, remain - 1, -1):
            env_R_tmp = env_R_i(mps[i], mpo[i], env_R_tmp)
            envs_R_list.append(env_R_tmp)
    elif mpo_type == 'M':
        M_MPO_len = len(mpo)
        for L in range(M_MPO_len):
            single_env_list = []
            env_R_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
            single_env_list.append(env_R_tmp)
            for i in range(nsites-1, remain - 1, -1):
                env_R_tmp = env_R_i(mps[i], mpo[L][i], env_R_tmp)
                single_env_list.append(env_R_tmp)
            envs_R_list.append(single_env_list)
    return envs_R_list

def contract_one_site(environ, ms, mo, domain):
    """
    contract one mpo/mps(mpdm) site
             _   _
            | | | |
    S-S-    | S-|-S-
    O-O- or | O-|-O- (the ancillary bond is traced)
    S-S-    | S-|-S-
            |_| |_|
    """
    assert domain in ["L", "R"]

    if domain == "L":
        assert environ.shape[0] == ms.shape[0]
        assert environ.shape[1] == mo.shape[0]
        assert environ.shape[2] == ms.shape[0]
        """
        S-a-S-f
            d  
        O-b-O-g
            e   
        S-c-S-h
        """
        outtensor = env_L_i(ms, mo, environ)

    else:
        assert environ.shape[0] == ms.shape[-1]
        assert environ.shape[1] == mo.shape[-1]
        assert environ.shape[2] == ms.shape[-1]
        """
        -f-S-a-S
           d    
        -g-O-b-O
           e    
        -h-S-c-S
        """
        outtensor = env_R_i(ms, mo, environ)
    return outtensor
