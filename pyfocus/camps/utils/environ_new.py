# -*- coding: utf-8 -*-
# A safer + faster Environ for "h5py scratch" with:
# - single h5py.File handle (opened once in 'w')
# - a global lock to make read/write thread-safe (so you can prefetch in a background thread)
# - read_direct() into (reusable) pinned CPU buffers to avoid huge NumPy allocations
# - optional non_blocking H2D copy to overlap IO + GPU compute

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch

from pyfocus.camps.utils.config import dtype_config

STORAGE_BACKEND = "h5py"


def _np_dtype_to_torch(np_dtype: np.dtype) -> torch.dtype:
    # extend if needed
    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.float64:
        return torch.float64
    if np_dtype == np.complex64:
        return torch.complex64
    if np_dtype == np.complex128:
        return torch.complex128
    if np_dtype == np.int32:
        return torch.int32
    if np_dtype == np.int64:
        return torch.int64
    if np_dtype == np.uint8:
        return torch.uint8
    # fall back to your default dtype
    return dtype_config.default_dtype


class Environ:
    """
    Environment storage with two backends:
      - "h5py": store tensors in a scratch temporary HDF5 file (CPU), read back when needed
      - "memory": store tensors directly (GPU) in a dict

    Improvements vs your version:
      1) self._h5_lock guards ALL HDF5 ops -> safe to use background-thread prefetch.
      2) read_cpu_pinned(): uses HDF5 read_direct into pinned torch buffers (no huge numpy alloc).
      3) read_gpu_async(): pinned CPU -> GPU in a separate CUDA stream (non_blocking).
    """

    def __init__(self, mps, mpo, domain=None, mps_save_mode: str = "save"):
        self.mps_save_mode = mps_save_mode
        if self.mps_save_mode == "save":
            self.length = mps.get_system_size()
        elif self.mps_save_mode == "normal":
            self.length = len(mps)
        else:
            raise ValueError(f"Unknown mps_save_mode={mps_save_mode}")

        self.storage_backend = STORAGE_BACKEND.lower()
        assert self.storage_backend in ("h5py", "memory")

        self._setup_storage_backend()
        self._virtual_disk = {}  # unused but kept for compat
        self._pinned_pool: Dict[Tuple[str, int], torch.Tensor] = {}  # (domain, siteidx)-> pinned buf
        self._copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.sentinel = torch.ones([1] * 3, dtype=dtype_config.default_dtype, device=dtype_config.device)
        self._construct(mps, mpo, domain)

        info = self.get_storage_info()
        s = f"Use {info['backend']}, size: {info['size']} "
        if info["backend"] == "h5py":
            s += f"Temporary HDF5 file: {info['file_path']}"
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    # ----------------- storage setup / teardown -----------------

    def _setup_storage_backend(self):
        if self.storage_backend == "h5py":
            scratch_dir = Path("scratch")
            scratch_dir.mkdir(exist_ok=True)

            self._temp_file = tempfile.NamedTemporaryFile(delete=False, dir=scratch_dir, suffix=".h5")
            self._temp_file.close()

            # Single writer handle kept open; do NOT open a second handle in other threads.
            self.h5_file = h5py.File(self._temp_file.name, "w")
            self._h5_lock = threading.Lock()  # critical: protect all HDF5 ops
            self._storage = self.h5_file
        else:
            self._storage = {}

    def close(self):
        # explicit close is safer than relying on __del__
        if self.storage_backend == "h5py":
            try:
                if hasattr(self, "h5_file") and self.h5_file:
                    with self._h5_lock:
                        try:
                            self.h5_file.flush()
                        except Exception:
                            pass
                        self.h5_file.close()
            finally:
                if hasattr(self, "_temp_file") and os.path.exists(self._temp_file.name):
                    try:
                        os.unlink(self._temp_file.name)
                    except Exception:
                        pass

    def __del__(self):
        try:
            self.close()
            if hasattr(self, "_temp_file"):
                sys.stdout.write(f"del temporary HDF5 file {self._temp_file.name}\n")
                sys.stdout.flush()
        except Exception:
            pass

    # ----------------- helpers -----------------

    def _get_h5_dataset_path(self, domain: str, siteidx: int) -> str:
        return f"{domain}/{siteidx}"

    def _ensure_pinned_buf(
        self,
        key: Tuple[str, int],
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._pinned_pool.get(key)
        if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._pinned_pool[key] = buf
        return buf

    # ----------------- build envs -----------------

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
            if self.mps_save_mode == "save":
                # NOTE: these two reads are from your mps/mpo storage, not this Environ scratch.
                mps_i = torch.from_numpy(mps.read(idx)).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo.read(idx)).to(dtype_config.default_dtype).to(dtype_config.device)
            else:
                mps_i = torch.from_numpy(mps[idx]).to(dtype_config.default_dtype).to(dtype_config.device)
                mpo_i = torch.from_numpy(mpo[idx]).to(dtype_config.default_dtype).to(dtype_config.device)

            with torch.no_grad():
                tensor = contract_one_site(tensor, mps_i, mpo_i, domain).detach()
            del mps_i, mpo_i

            self.write(domain, idx, tensor)

        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def write_l_sentinel(self, length: int):
        self.write("L", -1, self.sentinel)

    def write_r_sentinel(self, length: int):
        self.write("R", length, self.sentinel)

    # ----------------- public API: GetLR / read / write -----------------

    def GetLR(self, domain, siteidx, mps, mpo, itensor=None, method="Scratch"):
        """
        method:
            - "Scratch": compute from boundary, reading mps/mpo (expensive)
            - "Enviro" : read env tensor from Environ scratch storage (cheap)
            - "System" : update one step from neighbor env (read) and local mps/mpo (read)
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
                if self.mps_save_mode == "save":
                    mps_i = torch.from_numpy(mps.read(imps)).to(dtype_config.default_dtype).to(dtype_config.device)
                    mpo_i = torch.from_numpy(mpo.read(imps)).to(dtype_config.default_dtype).to(dtype_config.device)
                else:
                    mps_i = torch.from_numpy(mps[imps]).to(dtype_config.default_dtype).to(dtype_config.device)
                    mpo_i = torch.from_numpy(mpo[imps]).to(dtype_config.default_dtype).to(dtype_config.device)
                itensor = contract_one_site(itensor, mps_i, mpo_i, domain)
            return itensor

        if method == "Enviro":
            # default: return GPU tensor
            return self.read(domain, siteidx)

        # method == "System"
        if itensor is None:
            offset = -1 if domain == "L" else 1
            itensor = self.read(domain, siteidx + offset)

        if self.mps_save_mode == "save":
            mps_i = torch.from_numpy(mps.read(siteidx)).to(dtype_config.default_dtype).to(dtype_config.device)
            mpo_i = torch.from_numpy(mpo.read(siteidx)).to(dtype_config.default_dtype).to(dtype_config.device)
        else:
            mps_i = torch.from_numpy(mps[siteidx]).to(dtype_config.default_dtype).to(dtype_config.device)
            mpo_i = torch.from_numpy(mpo[siteidx]).to(dtype_config.default_dtype).to(dtype_config.device)

        itensor = contract_one_site(itensor, mps_i, mpo_i, domain)
        self.write(domain, siteidx, itensor)
        return itensor

    def write(self, domain: str, siteidx: int, tensor: torch.Tensor):
        """
        Store env tensor to scratch.
        NOTE: This still does D2H sync + HDF5 write; if this becomes your bottleneck,
              you should pipeline it (writer thread) or change storage format/layout.
        """
        key = (domain, siteidx)
        if self.storage_backend == "h5py":
            dataset_path = self._get_h5_dataset_path(domain, siteidx)
            arr = tensor.detach().to("cpu").contiguous().numpy()

            with self._h5_lock:
                if dataset_path in self.h5_file:
                    del self.h5_file[dataset_path]
                self.h5_file.create_dataset(dataset_path, data=arr)
        else:
            self._storage[key] = tensor.detach()

    def read_cpu_pinned(self, domain: str, siteidx: int) -> torch.Tensor:
        """
        Read env tensor into a reusable pinned CPU buffer.
        Perfect for background-thread prefetch and async H2D.
        """
        key = (domain, siteidx)

        if self.storage_backend == "memory":
            # stored on GPU; bring to CPU pinned if caller really wants CPU stage
            return self._storage.get(key, self.sentinel).detach().to("cpu")

        dataset_path = self._get_h5_dataset_path(domain, siteidx)
        with self._h5_lock:
            if dataset_path not in self.h5_file:
                # pinned sentinel-like
                return torch.ones([1] * 3, dtype=dtype_config.default_dtype, pin_memory=True)

            dset = self.h5_file[dataset_path]
            torch_dtype = _np_dtype_to_torch(dset.dtype)
            buf = self._ensure_pinned_buf(key, tuple(dset.shape), torch_dtype)
            # HDF5 -> pinned CPU (no big numpy temp)
            dset.read_direct(buf.numpy())
            return buf

    def read_gpu_async(self, domain: str, siteidx: int) -> torch.Tensor:
        """
        Read env tensor as GPU tensor using:
          HDF5 -> pinned CPU (read_direct) -> GPU (non_blocking) on copy stream.
        """
        if not torch.cuda.is_available():
            # CPU-only fallback
            if self.storage_backend == "h5py":
                return self.read(domain, siteidx).to("cpu")
            return self._storage.get((domain, siteidx), self.sentinel).to("cpu")

        cpu_buf = self.read_cpu_pinned(domain, siteidx)
        with torch.cuda.stream(self._copy_stream):
            gpu = cpu_buf.to(dtype_config.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self._copy_stream)
        return gpu

    def read(self, domain: str, siteidx: int) -> torch.Tensor:
        """
        Backwards-compatible read: return tensor on dtype_config.device.
        For h5py backend we use the async pipeline above (still waits before returning).
        """
        key = (domain, siteidx)
        if self.storage_backend == "memory":
            return self._storage.get(key, self.sentinel)

        # h5py: use pinned+read_direct path
        return self.read_gpu_async(domain, siteidx)

    # ----------------- misc utils -----------------

    def keys(self):
        if self.storage_backend == "h5py":
            out = []
            with self._h5_lock:
                for domain in self.h5_file.keys():
                    if domain in ["L", "R"]:
                        for siteidx in self.h5_file[domain].keys():
                            out.append((domain, int(siteidx)))
            return out
        return list(self._storage.keys())

    def clear(self):
        if self.storage_backend == "h5py":
            with self._h5_lock:
                try:
                    self.h5_file.close()
                except Exception:
                    pass
            try:
                os.unlink(self._temp_file.name)
            except Exception:
                pass
            self._setup_storage_backend()
            self._pinned_pool.clear()
        else:
            self._storage.clear()

    def get_storage_info(self):
        if self.storage_backend == "h5py":
            size = os.path.getsize(self._temp_file.name) if os.path.exists(self._temp_file.name) else 0
            size_mib = size / (1024 * 1024)
            return {
                "backend": "h5py",
                "file_path": self._temp_file.name,
                "num_items": len(self.keys()),
                "size": f"{size_mib:.3f} MiB",
            }
        size = sum(arr.nbytes for arr in self._storage.values()) if self._storage else 0
        size_mib = size / (1024 * 1024)
        return {"backend": "memory", "num_items": len(self._storage), "size": f"{size_mib:.3f} MiB"}


# ---- keep your existing contract_one_site / env_L_i / env_R_i etc. ----
# This Environ expects contract_one_site(environ, ms, mo, domain) to exist.
def contract_one_site(environ, ms, mo, domain):
    raise NotImplementedError("Use your existing contract_one_site implementation.")
