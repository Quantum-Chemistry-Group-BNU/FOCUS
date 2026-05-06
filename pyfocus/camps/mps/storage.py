######################################## MPS class #########################################
# -*- coding: utf-8 -*-
import os
import sys
import h5py
from pathlib import Path
import tempfile
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
import torch
from opt_einsum import contract
# from torch import Tensor
from camps.utils.config import dtype_config

def RenyiEntropy(pop: NDArray, alpha: float) -> NDArray:
    if alpha == -1:
        return np.sum(pop[len(pop) // 2 :])  # cutoff by half
    elif alpha == 1:
        raise NotImplementedError
    else:
        return 1 / (1 - alpha) * np.log(np.sum(pop**alpha, axis=-1))

class MPSStorage:
    def __init__(self, mps_list: Optional[List[NDArray]] = None, file_name: str = 'mps.h5'):
        scratch_dir_path = dtype_config.scratch_dir
        self.storage_backend = 'h5py'
        if scratch_dir_path is None:
            self.file_path = file_name
        else:
            self.file_path = scratch_dir_path + '/' + file_name
        # 根据是否提供mps_list决定打开模式
        self._is_writer = mps_list is not None
        # if self._is_writer:
        self._setup_storage_backend(mps_list)
        
        info = self.get_storage_info()
        size = info['size']
        mode = "write" if self._is_writer else "read"
        self.n_sites = self.get_system_size()
        s = f"MPSStorage: Use {self.storage_backend} in {mode} mode, size: {size} "
        s += f"HDF5 file: {self.file_path}"
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def _setup_storage_backend(self, mps_list: Optional[List[NDArray]] = None):
        # 确保文件路径在scratch目录中
        if dtype_config.scratch_dir is None:
            self.file_path = os.path.join('scratch', self.file_path)
        #if not self.file_path.startswith('scratch/'):
            
        # 确保目录存在
        file_dir = os.path.dirname(self.file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        # 根据是否提供mps_list选择打开模式
        if self._is_writer:
            # 写入模式：以'a'模式打开，允许读写，不存在则创建
            self.h5_file = h5py.File(self.file_path, 'a')
            # 如果提供了mps_list，存储数据
            if mps_list is not None:
                self.store_mps_list(mps_list)
        else:
            # 读取模式：以'r'模式打开，只读
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"MPS storage file {self.file_path} does not exist")
            self.h5_file = h5py.File(self.file_path, 'r')

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
        mode = "write" if self._is_writer else "read"
        sys.stdout.write(f"Close HDF5 file {self.file_path} ({mode} mode)" + "\n")
        sys.stdout.flush()

    def _get_h5_dataset_path(self, siteidx: int) -> str:
        return f"mps/{siteidx}"

    def store_mps_list(self, mps_list: List[NDArray]):
        """存储整个MPS列表（仅写入模式可用）"""
        if not self._is_writer:
            raise PermissionError("Cannot write in read-only mode")
        
        for idx, tensor in enumerate(mps_list):
            self.write(idx, tensor)

    def write(self, siteidx: int, tensor: NDArray):
        """写入单个MPS张量（仅写入模式可用）"""
        if not self._is_writer:
            raise PermissionError("Cannot write in read-only mode")
        
        dataset_path = self._get_h5_dataset_path(siteidx)
        if dataset_path in self.h5_file:
            del self.h5_file[dataset_path]
        # 存储到CPU
        self.h5_file.create_dataset(dataset_path, data=tensor)
        self.sync()  # 立即同步到磁盘

    def read(self, siteidx: int) -> NDArray:
        """读取单个MPS张量"""
        dataset_path = self._get_h5_dataset_path(siteidx)
        if dataset_path not in self.h5_file:
            raise KeyError(f"MPS tensor at site {siteidx} not found in storage")
        return self.h5_file[dataset_path][()]

    def read_slice(self, start_idx: int, end_idx: int) -> List[NDArray]:
        """读取MPS切片"""
        return [self.read(i) for i in range(start_idx, end_idx)]

    def read_all(self) -> List[NDArray]:
        """读取所有MPS张量"""
        keys = self.keys()
        if not keys:
            return []
        max_idx = max(keys)
        return [self.read(i) for i in range(max_idx + 1)]
    
    def leftCanonicalization(self, dcut: int = -1):
        if not self._is_writer:
            raise PermissionError("Cannot write in read-only mode")
        nsite = self.n_sites
        cpsi = torch.from_numpy(self.read(0)).to(dtype = dtype_config.default_dtype, 
                                                 device = dtype_config.device)
        shape = cpsi.shape
        cpsi = cpsi.reshape(shape[0], 1, shape[1], shape[2])
        for i in range(nsite-1):
            psi2 = cpsi.permute(1, 2, 3, 0) # ilnr->ln|ri
            shape = psi2.shape
            psi2 = psi2.reshape(shape[0] * shape[1], shape[2] * shape[3])
            if dtype_config.device == 'cpu':
                u, s, vt = torch.linalg.svd(psi2, full_matrices=False,)
            else:
                u, s, vt = torch.linalg.svd(psi2, full_matrices=False, driver=dtype_config.driver)
            d = s.shape[0]
            u = u.reshape(shape[0], shape[1], d)
            vt = contract("l,lr->lr", s, vt)
            vt = vt.reshape(d, shape[2], shape[3])
            if dcut > 0:
                d = min(d, dcut)
            data_u = u[:, :, :d]
            sites_tmp = data_u.to('cpu').numpy()
            self.write(i, sites_tmp)
            sites_tmp_new = torch.from_numpy(self.read(i+1)).to(dtype = dtype_config.default_dtype, 
                                                              device = dtype_config.device)
            cpsi = contract("lci,cnr->ilnr", vt[:d, :, :], sites_tmp_new)
        shape = cpsi.shape
        assert shape[3] == 1
        cpsi = cpsi.reshape(shape[0], shape[1], shape[2])
        data = contract("iln->lni", cpsi) / torch.linalg.norm(cpsi)
        sites_tmp = data.to('cpu').numpy()
        self.write(nsite-1, sites_tmp)
        del sites_tmp
        
    def rightCanonicalization(self, dcut: int = -1):
        if not self._is_writer:
            raise PermissionError("Cannot write in read-only mode")
        nsite = self.n_sites
        cpsi = torch.from_numpy(self.read(nsite-1)).to(dtype = dtype_config.default_dtype, 
                                                       device = dtype_config.device)
        shape = cpsi.shape
        cpsi = cpsi.reshape(shape[0], shape[1], shape[2], 1)  # lnir
        for i in range(nsite - 1, 0, -1):
            psi2 = cpsi.permute(2, 0, 1, 3)  # lnir -> il|nr
            shape = psi2.shape
            psi2 = psi2.reshape(shape[0] * shape[1], shape[2] * shape[3])
            if dtype_config.device == 'cpu':
                u, s, vt = torch.linalg.svd(psi2, full_matrices=False,)
            else:
                u, s, vt = torch.linalg.svd(psi2, full_matrices=False, driver=dtype_config.driver)
            d = s.shape[0]
            vt = vt.reshape(d, shape[2], shape[3])  # cnr
            u = contract("lr,r->lr", u, s)
            u = u.reshape(shape[0], shape[1], d)  # ilc
            if dcut > 0:
                d = min(d, dcut)
            data_v = vt[:d, :, :]
            sites_tmp = data_v.to('cpu').numpy()
            self.write(i, sites_tmp)
            sites_tmp_new = torch.from_numpy(self.read(i-1)).to(dtype = dtype_config.default_dtype, 
                                                                device = dtype_config.device)
            cpsi = contract("lnr,irc->lnic", sites_tmp_new, u[:, :, :d])
        # construct the first site
        shape = cpsi.shape
        assert shape[0] == 1
        cpsi = cpsi.reshape(shape[1], shape[2], shape[3])
        data = contract("nic->inc", cpsi) / torch.linalg.norm(cpsi)
        sites_tmp = data.to('cpu').numpy()
        self.write(0, sites_tmp)
        del sites_tmp
    
    @torch.no_grad()
    def tree_sample(self, N:int, renorm_each_step: bool = False, mode: str = "floor_bernoulli",):
        """
        Tree/branching sampling for a RIGHT-canonical OBC MPS with tensors shaped (Dl, 2, Dr).

        Given N total samples, we recursively split the sample count at each site using conditional
        probabilities, pruning branches that receive 0 samples. At leaves (full bitstrings),
        return (bitstring, count_assigned, exact_probability).

        Assumptions:
          - A_list[i] has shape (Dl, 2, Dr)
          - Right-canonical condition: sum_s A[:,s,:] @ A[:,s,:].conj().T = I_{Dl}
          - OBC: D0=1, Dn=1 (first Dl=1, last Dr=1)
          - Physical dim is 2

        Args:
          A_list: list of np.ndarray, each (Dl, 2, Dr)
          N: int, total number of samples to allocate
          rng: np.random.Generator
          renorm_each_step: if True, normalize ell at each step; DOES NOT change probabilities
                            computed from ratios if we handle Z consistently (we do).

        Returns:
          results: list of tuples (bits_tuple, count, prob_exact)
                   where prob_exact is the exact Born probability of that bitstring.
        """
        sites = self.read_all()
        n = self.n_sites
        if N <= 0:
            return []
        # Each node in the "frontier" is: (prefix_bits_tuple, ell_vector, prob_prefix, count)
        # prob_prefix = product of conditional probabilities along the prefix (exact)
        frontier = [(tuple(), 
                    torch.tensor([1.0 + 0.0j], 
                                 dtype=dtype_config.default_dtype,
                                 device=dtype_config.device), 
                    1.0, 
                    int(N))]
        for i in range(n):
            Ai = sites[i]
            if Ai.ndim != 3 or Ai.shape[1] != 2:
                raise ValueError(f"Site {i}: expected shape (Dl,2,Dr), got {tuple(Ai.shape)}")

            Dl, _, Dr = Ai.shape
            new_frontier = []
            
            # Pre-slice once (saves a little overhead)
            A0 = torch.from_numpy(Ai[:, 0, :]).to(dtype=dtype_config.default_dtype,
                                                  device=dtype_config.device)  # (Dl, Dr)
            A1 = torch.from_numpy(Ai[:, 1, :]).to(dtype=dtype_config.default_dtype,
                                                  device=dtype_config.device)  # (Dl, Dr)
            
            for prefix, ell, p_pref, cnt in frontier:
                if cnt == 0:
                    continue
                if ell.numel() != Dl:
                    raise ValueError(f"Site {i}: ell dim {ell.numel()} != Dl {Dl}")

                # v_s = ell @ A[:, s, :]  -> (Dr,)
                # ell: (Dl,), A0: (Dl,Dr) => v0: (Dr,)
                v0 = ell @ A0
                v1 = ell @ A1

                # w_s = ||v_s||^2 = sum |v_s|^2  (real scalar)
                # Use float64 accumulator for stability, then convert to python float for branching logic
                w0 = (v0.conj() * v0).real.sum()
                w1 = (v1.conj() * v1).real.sum()
                Z = w0 + w1
            
                # Pull to CPU scalars for control flow; this syncs, but branching itself is inherently sequential
                Z_val = float(Z.item())
                if not (Z_val > 0.0) or not (Z_val == Z_val):  # Z>0 and not NaN
                    raise FloatingPointError(f"Site {i}: invalid Z={Z_val}")

                p0 = float((w0 / Z).item())
                # Numerical clamp
                if p0 < 0.0:
                    p0 = 0.0
                elif p0 > 1.0:
                    p0 = 1.0
                p1 = 1.0 - p0
            
                # Allocate counts
                if mode == "binomial":
                    # exact: n0 ~ Binomial(cnt, p0)
                    # torch.binomial expects tensors; do on device for RNG then item()
                    p0_t = torch.tensor(p0, device=dtype_config.device)
                    n0 = int(torch.binomial(torch.tensor(float(cnt), device=dtype_config.device), p0_t).item())
                else:
                    # low-variance: floor(cnt*p0) + Bernoulli(frac)
                    expected = cnt * p0
                    n0 = int(expected)  # floor for positive numbers
                    frac = expected - n0
                    if frac > 0.0:
                        u = float(torch.rand((), device=dtype_config.device, dtype=torch.float64).item())
                        if u < frac:
                            n0 += 1
                n1 = cnt - n0
            
                # Push branches (keep ell on GPU)
                if n0 > 0:
                    ell0 = v0
                    if renorm_each_step:
                        norm0 = torch.linalg.norm(ell0)
                        if float(norm0.item()) > 0.0:
                            ell0 = ell0 / norm0
                    new_frontier.append((prefix + (0,), ell0, p_pref * p0, n0))

                if n1 > 0:
                    ell1 = v1
                    if renorm_each_step:
                        norm1 = torch.linalg.norm(ell1)
                        if float(norm1.item()) > 0.0:
                            ell1 = ell1 / norm1
                    new_frontier.append((prefix + (1,), ell1, p_pref * p1, n1))
        
            frontier = new_frontier
            if len(frontier) == 0:
                break
        results = [(bits, cnt, float(p)) for (bits, ell, p, cnt) in frontier]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results
        
    def slstFromDM(self, iroot=0) -> list[NDArray]:
        n_sites = self.n_sites
        mps_tmp = MPSStorage(self.read_all(), file_name='mps_tmp4slst.h5')
        slst: list[NDArray] = []
        for i in range(1, n_sites):
            if i == 1:
                site0 = mps_tmp.read(i-1)[iroot]
                shape = site0.shape
                site0 = site0.reshape(1, shape[0], shape[1])
            else:
                site0 = mps_tmp.read(i-1)
            site1 = mps_tmp.read(i)
            psi = contract("lnr,rmx->lnmx", torch.from_numpy(site0).to(dtype_config.device), 
                           torch.from_numpy(site1).to(dtype_config.device))  # twodot wavefunction
            shape = psi.shape
            psi = psi.reshape(shape[0] * shape[1], shape[2] * shape[3])
            if dtype_config.device == 'cpu':
                u, s, vt = torch.linalg.svd(psi, full_matrices=False,)
            else:
                u, s, vt = torch.linalg.svd(psi, full_matrices=False, driver=dtype_config.driver)    # todo: test which driver is best.
            vt = contract("l,lr->lr", s, vt)
            mps_tmp.write(i - 1, (u.reshape(shape[0], shape[1], s.shape[0])).to('cpu').numpy())
            mps_tmp.write(i, vt.reshape(s.shape[0], shape[2], shape[3]).to('cpu').numpy())
            slst.append((s.to('cpu').numpy())**2)
        mps_tmp.delete_file()
        return slst
    
    def sumOfReyniEntropyFromSites(self, alpha: float) -> float:
        slst = self.slstFromDM()
        return np.sum(np.stack([RenyiEntropy(slst[i], alpha) for i in range(len(slst))]))

    def keys(self):
        """返回所有存储的索引"""
        if 'mps' not in self.h5_file:
            return []
        keys = [int(siteidx) for siteidx in self.h5_file['mps'].keys()]
        return sorted(keys)

    def clear(self):
        """清空存储（删除mps组，仅写入模式可用）"""
        if not self._is_writer:
            raise PermissionError("Cannot clear in read-only mode")
        
        if 'mps' in self.h5_file:
            del self.h5_file['mps']
        self.sync()

    def delete_file(self):
        """删除整个HDF5文件"""
        # if not self._is_writer:
        #     raise PermissionError("Cannot delete file in read-only mode")
        self.close()
        if os.path.exists(self.file_path):
            os.unlink(self.file_path)

    def close(self):
        """关闭HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

    def reopen(self, mode: str = None):
        """重新打开HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()
        
        if mode is None:
            mode = 'a' if self._is_writer else 'r'
        if mode == 'a':
            self._is_writer = True
        elif mode == 'r':
            self._is_writer = False
        else:
            assert False
        self.h5_file = h5py.File(self.file_path, mode)

    def get_storage_info(self):
        """获取存储信息"""
        size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        size = size / (1024 * 1024)  # MiB
        mode = "write" if self._is_writer else "read"
        return {
            "backend": "h5py",
            "file_path": self.file_path,
            "mode": mode,
            "num_items": len(self.keys()),
            "size": f"{size:.3f} MiB"
        }

    def sync(self):
        """确保数据写入磁盘（仅写入模式有效）"""
        if self.h5_file and self._is_writer:
            self.h5_file.flush()

    def __len__(self) -> int:
        """返回存储的MPS张量数量"""
        return len(self.keys())

    def __contains__(self, siteidx: int) -> bool:
        """检查指定位置的MPS张量是否存在"""
        dataset_path = self._get_h5_dataset_path(siteidx)
        return dataset_path in self.h5_file

    def get_system_size(self) -> int:
        """获取系统大小（MPS链长度）"""
        keys = self.keys()
        return max(keys) + 1 if keys else 0

    @property
    def is_writer(self) -> bool:
        """返回是否为写入模式"""
        return self._is_writer

    @classmethod
    def open_for_reading(cls, file_name: str = 'mps.h5'):
        """专门用于读取的模式打开MPS存储文件"""
        return cls(mps_list=None, file_name=file_name)

    @classmethod  
    def open_for_writing(cls, mps_list: Optional[List[NDArray]] = None, file_name: str = 'mps.h5'):
        """专门用于写入的模式打开MPS存储文件"""
        return cls(mps_list=mps_list, file_name=file_name)
    
class MPOStorage:
    def __init__(self, mpo_list: Optional[List[NDArray]] = None):
        scratch_dir_path = dtype_config.scratch_dir
        self.storage_backend = 'h5py'
        self.file_dir = scratch_dir_path
        self._setup_storage_backend(mpo_list)
        info = self.get_storage_info()
        size = info['size']
        # mode = "write" if self._is_writer else "read"
        s = f"MPOStorage: Use {self.storage_backend}, size: {size} "
        s += f"HDF5 file: {self._temp_file.name}"
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

    def _setup_storage_backend(self, mpo_list: Optional[List[NDArray]] = None):
        if self.file_dir is None:
            scratch_dir = Path("scratch")
            scratch_dir.mkdir(exist_ok=True)
        else:
            scratch_dir = Path(self.file_dir)
            #scratch_dir.mkdir(parents=True, exist_ok=True) 
        self._temp_file = tempfile.NamedTemporaryFile(delete=False,
                                                      dir=scratch_dir,
                                                      suffix='.h5')
        self._temp_file.close()
        self.h5_file = h5py.File(self._temp_file.name, 'w')
        if mpo_list is not None:
            self.store_mpo_list(mpo_list)

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

    def _get_h5_dataset_path(self, siteidx: int) -> str:
        return f"mpo/{siteidx}"

    def store_mpo_list(self, mpo_list: List[NDArray]):
        """存储整个MPO列表"""
        for idx, tensor in enumerate(mpo_list):
            self.write(idx, tensor)

    def write(self, siteidx: int, tensor: NDArray):
        """写入单个MPO张量"""
        dataset_path = self._get_h5_dataset_path(siteidx)
        if dataset_path in self.h5_file:
            del self.h5_file[dataset_path]
        self.h5_file.create_dataset(dataset_path, data=tensor)
        self.sync()  # 立即同步到磁盘

    def read(self, siteidx: int) -> NDArray:
        """读取单个MPO张量"""
        dataset_path = self._get_h5_dataset_path(siteidx)
        if dataset_path not in self.h5_file:
            raise KeyError(f"MPO tensor at site {siteidx} not found in storage")
        return self.h5_file[dataset_path][()]

    def read_slice(self, start_idx: int, end_idx: int) -> List[NDArray]:
        """读取MPO切片"""
        return [self.read(i) for i in range(start_idx, end_idx)]

    def read_all(self) -> List[NDArray]:
        """读取所有MPO张量"""
        keys = self.keys()
        if not keys:
            return []
        max_idx = max(keys)
        return [self.read(i) for i in range(max_idx + 1)]

    def keys(self):
        """返回所有存储的索引"""
        if 'mpo' not in self.h5_file:
            return []
        keys = [int(siteidx) for siteidx in self.h5_file['mpo'].keys()]
        return sorted(keys)

    def clear(self):
        """清空存储（删除mpo组，仅写入模式可用）"""
        if 'mpo' in self.h5_file:
            del self.h5_file['mpo']
        self.sync()

    def delete_file(self):
        """删除整个HDF5文件"""
        # if not self._is_writer:
        #     raise PermissionError("Cannot delete file in read-only mode")
        self.close()
        # 修复：使用 _temp_file.name 而不是 file_path
        if os.path.exists(self._temp_file.name):
            os.unlink(self._temp_file.name)

    def close(self):
        """关闭HDF5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

    def get_storage_info(self):
        """获取存储信息"""
        # 修复：统一使用 _temp_file.name
        file_path = self._temp_file.name
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        size = size / (1024 * 1024)  # MiB
        return {
            "backend": "h5py",
            "file_path": file_path,
            "num_items": len(self.keys()),
            "size": f"{size:.3f} MiB"
        }

    def sync(self):
        """确保数据写入磁盘（仅写入模式有效）"""
        self.h5_file.flush()

    def __len__(self) -> int:
        """返回存储的MPO张量数量"""
        return len(self.keys())

    def __contains__(self, siteidx: int) -> bool:
        """检查指定位置的MPO张量是否存在"""
        dataset_path = self._get_h5_dataset_path(siteidx)
        return dataset_path in self.h5_file

    def get_system_size(self) -> int:
        """获取系统大小（MPO链长度）"""
        keys = self.keys()
        return max(keys) + 1 if keys else 0