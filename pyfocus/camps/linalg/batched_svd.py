# import time
import opt_einsum
import numpy as np
import torch

from functools import partial
from numpy.typing import NDArray
from torch import Tensor
from loguru import logger

from camps.utils.config import DtypeConfig, batch_svd_config


def split_batch_idx(dim: int, min_batch: int) -> list[int]:
    n_batches = int(np.ceil(dim / min_batch))
    idx_arr = np.full(n_batches, min_batch, dtype=np.int64)
    idx_arr[-1] = dim - (n_batches - 1) * min_batch
    return idx_arr.cumsum().tolist()


def gram_svd(x: Tensor) -> Tensor:
    *_, M, N = x.shape
    k = min(M, N)
    if M >= N:
        # Gram = A^H A, shape(..., N, N)
        gram = torch.matmul(x.conj().transpose(-2, -1), x)
    else:
        # Gram = A A^H, shape(..., M, M)
        gram = torch.matmul(x, x.conj().transpose(-2, -1))

    eigvals = torch.linalg.eigvalsh(gram)
    svals = torch.sqrt(torch.clamp(eigvals, min=0))
    svals = svals.flip(-1)[..., :k]

    return svals


def svd_cuda(x: Tensor, direct_limit: int = 400, driver=None) -> Tensor:
    if min(x.shape[-2:]) >= direct_limit or direct_limit == -1:
        return gram_svd(x)
    else:
        return torch.linalg.svdvals(x, driver=driver)


@torch.no_grad()
def backend_svd(
    psi4new: Tensor,
    gates: Tensor,
    n_dim: int,
    label: str,
    config: DtypeConfig = None,
) -> NDArray:
    """
    Compute singular values with batched SVD support.
    Args:
        psi4new: Input array of shape (c, d, l, r)
        gates: Clifford gates: (nbatch, dim**2, dim**2)
        n_dim: 2-qubits or 4-qubits,
        label: contract-path, dcba: little-endian, cdab: big-endian
        config: Optional configuration. Uses global config if None.

    Notes:
        use `example/benchmark-svd.py` to select the better config.
    """
    cfg = config or batch_svd_config
    def torch_svd(x):
        if cfg.device == "cuda":
            fn = partial(svd_cuda, direct_limit=cfg.direct_limit, driver=cfg.driver)
        else:
            fn = torch.linalg.svdvals
        return fn(x.to(cfg.device))

    svd_fn = torch_svd

    # T0 = time.time_ns()
    device = cfg.device
    is_complex = gates.is_complex() or psi4new.is_complex()
    dtype = torch.complex128 if is_complex else torch.double
    gates = gates.to(dtype=dtype, device=device)
    psi4new = psi4new.to(dtype=dtype, device=device)

    dim = gates.shape[0]
    if cfg.batch == -1 or dim <= cfg.batch:
        min_batch = gates.shape[0]
    else:
        min_batch = cfg.batch

    shape = psi4new.shape  # [c, d, l, r]
    l, r = shape[-2], shape[-1]
    values = torch.zeros((dim, min(l, r) * n_dim), dtype=cfg.default_dtype).to(device)
    idx_lst = [0] + split_batch_idx(gates.shape[0], min_batch=min_batch)
    path = f"k{label}, cdlr -> klabr"
    expr = opt_einsum.contract_expression(
        path,
        (min_batch, n_dim, n_dim, n_dim, n_dim),
        psi4new,
        constants=[1],
    )
    for i in range(len(idx_lst) - 1):
        start = idx_lst[i]
        end = idx_lst[i + 1]
        _gates = gates[start:end].reshape(-1, n_dim, n_dim, n_dim, n_dim)
        if (end - start) == min_batch:
            _psi4 = expr(_gates.conj())
        else:
            _psi4 = opt_einsum.contract(path, _gates.conj(), psi4new)
        shape = _psi4[0].shape
        batch = _psi4.shape[0]
        _psi4 = _psi4.reshape(batch, shape[0] * shape[1], shape[2] * shape[3])  # [batch, la, br]
        values[start:end] = svd_fn(_psi4)
    # torch.cuda.synchronize()
    # T1 = time.time_ns()
    # logger.debug(f"batched-SVD cost: {(T1-T0)/1e09:.3f} s")
    return values
