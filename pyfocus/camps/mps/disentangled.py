import copy
import time
import random
import platform
import opt_einsum
import scipy
import torch
import numpy as np

from collections.abc import Callable
from functools import partial
from typing import Literal, Tuple, List, Union
from loguru import logger
from numpy.typing import NDArray
from torch import Tensor

from pyfocus.camps.mps.mps_simple import sumOfReyniEntropyFromSites, RenyiEntropy, overlap
from pyfocus.camps.mps.storage import MPSStorage, MPOStorage
from pyfocus.camps.utils.pauli_alg import (
    mapping_inverse,
    pauli_transform,
    pauli_to_vector,
    vector_to_pauli,
)
from pyfocus.camps.utils.typing import Sites, PauliArray, RandomClifford, Clifford, Endianness
from pyfocus.camps.utils.clifford import random_clifford
from pyfocus.camps.utils.config import dtype_config, batch_svd_config
from pyfocus.camps.linalg import backend_svd

ContractPath = Literal["badc", "abcd"]
ENDIAN_CONTACT_MAP: dict[Endianness, ContractPath] = {"big": "abcd", "little": "badc"}

# from pyfocus.camps.linalg.svd import optimized_svd


def s2forPsi4_clifford(
    psi4new: Tensor,
    gate: Tensor,
    n_dim: int = 2,
    label: str = "abcd",
) -> Tensor:
    gate = gate.reshape(n_dim, n_dim, n_dim, n_dim)
    # path = "badc/abcd, cdlr -> labr"  # big/little endian
    path = f"{label}, cdlr -> labr"
    # print(f'gate: {gate.dtype}, psi4new: {psi4new.dtype}')
    psi4 = opt_einsum.contract(path, gate, psi4new)
    shape = psi4.shape
    psi4 = psi4.reshape(shape[0] * shape[1], shape[2] * shape[3])
    if dtype_config.device == 'cpu':
        u, s, vt = torch.linalg.svd(psi4, full_matrices=False,)
    else:
        u, s, vt = torch.linalg.svd(psi4, full_matrices=False, driver=dtype_config.driver)
    # s2 = np.asarray(s) ** 2  # s is real
    s2 = s**2
    # normalize
    s2 = s2 / s2.sum()

    return s2


def minimize_entropy(
    psi4new: Tensor,
    clifford_gates: Tensor,
    alpha: float = 0.5,
    n_dim: int = 2,
    label: str = "abcd",
) -> tuple[int, float, Tensor, Tensor]:
    with torch.no_grad():
        s = backend_svd(
            psi4new.to(batch_svd_config.device),
            clifford_gates.to(batch_svd_config.device),
            n_dim,
            label,
            )
    # u: [batch, la, k], s: [nbatch, k], vt: [batch, k, br], k = min(la, br)
    s2 = s ** 2
    # normalize
    s2 = s2 / s2.sum(dim=-1, keepdim=True)
    s2_0 = RenyiEntropy(s2, alpha)
    index = torch.argmin(s2_0)
    value = s2_0[index]
    gate = clifford_gates[index]
    return index, value, gate.to(dtype_config.device)


def minimize_entropy_singleSweep(
    sites: Union[List[NDArray], MPSStorage],
    dmax: int,
    clifford_gates: Tensor,
    alpha: float = 0.5,
    iroot: int = 0,
    n_dim: int = 2,
    contract_path: str = "abcd",
    save_mode: str = 'save'
) -> tuple[Sites, torch.Tensor, float, torch.Tensor]:
    
    device = dtype_config.device
    dtype = dtype_config.default_dtype
    
    if save_mode == 'save':
        s0 = sites.sumOfReyniEntropyFromSites(alpha)
        n_sites = sites.get_system_size()
    elif save_mode == 'normal':
        sites_tmp = [site for site in sites]
        s0 = sumOfReyniEntropyFromSites(sites_tmp, alpha).to('cpu').numpy()
        del sites_tmp
        n_sites = len(sites)
    assert n_dim in (2, 4)
    # badc: little-endian(Gates), abcd: big-endian
    assert contract_path in ("badc", "abcd")

    max_dwt = 0
    totaldiff = 0
    entropy_lst = []
    idx_clifford = torch.zeros((2, n_sites - 1), dtype=torch.int64).to(dtype_config.device)

    # Forward sweep
    for i in range(1, n_sites):
        t0 = time.time_ns()
        if i == 1:
            if save_mode == 'save':
                site0 = sites.read(i - 1)[iroot]
                site0 = torch.from_numpy(site0).to(dtype=dtype, device=device)
            elif save_mode == 'normal':   # 一直使用GPU
                site0 = sites[i - 1][iroot]
            shape = site0.shape
            site0 = site0.reshape(1, shape[0], shape[1])
        else:
            if save_mode == 'save':
                site0 = sites.read(i - 1)
                site0 = torch.from_numpy(site0).to(dtype=dtype, device=device)
            elif save_mode == 'normal':
                site0 = sites[i - 1]
        if save_mode == 'save':
            site1 = sites.read(i)
            site1 = torch.from_numpy(site1).to(dtype=dtype, device=device)
        elif save_mode == 'normal':
            site1 = sites[i]
        psi4 = opt_einsum.contract("lnr,rmx->lnmx", site0, site1)  # twodot wavefunction
        psi4new = opt_einsum.contract("lnmr->nmlr", psi4)
        func = lambda gate: RenyiEntropy(
            s2forPsi4_clifford(psi4new, gate, n_dim, contract_path),
            alpha,
        )

        idx, value, gate = minimize_entropy(
            psi4new,
            clifford_gates,
            alpha,
            n_dim,
            contract_path,
        )
        # idx = random.randint(0, clifford_gates.shape[0] - 1)
        # gate = clifford_gates[idx]

        idx_clifford[0][i - 1] = idx
        gate = gate.reshape(n_dim, n_dim, n_dim, n_dim)
        psi4 = opt_einsum.contract(f"{contract_path}, cdlr -> labr", gate, psi4new)
        shape = psi4.shape
        psi4 = psi4.reshape(shape[0] * shape[1], shape[2] * shape[3])
        # u, s, vt = torch.linalg.svd(psi4, full_matrices=False, driver=dtype_config.driver)
        if dtype_config.device == 'cpu':
            u, s, vt = torch.linalg.svd(psi4, full_matrices=False,)
        else:
            u, s, vt = torch.linalg.svd(psi4, full_matrices=False, driver=dtype_config.driver)
        # Update new site: LCF
        vt = opt_einsum.contract("l,lr->lr", s, vt)
        dcut = min(dmax, len(s))
        dwt = torch.sum(s[dcut:] ** 2)
        f0 = func(torch.eye(2**n_dim, dtype=dtype).to(device))
        # ft = value
        # ft = func(gate.reshape(n_dim**2, n_dim**2))
        ft = func(gate.reshape(n_dim**2, n_dim**2))
        max_dwt = max(max_dwt, dwt)
        tmp_i1 = u[:, :dcut].reshape(shape[0], shape[1], dcut)
        tmp_i = vt[:dcut, :].reshape(dcut, shape[2], shape[3])
        if save_mode == 'save':
            tmp_i1 = tmp_i1.to('cpu').numpy()
            tmp_i = tmp_i.to('cpu').numpy()
            sites.write(i-1, tmp_i1)
            sites.write(i, tmp_i)
        elif save_mode == 'normal':
            sites[i - 1] = tmp_i1
            sites[i] = tmp_i

        t1 = time.time_ns()
        totaldiff += ft - f0
        entropy_lst.append((ft, f0))
        delta = ft - f0
        logger.info(f"Disentangle sites: ({i-1}, {i}) f0: {f0:.5E} -> ft: {ft:.5E} delta: {delta:.5E}")
    # logger.info(f"idx-gate: {idx_clifford}")
    # return sites_tmp, idx_clifford, None, None
    # Backward sweep
    for i in reversed(range(1, n_sites)):
        t0 = time.time_ns()
        if save_mode == 'save':
            site0 = sites.read(i - 1)
            site1 = sites.read(i)
            site0 = torch.from_numpy(site0).to(dtype=dtype, device=device)
            site1 = torch.from_numpy(site1).to(dtype=dtype, device=device)
        elif save_mode == 'normal':
            site0 = sites[i - 1]
            site1 = sites[i]
        psi4 = opt_einsum.contract("lnr,rmx->lnmx", site0, site1)  # twodot wavefunction
        psi4new = opt_einsum.contract("lnmr->nmlr", psi4)
        # Update psi4
        func = lambda gate: RenyiEntropy(s2forPsi4_clifford(psi4new, gate, n_dim, contract_path), alpha)
        idx, value, gate = minimize_entropy(
            psi4new,
            clifford_gates,
            alpha,
            n_dim,
            contract_path,
        )

        # idx = random.randint(0, clifford_gates.shape[0] - 1)
        # gate = clifford_gates[idx]

        idx_clifford[1][i - 1] = idx
        gate = gate.reshape(n_dim, n_dim, n_dim, n_dim)
        psi4 = opt_einsum.contract(f"{contract_path}, cdlr -> labr", gate, psi4new)
        shape = psi4.shape
        psi4 = psi4.reshape(shape[0] * shape[1], shape[2] * shape[3])  # [la, br]
        if dtype_config.device == 'cpu':
            u, s, vt = torch.linalg.svd(psi4, full_matrices=False,)
        else:
            u, s, vt = torch.linalg.svd(psi4, full_matrices=False, driver=dtype_config.driver)
        # Update new site: RCF
        u = opt_einsum.contract("lr,r->lr", u, s)
        dcut = min(dmax, len(s))
        dwt = torch.sum(s[dcut:] ** 2)

        f0 = func(torch.eye(n_dim**2, dtype=dtype).to(device))
        ft = func(gate.reshape(n_dim**2, n_dim**2))

        max_dwt = max(max_dwt, dwt)
        max_dwt = max(max_dwt, dwt)
        tmp_i1 = u[:, :dcut].reshape(shape[0], shape[1], dcut)
        tmp_i = vt[:dcut, :].reshape(dcut, shape[2], shape[3])
        if save_mode == 'save':
            tmp_i1 = tmp_i1.to('cpu').numpy()
            tmp_i = tmp_i.to('cpu').numpy()
            sites.write(i-1, tmp_i1)
            sites.write(i, tmp_i)
        elif save_mode == 'normal':
            sites[i - 1] = tmp_i1
            sites[i] = tmp_i

        t1 = time.time_ns()
        totaldiff += ft - f0
        entropy_lst.append((ft, f0))
        delta = ft - f0
        logger.info(f"Disentangle sites: ({i-1}, {i}) f0: {f0:.5E} -> ft: {ft:.5E} delta: {delta:.5E}")
        
    # final
    if save_mode == 'save':
        site0 = sites.read(0)
        ovlp = opt_einsum.contract("lnr,lnr", site0, site0.conj())
        sites.write(0, site0 / np.sqrt(ovlp))
        s1 = sites.sumOfReyniEntropyFromSites(alpha)
    elif save_mode == 'normal':
        ovlp = opt_einsum.contract("lnr,lnr", sites[0], sites[0].conj())
        sites[0] = sites[0] / torch.sqrt(ovlp)
        s1 = sumOfReyniEntropyFromSites(sites, alpha).to('cpu').numpy()
    totaldiff = s1 - s0  # when there is truncation, s1-s0 can differ from totaldiff!
    # idx_clifford = np.asarray(idx_clifford, dtype=np.int64)
    logger.info(f"Max truncation error: {max_dwt:.3E}")
    logger.info(f"SingleSweep finished Renyi entropy: {s0:.5E} -> {s1:.5E}, delta: {s0-s1:.5E}")
    logger.info(f"Gate idx: {idx_clifford}")
    #entropy = np.asarray(entropy_lst)
    return sites, idx_clifford, totaldiff, entropy_lst


def minimize_entropy_multiSweep(
    sites: Union[List[NDArray], MPSStorage],
    dmax: int,
    clifford: Clifford,
    microiter: int = 5,
    iroot: int = 0,
    alpha: float = 0.5,
    n_dim: int = 2,
    *,
    use_random_gates: bool = False,
    random_nums: int = 200,
    random_endian: str = "big",
    given_clifford: Clifford = None,
    save_mode: str = 'save'
) -> tuple[Union[List[NDArray], MPSStorage], Tensor, float, list[RandomClifford]]:
    # cdab: big-endian, dcba: little-endian
    device = dtype_config.device
    dtype = dtype_config.default_dtype
    
    if use_random_gates:
        assert clifford.endian == random_endian
    endian = clifford.endian
    path = ENDIAN_CONTACT_MAP[endian]

    T0 = time.time_ns()
    # sites_tmp = copy.deepcopy(sites)
    if save_mode == 'save':
        sites_tmp = sites.open_for_writing(sites.read_all(), file_name='mps_disen.h5')
    elif save_mode == 'normal':
        sites_tmp = [torch.from_numpy(site).to(dtype_config.device) for site in sites]
    delta_total = 0.0
    flag_conv = False
    idx_clifford = []
    if save_mode == 'save':
        s0 = sites_tmp.sumOfReyniEntropyFromSites(alpha)
        n_qubits = sites_tmp.read(0).shape[1]
    elif save_mode == 'normal':
        s0 = sumOfReyniEntropyFromSites(sites_tmp, alpha).to('cpu').numpy()
        n_qubits = sites[0].shape[1]
    logger.info(f"Initial Renyi entropy: {s0:.5E}")

    random_lst: list[RandomClifford] = [] if use_random_gates else None

    for i in range(microiter):
        # print('=== micro iter=',i,'===')
        t0 = time.time_ns()
        logger.info(f"{'*' * 20} micro iter {i} {'*' * 20}")
        if use_random_gates:
            res = random_clifford(
                random_nums,
                n_qubits,
                given_clifford=given_clifford,
                endian=random_endian,
            )
            random_lst.append(res)
            _gates = res.gates
            logger.info(f"Use random-gates: {_gates.shape}")
        else:
            _gates = clifford.gates
            logger.info(f"gates: {_gates.shape}")
        _gates = torch.from_numpy(_gates).to(dtype=dtype, device=device)
            
        logger.info(f"dmax: {dmax}")
        sites_tmp, idx, delta_i, entropy = minimize_entropy_singleSweep(
            sites_tmp,
            dmax,
            _gates,
            alpha,
            iroot,
            n_dim,
            path,
            save_mode,
        )
        del _gates
        idx_clifford.append(idx)
        # smat = overlap(sites, sites_tmp)
        # logger.info(f"<MPS[0]|MPS[new]>= {smat.reshape(-1)[0]:.5E}")
        t1 = time.time_ns()
        logger.info(f"micro i-step: diff: {delta_i}, cost: {(t1-t0)/1e09:.5E} s")
        delta_total += delta_i
        if np.abs(delta_i) < 1.0e-8:
            logger.info(f"minimize entropy multiSweep: Convergence is reached in {i} steps!")
            flag_conv = True
            break
    if save_mode == 'save':
        s1 = sites_tmp.sumOfReyniEntropyFromSites(alpha)
    elif save_mode == 'normal':
        s1 = sumOfReyniEntropyFromSites(sites_tmp, alpha).to('cpu').numpy()
    T1 = time.time_ns()
    delta_time = (T1 - T0) / 1e09
    logger.info(f"Renyi entropy: {s0:.5E} -> {s1:.5E}, delta: {abs(s1-s0):.5E}, cost: {delta_time:.5E} s")
    if flag_conv == False:
        logger.warning("Warning: Convergence fails! out of micro-iter=", microiter)
    idx_clifford = torch.stack(idx_clifford)
    if save_mode == 'normal':
        sites_tmp = [site.to('cpu').numpy() for site in sites_tmp]
    return sites_tmp, idx_clifford, delta_total, random_lst


# It is update the hamiltonian 
def update_ham_singleSweep(
    Ham_array: PauliArray,
    idx: NDArray[np.int64],
    tableau: NDArray[np.int64],
    n_sites: int,
    n_dim: int = 2,
    in_place: bool = False,
    phase: NDArray[np.int64] = None,
) -> tuple[PauliArray, NDArray[np.int64]]:
    assert idx.shape == (2, n_sites - 1), f"{idx.shape}"
    # idx: 1 -> n_sites; 1-> n_sites

    n_ham, n_qubits = Ham_array.shape
    offset = n_qubits // n_sites
    assert offset in (1, 2)
    assert n_dim in (2, 4)

    if n_dim == 2:
        # X0 X1 Z1 Z2 -> X0 Z1 X1 Z2
        new_order = [0, 2, 1, 3]
        assert tableau[0].shape == (4, 4)
    else:
        # X0 X1 X2 X3 Z0 Z1 Z2 Z3 -> X0 Z0 X1 Z1 X2 Z2 X3 Z3
        new_order = [0, 4, 1, 5, 2, 6, 3, 7]
        assert tableau[0].shape == (8, 8)
    sign = np.ones(n_ham, dtype=np.int64)

    if phase is None:
        # TODO: check it when using phases
        phase = np.zeros((len(tableau), n_qubits * 2), dtype=np.int64)

    Ham_tmp = Ham_array if in_place else Ham_array.copy()
    # stupid/foolish bug note: offset Acknowledge LongFei Chang
    for i in range(n_sites - 1):
        start = i * offset
        end = start + n_dim
        gs_in = pauli_to_vector(Ham_tmp[:, start:end])
        M = tableau[idx[0][i]][new_order]
        p = phase[idx[0][i]][new_order]
        # print(f'M: {M.dtype}, p: {p.dtype}')
        # M_inv, ps_inv = mapping_inverse(M, p)
        result, ps = pauli_transform(gs_in, gs_map=M, ps_map=p)
        pauli_new = vector_to_pauli(result)

        # import pyclifford as pc
        # from pyclifford.stabilizer import CliffordMap
        # cmap = CliffordMap(M, p)
        # cmap_inv = cmap.inverse()
        # string = [Ham_tmp[j, i: i+n_dim].tobytes().decode() for j in range(n_ham)]
        # _pauli = pc.paulis(*string).transform_by(cmap_inv)
        # pauli_1 = vector_to_pauli(_pauli.gs)
        # logger.info(f"Delta: {np.sum(pauli_1 != pauli_new)}")
        Ham_tmp[:, start:end] = pauli_new
        # 1j**[0, 2] -> 1j**[1, -1]
        assert np.all(np.isin(ps, [0, 2]))
        sign *= np.where(ps == 0, 1, -1)
    # return Ham_tmp, sign
    for i in reversed(range(n_sites - 1)):
        start = i * offset
        end = start + n_dim
        gs_in = pauli_to_vector(Ham_tmp[:, start:end])
        M = tableau[idx[1][i]][new_order]
        p = phase[idx[0][i]][new_order]
        #M_inv, ps_inv = mapping_inverse(M, p)
        result, ps = pauli_transform(gs_in, gs_map=M, ps_map=p)
        pauli_new = vector_to_pauli(result)
        Ham_tmp[:, start:end] = pauli_new

        # 1j**[0, 2] -> 1j**[1, -1]
        assert np.all(np.isin(ps, [0, 2]))
        sign *= np.where(ps == 0, 1, -1)

    return Ham_tmp, sign


def update_ham_multiSweep(
    Ham_array: PauliArray,
    idx: NDArray[np.int64],
    clifford: Clifford | list[Clifford],
    n_sites: int,
    in_place: bool = False,
    n_dim: int = 2,
) -> tuple[PauliArray, NDArray[np.int64]]:
    n_macro = idx.shape[0]
    n_ham = Ham_array.shape[0]
    assert idx.shape == (n_macro, 2, (n_sites - 1)), f"idx: {idx.shape}"
    Ham_tmp = Ham_array if in_place else Ham_array.copy()
    sign = np.ones(n_ham, dtype=np.int64)

    t0 = time.time_ns()
    if isinstance(clifford, list):
        assert len(clifford) == n_macro
    else:
        # shallow copy
        clifford = [clifford] * n_macro

    for i in range(n_macro):
        tableau = clifford[i].mapping
        phase = clifford[i].phases

        Ham_tmp, sign_tmp = update_ham_singleSweep(
            Ham_tmp,
            idx[i],
            tableau,
            n_sites,
            n_dim,
            phase=phase,
            in_place=True,
        )
        sign *= sign_tmp
    t1 = time.time_ns()
    delta = (t1 - t0) / 1e9
    logger.info(f"Update hamiltonian finished, cost: {delta:.5E} s")
    return Ham_tmp, sign
