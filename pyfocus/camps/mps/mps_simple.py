import copy
import torch
import numpy as np
import scipy

from functools import partial
from opt_einsum import contract
#from numpy.typing import NDArray
from torch.types import Tensor

from pyfocus.camps.utils.typing import Sites
# from pyfocus.camps.linalg.svd import optimized_svd   # todo: change the driver to the torch version.
from pyfocus.camps.utils.config import dtype_config

print = partial(print, flush=True)


# copy from Focus/pyutils
def get_SvN(pop: Tensor, thresh: float = 1.0e-100) -> float:
    s = 0.0
    for p in pop:
        if p < thresh:
            continue
        s += -p * torch.log(p)
    return s


def bipartiteEntropy(sites: Sites, iroot: int = 0) -> list[float]:
    slst = slstFromDM(sites)
    return [get_SvN(x) for x in slst]


def overlap(sites1: Sites, sites2: Sites):
    assert len(sites1) == len(sites2)
    env = torch.ones((1, 1), dtype=sites1[0].dtype).to(dtype_config.device)
    for i in range(len(sites1) - 1, -1, -1):
        tmp = contract("lnr,dr->lnd", sites2[i], env)
        env = contract("lnr,mnr->lm", sites1[i].conj(), tmp)
    return env


def checkRCF(sites: Sites, thresh=1.0e-10):
    n_sites = len(sites)
    diff = 0
    for i in range(n_sites - 1, -1, -1):
        ova = contract("unr,dnr->ud", sites[i], sites[i])
        d = ova.shape[0]
        diff_i = torch.linalg.norm(ova - torch.eye(d, dtype=dtype_config.default_dtype).to(dtype_config.device))
        diff += diff_i
        print("check i=", i, " site.shape=", sites[i].shape, " |S-I|=", diff_i)
    if diff > n_sites * thresh:
        print("MPS is not in RCF: diff=", diff)
        return False
    else:
        print("MPS is in RCF")
        return True


def checkLCF(sites: Sites, thresh=1.0e-10):
    nsite = len(sites)
    diff = 0
    for i in range(0, nsite):
        ova = contract("lnu,lnd->ud", sites[i], sites[i])
        d = ova.shape[0]
        diff_i = torch.linalg.norm(ova - torch.eye(d, dtype=dtype_config.default_dtype).to(dtype_config.device))
        diff += diff_i
        print("check i=", i, " site.shape=", sites[i].shape, " |S-I|=", diff_i)
    if diff > nsite * thresh:
        print("MPS is not in LCF: diff=", diff)
        return False
    else:
        print("MPS is in LCF")
        return True


# We assume MPS is only for a single state!
def leftCanonicalization(sites: Sites, dcut: int = -1):
    nsite = len(sites)
    sites_tmp = copy.deepcopy(sites)
    cpsi = sites_tmp[0]
    shape = cpsi.shape
    cpsi = cpsi.reshape(shape[0], 1, shape[1], shape[2])  # ilnr
    for i in range(nsite - 1):
        psi2 = cpsi.permute(1, 2, 3, 0)  # ilnr->ln|ri
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
        sites_tmp[i] = data_u.contiguous().clone()
        # update cpsi for the next site
        cpsi = contract("lci,cnr->ilnr", vt[:d, :, :], sites_tmp[i + 1])
    # construct the last site
    shape = cpsi.shape
    assert shape[3] == 1
    cpsi = cpsi.reshape(shape[0], shape[1], shape[2])
    data = contract("iln->lni", cpsi) / torch.linalg.norm(cpsi)
    sites_tmp[nsite - 1] = data.contiguous().clone()
    del sites
    return sites_tmp


def rightCanonicalization(sites: Sites, dcut: int = -1):
    nsite = len(sites)
    sites_tmp = copy.deepcopy(sites)
    cpsi = sites_tmp[-1]
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
        sites_tmp[i] = data_v.contiguous().clone()
        # update cpsi for the next site
        cpsi = contract("lnr,irc->lnic", sites_tmp[i - 1], u[:, :, :d])
    # construct the first site
    shape = cpsi.shape
    assert shape[0] == 1
    cpsi = cpsi.reshape(shape[1], shape[2], shape[3])
    data = contract("nic->inc", cpsi) / torch.linalg.norm(cpsi)
    sites_tmp[0] = data.contiguous().clone()
    del sites
    return sites_tmp


def slstFromDM(sites: Sites, iroot=0) -> list[Tensor]:
    n_sites = len(sites)
    sites_tmp = copy.deepcopy(sites)
    slst: list[Tensor] = []
    for i in range(1, n_sites):
        if i == 1:
            site0 = sites_tmp[i - 1][iroot]
            shape = site0.shape
            site0 = site0.reshape(1, shape[0], shape[1])
        else:
            site0 = sites_tmp[i - 1]
        site1 = sites_tmp[i]
        psi = contract("lnr,rmx->lnmx", site0, site1)  # twodot wavefunction
        shape = psi.shape
        psi = psi.reshape(shape[0] * shape[1], shape[2] * shape[3])
        if dtype_config.device == 'cpu':
            u, s, vt = torch.linalg.svd(psi, full_matrices=False,)
        else:
            u, s, vt = torch.linalg.svd(psi, full_matrices=False, driver=dtype_config.driver)    # todo: test which driver is best.
        vt = contract("l,lr->lr", s, vt)
        sites_tmp[i - 1] = u.reshape(shape[0], shape[1], s.shape[0])
        sites_tmp[i] = vt.reshape(s.shape[0], shape[2], shape[3])
        slst.append(s**2)
    return slst


def RenyiEntropy(pop: Tensor, alpha: float) -> Tensor:
    if alpha == -1:
        return torch.sum(pop[len(pop) // 2 :])  # cutoff by half
    elif alpha == 1:
        # raise NotImplementedError
        return get_SvN(pop)
    else:
        return 1 / (1 - alpha) * torch.log(torch.sum(pop**alpha, axis=-1))

def ReyniEntropyFromSites(sites: Sites, alpha: float) -> Tensor:
    slst = slstFromDM(sites)
    return torch.stack([RenyiEntropy(slst[i], alpha) for i in range(len(slst))])

def sumOfReyniEntropyFromSites(sites: Sites, alpha: float) -> float:
    return torch.sum(ReyniEntropyFromSites(sites, alpha))

@torch.no_grad()
def tree_sample(
    sites: Sites,
    N: int,
    renorm_each_step: bool = False,
    mode: str = "floor_bernoulli",
):
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
    if not checkRCF(sites):
        sites = rightCanonicalization(sites)
    
    n = len(sites)
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
        A0 = Ai[:, 0, :]  # (Dl, Dr)
        A1 = Ai[:, 1, :]  # (Dl, Dr)

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


# to abab ordering
def to_abab(sites: Sites, thresh: float = 1.0e-14) -> Sites:
    """ "
    4-qubits: [0, 2, a, b] -> [0, b, a, 2]
    """
    nsite = len(sites)
    sites_new: Sites = [None] * (2 * nsite)
    for i in range(nsite):
        site = sites[i]  # lnr (n=0,2,a,b) => (n=0,b,a,2)
        site_new = np.zeros_like(site)
        site_new[:, 0, :] = site[:, 0, :]
        site_new[:, 1, :] = site[:, 3, :]
        site_new[:, 2, :] = site[:, 2, :]
        site_new[:, 3, :] = site[:, 1, :]
        shape = site_new.shape
        assert shape[1] == 4
        site_new = site_new.reshape(shape[0] * 2, 2 * shape[2])
        u, s, vt = scipy.linalg.svd(site_new, full_matrices=False)
        idx = np.argwhere(s > thresh).flatten()
        vtnew = vt[idx, :]
        wnew = contract("ij,j->ij", u[:, idx], s[idx])
        d = len(idx)
        sites_new[2 * i] = wnew.reshape(shape[0], 2, d)
        sites_new[2 * i + 1] = vtnew.reshape(d, 2, shape[2])
    return sites_new
