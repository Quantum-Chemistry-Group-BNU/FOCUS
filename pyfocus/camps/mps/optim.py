from loguru import logger
import numpy as np

from numpy.typing import NDArray

from renormalizer.model.basis import BasisSet
from renormalizer.mps.gs import optimize_mps
from renormalizer.mps.mpo import Mpo
from renormalizer.mps.mps import Mps

from camps.mps.disentangled import minimize_entropy_multiSweep, update_ham_multiSweep
from camps.mps.mpo import construct_mpo_pauli
from camps.mps.operators import ops_expectation
from camps.utils.typing import Clifford, SaveInfo, Sites, Hamiltonian

def optimize_mps_disentangle(
    hams_lst: list[Hamiltonian] | Hamiltonian,
    clifford: Clifford,
    basis: list[BasisSet],
    procedure0: list[tuple[int, float]],
    procedure1: list[tuple[int, float]],
    dmax: int,
    n_sites: int,
    sites_order: NDArray[np.int64],
    n_dim: int = 2,
    use_orb: bool = False,
    debug: bool = False,
    # model: Model = None,
    mps: Mps = None,
    mpo: Mpo = None,
    *,
    mps_is_optimized: bool = False,
    use_random_gates: bool = False,
    random_nums: int = 200,
    given_clifford: Clifford = None,
    alpha: int | float = 1,
):

    # assert len(hams_lst) == 2
    if not isinstance(hams_lst, (tuple, list)):
         hams_lst = [hams_lst]
    assert len(hams_lst) == 1 or 2
    assert isinstance(alpha, (int, float)) and alpha >= 1
    hams = hams_lst[0]
    ham_array = hams["array"]
    Ham_coeff = hams["coeff"]
    assert n_dim in (2, 4)

    if debug:
        if n_dim == 2:
            sites_order = np.array([0, 1])
        else:
            sites_order = np.array([0, 1, 2, 3])
    assert (n_dim == 4 and use_orb) or (n_dim == 2 and not use_orb)
    if mpo is None:
        logger.info(f"Start Construct MPO")
        mpo, model = construct_mpo_pauli(hams, basis, use_orb)
        logger.info(f"End Construct MPO")

    if mps is None:
        logger.info(f"random MPS")
        mps = Mps.random(model, 0, dmax, percent=1.0)
    else:
        logger.info(f"Copy MPS")
        mps = mps.copy()

    mps.dtype = np.complex128
    mps.optimize_config.procedure = procedure0
    mps.optimize_config.method = "2site"
    logger.info(f"Start procedure-one optimize mps")
    if not mps_is_optimized:
        energies_init, mps = optimize_mps(mps.copy(), mpo)
    else:
        energies_init = np.array([0.0])
    logger.info(f"End procedure-one optimize mps")
    e_old = mps.expectation(mpo)

    sites: Sites = [p.array[:, sites_order, :] for p in mps]
    sorb = len(sites) if not use_orb else 2 * len(sites)
    if len(hams_lst) > 1:
        ops_e_old = ops_expectation(hams_lst[1], sites, dmax, sorb,use_orb)

    if not debug:
        logger.info(f"dcut: {dmax}, alpha: {alpha}, dmax: {int(dmax * alpha)}")
        sites_new, gates_idx, ee_diff, random_clifford = minimize_entropy_multiSweep(
            sites=sites,
            dmax= int(dmax * alpha),
            clifford=clifford,
            microiter=10,
            n_dim=n_dim,
            use_random_gates=use_random_gates,
            random_endian=clifford.endian,
            random_nums=random_nums,
            given_clifford=given_clifford,
        )

        if use_random_gates:
            res = random_clifford
        else:
            res = clifford
        logger.info(f"gate: {gates_idx}")

        ham_array_new, sign = update_ham_multiSweep(
            Ham_array=ham_array,
            idx=gates_idx,
            clifford=res,
            n_sites=n_sites,
            n_dim=n_dim,
        )
        ham_coeff_new = Ham_coeff * sign
        logger.info(f"Start Construct MPO")
        hams_new = Hamiltonian(array=ham_array_new, coeff=ham_coeff_new)
        mpo_new, model_new = construct_mpo_pauli(hams_new, basis, use_orb)
        logger.info(f"End Construct MPO")

        #--------Testing Na/Nb ops----
        if len(hams_lst) > 1:
            ops_array = hams_lst[1]["array"]
            ops_coeff = hams_lst[1]["coeff"]
            ops_array_new, sign = update_ham_multiSweep(
                Ham_array=ops_array,
                idx=gates_idx,
                clifford=res,
                n_sites=n_sites,
                n_dim=n_dim,
            )
            ops_coeff_new = ops_coeff * sign
            logger.info(f"Start Construct MPO")
            ops_hams_new = Hamiltonian(array=ops_array_new, coeff=ops_coeff_new)

    else:
        ham_array_new = ham_array
        ham_coeff_new = Ham_coeff
        model_new = mps.model
        mpo_new = mpo
        ee_diff = 0.0

        #----Testing Na/Nb ops
        if len(hams_lst) > 1:
            ops_hams_new = hams_lst[1]


    if alpha > 1:
        # if dmax > dcut, MPS is random initialization
        mps_new = Mps.random(model_new, 0, dmax * alpha, percent=1.0)
        for i in range(len(mps)):
           # padding 1e-10
           _padding = np.random.rand(*mps_new[i].shape) * 1e-10
           mps_new[i].array = _padding.astype(sites_new[0].dtype) 
    else:
        mps_new: Mps = mps.copy()

    mps_new.model = model_new
    mps_new.dtype = np.complex128
    mps_new.optimize_config.procedure = procedure1
    mps_new.optimize_config.method = "2site"

    if not debug:
        idx = np.argsort(sites_order)
        # update sites
        for i, site in enumerate(sites_new):
            # assert mps_new[i].array.shape == sites_new[i].shape
            # mps_new[i].array = site.copy()[:, idx, :]
            l0, m, r0 = site.shape
            l1, m, r1 = mps_new[i].array.shape
            l = min(l0, l1)
            r = min(r0, r1)
            mps_new[i].array[:l, :, :r] = site[:l, idx, :r].copy()

    e_new = mps_new.expectation(mpo_new)
    logger.info(f"mps-old expectation: {e_old}")
    logger.info(f"mps-new expectation: {e_new}")
    logger.info(f"Start procedure-two optimize mps")
    energies_new, mps_new = optimize_mps(mps_new.copy(), mpo_new)
    logger.info(f"End procedure-two optimize mps")
    e_opt = mps_new.expectation(mpo_new)
    logger.info(f"mps-opt expectation: {e_opt}")

    e = (energies_init, energies_new)
    sites_last: Sites = [p.array for p in mps_new]

    if len(hams_lst) > 1:
        ops_e_new = ops_expectation(ops_hams_new, sites_new, dmax, sorb,use_orb)
        ops_e_opt = ops_expectation(ops_hams_new, sites_last, dmax, sorb,use_orb)
        logger.info(f"Ops-expectations")
        logger.info(f"ops-e-old: {ops_e_old}")
        logger.info(f"ops-e-new: {ops_e_new}")
        logger.info(f"ops-e-opt: {ops_e_opt}")

    if not debug:
        if use_random_gates:
            save_clifford = [res.to_dict() for res in random_clifford]
        else:
            save_clifford = None
        idx = np.argsort(sites_order)
        sites_new = [p[:, idx, :] for p in sites_new]  # notice order
        sites_dict = {"init": sites, "disentangle": sites_new, "last": sites_last}
        save_info = SaveInfo(
            hams=[hams, hams_new],
            clifford=save_clifford,
            sites_dict=sites_dict,
            energy=e,
            clifford_idx=gates_idx,
            random_clifford=use_random_gates,
        )
    else:
        sites_dict = {"init": sites, "disentangle": None, "last": sites_last}
        save_info = SaveInfo(
            hams=[hams],
            clifford=None,
            energy=e,
            clifford_idx=None,
            random_clifford=None,
        )

    hams = Hamiltonian(array=ham_array_new, coeff=ham_coeff_new)
    res = [hams]
    if len(hams_lst) > 1:
        if not debug:
            res.append(ops_hams_new)
        else:
            # not change ops-hams
            res.append(hams_lst[1])
    
    return e, mps_new, mpo_new, res, ee_diff, save_info
