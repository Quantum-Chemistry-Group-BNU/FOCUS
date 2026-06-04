import numpy as np
from numpy.typing import NDArray
from renormalizer.model.basis import BasisHalfSpin
try:
    from renormalizer.model.basis import BasisTwoHalfSpin
except ImportError:
    BasisTwoHalfSpin = None
# from renormalizer.model.model import Model
# from renormalizer.mps.mpo import Mpo
from renormalizer.mps.mps import Mps
import openfermion.ops.representations as reps
from openfermion.transforms import (get_fermion_operator, 
                                    normal_ordered, 
                                    jordan_wigner,
                                    bravyi_kitaev,
                                    parity_code, 
                                    binary_code_transform, 
                                    reorder)
from openfermion.utils import up_then_down
from openfermion import QubitOperator
from camps.utils.typing import PauliArray, Sites, Hamiltonian
from camps.mps.mpo import construct_mpo_pauli
try:
    from pyscf_helper.operator import operators as operators_pyhp
except ImportError:
    operators_pyhp = None


def _number_ops_from_pyscf_helper(sorb: int, nelec: tuple):
    if operators_pyhp is None:
        raise ImportError(
            "pyscf_helper is required when penalty_nanb is enabled. "
            "Install pyscf_helper or pass penalty_nanb=None."
        )
    na, nb = nelec
    opss = operators_pyhp(sorb, False)
    ops_a, ops_b = opss["Na"], opss["Nb"]
    return na, nb, ops_a, ops_b

def get_fermion_ham(h1s, h2s, ecore):
    h2s = h2s.transpose(0, 1, 3, 2)  # <pq||rs> -> <pq||sr>
    second_q_hamiltonian = reps.InteractionOperator(ecore, h1s, 0.25 * h2s)
    fermion_H = normal_ordered(get_fermion_operator(second_q_hamiltonian)) #normal_ordered(get_fermion_operator(second_q_hamiltonian))
    return fermion_H

def get_parity_ham(fermion_H, sorb, reord = True):
    # fermion_H = get_fermion_operator(second_q_hamiltonian)
    if reord:
        fermion_H = reorder(fermion_H, up_then_down) 
    code_parity = parity_code(sorb)
    qubit_H_parity = binary_code_transform(fermion_H, code_parity)
    return qubit_H_parity

def Z2R4parity(parity_ham_aabb, nqubits, na, nb):
    n_spatial = nqubits // 2      # 空间轨道数 M
    n_alpha = na        # RHF: Nα = Nβ = N/2

    # 这两个才是我们真正要用的 Z2 对称性
    parity_alpha = n_alpha % 2    # (-1)^{Nα}
    parity_total = (na + nb) % 2       # (-1)^{N_tot = Nα+Nβ}

    # 对应的 parity qubit 索引：
    idx_alpha_parity = n_spatial - 1       # q_{M-1} = parity(α block)
    idx_total_parity = 2 * n_spatial - 1   # q_{2M-1} = parity(total)

    tapered = QubitOperator()

    for term, coeff in parity_ham_aabb.terms.items():
        new_coeff = complex(coeff)
        new_term_list = []

        # identity 项直接保留
        if len(term) == 0:
            tapered += QubitOperator((), new_coeff)
            continue

        for q, pauli in term:
            # 命中 α-parity qubit
            if q == idx_alpha_parity:
                if pauli == 'Z':
                    new_coeff *= (1 if parity_alpha == 0 else -1)
                else:
                    # X/Y 在固定本征态上 ⇒ 子空间里为 0
                    new_coeff = 0.0
                    break

            # 命中 total-parity qubit
            elif q == idx_total_parity:
                if pauli == 'Z':
                    new_coeff *= (1 if parity_total == 0 else -1)
                else:
                    new_coeff = 0.0
                    break

            else:
                # 对剩余 qubit 做 index 压缩：
                # 如果原 index 超过 alpha-parity，那删掉它时右侧 qubit 要左移 1
                # 如果原 index 再超过 total-parity，那再左移 1
                shift = (1 if q > idx_alpha_parity else 0) + \
                        (1 if q > idx_total_parity else 0)
                new_term_list.append((q - shift, pauli))

        if new_coeff != 0.0:
            new_term_tuple = tuple(sorted(new_term_list, key=lambda x: x[0]))
            tapered += QubitOperator(new_term_tuple, new_coeff)

    return tapered

def get_ham_from_qop(ham_qop, sorb):
    Ham_dict: dict[str, float] = {}
    for i, (pauli_prod, coeff) in enumerate(ham_qop.terms.items()):
        coeff = coeff * 1 + 0.0j
        string = ["I"] * sorb
        for j, (idx_path, pauli) in enumerate(pauli_prod):
            string[idx_path] = pauli
        Ham_dict["".join(string)] = coeff
    Ham_list = [[char for char in key] for key in Ham_dict.keys()]
    ops_coeff = np.asarray(list(Ham_dict.values()))
    ops_array: PauliArray = np.array(Ham_list, dtype="S")

    ops_dict = Hamiltonian(array=ops_array, coeff=ops_coeff)
    return ops_dict

def integral2pauli_JW(
    h1e: NDArray,
    h2e: NDArray,
    ecore: float,
    reord: bool,
    penalty_nanb: float = None,
    nelec: tuple = None
) -> Hamiltonian:
    """
    h1e: pq 2-rank NDarray
    h2e: <pq||rs>, 4-rank NDarray
    ecore: float
    """
    fermion_H = get_fermion_ham(h1e, h2e, ecore)
    sorb = h1e.shape[0]
    if penalty_nanb is not None:
        na, nb, ops_a, ops_b = _number_ops_from_pyscf_helper(sorb, nelec)
        Na_op = get_fermion_ham(ops_a[0], ops_a[1], 0.0)
        Nb_op = get_fermion_ham(ops_b[0], ops_b[1], 0.0)
        op_pn = ((Na_op-na)+(Nb_op-nb))**2
        fermion_H += penalty_nanb * op_pn
    if reord:
        fermion_H = reorder(fermion_H, up_then_down)
    jw_hamiltonian = jordan_wigner(fermion_H)
    ops_dict = get_ham_from_qop(jw_hamiltonian, sorb)
    return ops_dict

def integral2pauli_parity_abab(
    h1e: NDArray,
    h2e: NDArray,
    ecore: float,
    penalty_nanb: float = None,
    nelec: tuple = None
) -> Hamiltonian:
    fermion_H = get_fermion_ham(h1e, h2e, ecore)
    sorb = h1e.shape[0]
    if penalty_nanb is not None:
        na, nb, ops_a, ops_b = _number_ops_from_pyscf_helper(sorb, nelec)
        Na_op = get_fermion_ham(ops_a[0], ops_a[1], 0.0)
        Nb_op = get_fermion_ham(ops_b[0], ops_b[1], 0.0)
        op_pn = ((Na_op-na)+(Nb_op-nb))**2
        fermion_H += penalty_nanb * op_pn
    qubit_H_parity = get_parity_ham(fermion_H, sorb, False) # aabb
    ops_dict = get_ham_from_qop(qubit_H_parity, sorb)
    return ops_dict

def integral2pauli_parity_aabb(
    h1e: NDArray,
    h2e: NDArray,
    ecore: float,
    reduced: bool = True,
    penalty_nanb: float = None,
    nelec: tuple = (),
) -> Hamiltonian:
    fermion_H = get_fermion_ham(h1e, h2e, ecore)
    sorb = h1e.shape[0]
    if penalty_nanb is not None:
        na, nb, ops_a, ops_b = _number_ops_from_pyscf_helper(sorb, nelec)
        Na_op = get_fermion_ham(ops_a[0], ops_a[1], 0.0)
        Nb_op = get_fermion_ham(ops_b[0], ops_b[1], 0.0)
        op_pn = ((Na_op-na)+(Nb_op-nb))**2
        fermion_H += penalty_nanb * op_pn
    qubit_H_parity = get_parity_ham(fermion_H, sorb) # aabb
    if reduced:
        # print('?')
        na, nb = nelec
        qubit_H_parity = Z2R4parity(qubit_H_parity, sorb, na, nb)
        sorb = sorb-2
    ops_dict = get_ham_from_qop(qubit_H_parity, sorb)
    return ops_dict

def integral2pauli_BK(
    h1e: NDArray,
    h2e: NDArray,
    ecore: float,
    reord: bool,
    penalty_nanb: float = None,
    nelec: tuple = (),
) -> Hamiltonian:
    """
    h1e: pq 2-rank NDarray
    h2e: <pq||rs>, 4-rank NDarray
    ecore: float
    """
    fermion_H = get_fermion_ham(h1e, h2e, ecore)
    sorb = h1e.shape[0]
    if penalty_nanb is not None:
        na, nb, ops_a, ops_b = _number_ops_from_pyscf_helper(sorb, nelec)
        Na_op = get_fermion_ham(ops_a[0], ops_a[1], 0.0)
        Nb_op = get_fermion_ham(ops_b[0], ops_b[1], 0.0)
        op_pn = ((Na_op-na)+(Nb_op-nb))**2
        fermion_H += penalty_nanb * op_pn
    if reord:
        fermion_H = reorder(fermion_H, up_then_down)
    bk_hamiltonian = bravyi_kitaev(fermion_H)
    ops_dict = get_ham_from_qop(bk_hamiltonian, sorb)
    return ops_dict

def ops_expectation(
    ops_ham: Hamiltonian,
    sites: Sites,
    dcut: int,
    sorb: int,
    use_orb: bool = False,
):
    """
    ops_hams: Ham_array, Ham_coeff
    """

    if use_orb:
        if BasisTwoHalfSpin is None:
            raise ImportError(
                "The installed renormalizer package does not provide "
                "BasisTwoHalfSpin; use_orb=True is not supported in this "
                "environment."
            )
        # sigmaqn = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # # [0, b, a, ab]
        basis = [BasisTwoHalfSpin(i) for i in range(sorb // 2)]
        n_sites = sorb // 2
        n_dim = 4
    else:
        basis = [BasisHalfSpin(i) for i in range(sorb)]
        n_dim = 2
        n_sites = sorb

    assert len(sites) == n_sites
    assert sites[0].shape[1] == n_dim

    mpo_ops, model_ops = construct_mpo_pauli(ops_ham, basis, use_orb)
    mps_ops = Mps.random(model_ops, 0, dcut, percent=1.0)
    mps_ops.dtype = np.complex128
    mps_ops.optimize_config.method = "2site"

    for i in range(len(mps_ops)):
        shape = mps_ops[i].array.shape
        mps_ops[i].array = np.zeros(shape, dtype=np.complex128)

    # TODO:(zbwu-25-08-07) check shape
    for i, site in enumerate(sites):
        l0, m, r0 = site.shape
        l1, m, r1 = mps_ops[i].array.shape
        l = min(l0, l1)
        r = min(r0, r1)
        mps_ops[i].array[:l, :, :r] = site[:l, :, :r]
    e_ops = mps_ops.expectation(mpo_ops)
    return e_ops
