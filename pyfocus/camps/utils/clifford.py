import itertools
import numpy as np
import random

from numpy.typing import NDArray

from pyfocus.camps.utils.pauli_alg import (
    pauli_to_matrix,
    pauli_to_vector,
    pauli_transform,
    vector_to_pauli,
)
from pyfocus.camps.utils.typing import Clifford, RandomClifford


def check_operator_endian(
    operator: NDArray,
    tableau: NDArray,
    n_qubits: int,
    tableau_phase: NDArray = None,
    *,
    endian="big",
):
    """
    check the operator is big or little endian
    tableau
    X1 -> + ZI
    X2 -> + IX
    Z1 -> + XI
    Z2 -> + IZ
    """

    assert n_qubits in (2, 4)
    endian = endian.lower()
    assert endian in ("big", "little")
    pauli_all_qubits = ["".join(p) for p in itertools.product(["I", "X", "Y", "Z"], repeat=n_qubits)]
    pauli_all_qubits = random.sample(pauli_all_qubits, min(20, len(pauli_all_qubits)))
    vector_all_qubits: list[np.ndarray] = []
    for s in pauli_all_qubits:
        s = np.array([p for p in s], dtype="S1").reshape(1, -1)
        v = pauli_to_vector(s)  # [x1, z1, x2, z2]
        v = np.concatenate([v[::2], v[1::2]])
        vector_all_qubits.append(v.reshape(-1))

    big_endian = True
    if endian == "little":
        big_endian = False

    if tableau_phase is None:
        tableau_phase = np.ones((len(tableau), n_qubits * 2), dtype=np.int64)

    flag = True
    if tableau[0].shape[0] == 8:
        new_order = [0, 4, 1, 5, 2, 6, 3, 7]
        assert n_qubits == 4
    elif tableau[0].shape[0] == 4:
        new_order = [0, 2, 1, 3]
        assert n_qubits == 2
    else:
        raise NotImplementedError
    for idx in range(len(operator)):
        M = tableau[idx][new_order]
        p = tableau_phase[idx][new_order]
        O = operator[idx]

        for i, v in enumerate(vector_all_qubits):
            pauli = pauli_all_qubits[i]
            mat1 = O @ (pauli_to_matrix(pauli, big_endian=big_endian)[0]) @ O.conj().T
            # print(mat1.shape, pauli_to_matrix(pauli)[0].shape)
            # # v: [x1, x2, z1, z2]
            # result = (v @ M) % 2  # [x1, z1, x2, z2]
            # s = vector_to_pauli(result)
            # print(pauli)
            pauli = np.array([p for p in pauli], dtype="S1").reshape(1, -1)
            gs_in = pauli_to_vector(pauli).reshape(1, -1)
            result, phase = pauli_transform(gs_in, M, ps_map=p, ps_in=None)
            s = vector_to_pauli(result).reshape(-1)

            # using tableau
            s = s.tobytes().decode()
            mat2 = pauli_to_matrix(s, big_endian=big_endian)[0]
            if phase[0] == 0:
                sign = 1
            else:
                sign = -1

            if not np.allclose(mat1, mat2 * sign):
                flag = False
                break  # break all-qubits
        if not flag:
            break  # break all-operator
    assert flag, f"the endian is not {endian}"
    print(f"the endian is {endian}")


def random_clifford(
    nums: int,
    n_qubits: int,
    seed: int = 42,
    add_I: bool = True,
    given_clifford: Clifford = None,
    *,
    endian: str = "big",
) -> RandomClifford:
    """
    random clifford from given clifford or all clifford
    """
    assert endian in ("big", "little")
    import stim

    stim.Tableau.random(seed)

    # idx = [0, 4, 1, 5, 2, 6, 3, 7]
    idx = []
    for i in range(n_qubits):
        idx.extend([i, i + n_qubits])
    operator_lst = []
    mapping_lst = []
    # TODO:(zbwu-25-08-01) check bug with the phases
    # phase_lst = []
    # tab_lst = []
    if add_I:
        I = np.eye(n_qubits**2)
        tableau = stim.Tableau.from_unitary_matrix(I, endian=endian)
        x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
        gs = np.block([[x2x, x2z], [z2x, z2z]])[:, idx] * 1
        # ps = np.block([x_signs, z_signs]) * 2
        tableau = stim.Tableau.from_numpy(x2x=x2x, x2z=x2z, z2x=z2x, z2z=z2z)

        matrix = tableau.to_unitary_matrix(endian=endian)
        operator_lst.append(matrix)
        mapping_lst.append(gs)
        # phase_lst.append(ps)

    if given_clifford is None:
        for i in range(nums):
            tableau = stim.Tableau.random(n_qubits)
            x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
            gs = np.block([[x2x, x2z], [z2x, z2z]])[:, idx] * 1
            # ps = np.block([x_signs, z_signs]) * 2
            tableau = stim.Tableau.from_numpy(x2x=x2x, x2z=x2z, z2x=z2x, z2z=z2z)
            matrix = tableau.to_unitary_matrix(endian=endian)
            operator_lst.append(matrix)
            mapping_lst.append(gs)
    else:
        all_nums = given_clifford.gates.shape[0]
        assert all_nums >= nums
        assert given_clifford.endian == endian
        index = np.random.choice(np.arange(all_nums), size=nums, replace=False)
        matrix = given_clifford.gates[index]  # (nums, qubits**2, qubits**2)
        gs = given_clifford.mapping[index]  # (nums, qubits * 2, qubits * 2)
        operator_lst.extend(matrix[i] for i in range(nums))
        mapping_lst.extend(gs[i] for i in range(nums))
    operator = np.stack(operator_lst, dtype=np.complex128)
    mapping = np.stack(mapping_lst, dtype=np.int64)
    phases = np.zeros((mapping.shape[0], n_qubits * 2), dtype=np.int64)

    info = RandomClifford(gates=operator, mapping=mapping, phases=phases, endian=endian, with_I=add_I)
    return info


def operator_endian_change(operator: NDArray, *, endian_old: str, endian_new: str) -> NDArray:
    assert endian_old in ("little", "big")
    assert endian_new in ("little", "big")
    assert endian_new != endian_old
    op_new = []
    import stim

    for idx in range(len(operator)):
        op1 = stim.Tableau.from_unitary_matrix(operator[idx], endian=endian_old)
        op = op1.to_unitary_matrix(endian=endian_new)
        op_new.append(op)
    return np.stack(op_new)
