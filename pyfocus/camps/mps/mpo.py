import numpy as np
import openfermion.ops.representations as reps

from numpy.typing import NDArray
from renormalizer.model.basis import BasisSet
from renormalizer.model.model import Model
from renormalizer.model.op import Op, OpSum

from renormalizer.mps.mpo import Mpo
from camps.utils.typing import Hamiltonian


# copy from renormalizer
def spinorb2spatorb(old_op: Op) -> Op:
    split_symbol, dofs, qn_list = old_op.split_symbol, old_op.dofs, old_op.qn_list
    new_split_symbol, new_dofs = [], []
    for i in range(len(split_symbol)):
        new_split_symbol.append(split_symbol[i] + str(dofs[i] % 2))
        new_dofs.append(dofs[i] // 2)
    new_symbols = " ".join(new_split_symbol)
    op = Op(new_symbols, new_dofs, factor=old_op.factor, qn=qn_list)
    return op


def construct_mpo_pauli(
    ham: Hamiltonian,
    basis: list[BasisSet],
    spatial_orb: bool = False,
) -> tuple[Mpo, Model]:
    Ham_op: list[Op] = []

    array = ham["array"]
    coeff = ham["coeff"]
    n_ham = array.shape[0]
    for i in range(n_ham):
        string = array[i].tobytes().decode()
        idx = [j for j in range(len(string))]
        op = Op(" ".join(string), idx, coeff[i] * 1 + 0j)
        if spatial_orb:
            op = spinorb2spatorb(op)
        Ham_op.append(op)
    ham_terms = OpSum(Ham_op)

    model = Model(basis, ham_terms)
    mpo = Mpo(model)
    return mpo, model
