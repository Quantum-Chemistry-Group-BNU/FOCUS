import numpy as np

from numpy.typing import NDArray


def spinorb_from_spatial(int1e, int2e):
    """
    int2e: chem notation -> h1e, h2e: <pq||rs>
    """
    sbas = 2 * int1e.shape[0]
    h1e = np.zeros((sbas, sbas))
    h1e[0::2, 0::2] = int1e  # AA
    h1e[1::2, 1::2] = int1e  # BB
    h2e = np.zeros((sbas, sbas, sbas, sbas))
    h2e[0::2, 0::2, 0::2, 0::2] = int2e  # AAAA
    h2e[1::2, 1::2, 1::2, 1::2] = int2e  # BBBB
    h2e[0::2, 0::2, 1::2, 1::2] = int2e  # AABB
    h2e[1::2, 1::2, 0::2, 0::2] = int2e  # BBAA
    h2e = h2e.transpose(0, 2, 1, 3)  # <ij|kl> = [ik|jl]
    h2e = h2e - h2e.transpose(0, 1, 3, 2)  # Antisymmetrize V[pqrs]=<pq||rs>
    # h2e = h2e.transpose(0, 1, 3, 2) # change <pq||rs> -> <pq||sr>
    # second_q_hamiltonian = reps.InteractionOperator(ecore.item(), h1e, 0.25* h2e)
    # jw_hamiltonian = jordan_wigner(second_q_hamiltonian)
    return h1e, h2e
