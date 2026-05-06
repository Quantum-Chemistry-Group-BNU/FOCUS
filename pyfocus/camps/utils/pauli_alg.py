import numpy as np

from numba import njit
from numpy.typing import NDArray
from numpy import ndarray

from functools import reduce

from camps.utils.typing import PauliArray

PAULI_LOOKUP = np.array([b"I", b"Z", b"X", b"Y"], dtype="S1")


def vector_to_pauli(pauli_vc: NDArray[np.int64]) -> PauliArray:
    """
    pauli_vc: (nbatch, nqubits * 2) [x1, z1, x2, z2, ...]
    """
    nbatch, vec_len = pauli_vc.shape
    nqubit = vec_len // 2
    pauli_vc = pauli_vc.reshape(nbatch, nqubit, 2)

    key = pauli_vc[:, :, 0] * 2 + pauli_vc[:, :, 1]

    array = PAULI_LOOKUP[key]

    return array


@njit
def pauli_to_vector(pauli_arr: PauliArray) -> NDArray[np.int8]:
    """
    pauli_arr: (nbatch, nqubits)

    Return:
        out: (batch, nqubits * 2)
    """
    nbatch, nqubits = pauli_arr.shape
    out = np.zeros((nbatch, nqubits * 2), dtype=np.int8)

    for i in range(nbatch):
        for j in range(nqubits):
            c = pauli_arr[i, j][0]  # ASCII
            if c == ord("I"):
                x, z = 0, 0
            elif c == ord("X"):
                x, z = 1, 0
            elif c == ord("Z"):
                x, z = 0, 1
            elif c == ord("Y"):
                x, z = 1, 1
            else:
                raise ValueError(f"Unknown Pauli symbol: {chr(c)}")

            out[i, 2 * j] = x
            out[i, 2 * j + 1] = z

    return out


def _pauli_to_vector_numpy(string_arr: NDArray[np.bytes_]) -> NDArray[np.int8]:
    nbatch, nqubits = string_arr.shape
    v = np.zeros((nbatch, 2 * nqubits), dtype=np.int8)

    x_bits = ((string_arr == b"X") | (string_arr == b"Y")).astype(np.int8)
    z_bits = ((string_arr == b"Z") | (string_arr == b"Y")).astype(np.int8)

    v[:, 0::2] = x_bits
    v[:, 1::2] = z_bits

    return v


def pauli_to_matrix(string: list[str], big_endian: bool = False) -> list[NDArray[np.complex128]]:
    X_mat = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y_mat = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I_mat = np.array([[1, 0], [0, 1]], dtype=np.complex128)

    _dict = {"X": X_mat, "Y": Y_mat, "Z": Z_mat, "I": I_mat, "_": I_mat}

    if not isinstance(string, list):
        string = [string]
    result = []
    for str in string:
        str = str.upper().replace(" ", "")
        chars = str if big_endian else str[::-1]
        matrix = [_dict[i] for i in chars]
        result.append(reduce(np.kron, matrix))
    return result


# copy from https://github.com/hongyehu/PyClifford/blob/main/pyclifford/paulialg.py
@njit
def ipow(g1: ndarray, g2: ndarray) -> int:
    """Phase indicator for the product of two Pauli strings.

    Parameters:
    g1: int (2*N) - the first Pauli string in binary representation.
    g2: int (2*N) - the second Pauli string in binary representation.

    Returns:
    ipow: int - the phase indicator (power of i) when product
        sigma[g1] with sigma[g2]."""
    assert g1.shape == g2.shape
    (N2,) = g1.shape
    N = N2 // 2
    ipow = 0
    for i in range(N):
        g1x = g1[2 * i]
        g1z = g1[2 * i + 1]
        g2x = g2[2 * i]
        g2z = g2[2 * i + 1]
        gx = g1x + g2x
        gz = g1z + g2z
        ipow += g1z * g2x - g1x * g2z + 2 * ((gx // 2) * gz + gx * (gz // 2))
    return ipow % 4


@njit
def pauli_combine(C: ndarray, gs_in: ndarray, ps_in: ndarray) -> tuple[ndarray, ndarray]:
    """Combine Pauli operators by operator product.
        (left multiplication)

    Parameters:
    C: int (L_out, L_in) - one-hot encoding of selected operators.
    gs_in: int (L_in, 2*N) - input binary representation of Pauli strings.
    ps_in: int (L_in) - phase indicators of input operators.

    Returns:
    gs_out: int (L_out, 2*N) - output binary representation of Pauli strings.
    ps_out: int (L_out) - phase indicators of output operators.
    """
    (L_out, L_in) = C.shape
    N2 = gs_in.shape[-1]
    gs_out = np.zeros((L_out, N2), dtype=np.int64)  # identity
    ps_out = np.zeros((L_out,), dtype=np.int64)
    for j_out in range(L_out):
        for j_in in range(L_in):
            if C[j_out, j_in]:
                ps_out[j_out] = (ps_out[j_out] + ps_in[j_in] + ipow(gs_out[j_out], gs_in[j_in])) % 4
                gs_out[j_out] = (gs_out[j_out] + gs_in[j_in]) % 2
    return gs_out, ps_out


@njit
def ps0(gs) -> ndarray:
    """Bare phase factor due to x.z for Pauli strings.

    Parameters:
    gs: int (L,2*N) - array of Pauli strings in binary representation.

    Returns:
    ps0: int (L) - bare phase factor x.z for all strings."""
    # (L, N2) = gs.shape
    # N = N2//2
    # ps0 = np.zeros(L, dtype=np.int_)
    # for j in range(L):
    #     for i in range(N):
    #         ps0[j] += gs[j,2*i] * gs[j,2*i+1]
    # return ps0 % 4
    x = gs[:, 0::2]
    z = gs[:, 1::2]
    return np.sum(x * z, axis=1) % 4


def pauli_transform(
    gs_in: ndarray, gs_map: ndarray, ps_in: ndarray = None, ps_map: ndarray = None
) -> tuple[ndarray, ndarray]:
    """Transform Pauli operators by Clifford map.
        (right multiplication)

    Parameters:
    gs_in: int (L, 2*N) - input binary representation of Pauli strings.
    ps_in: int (L) - phase indicators of input operators.
    gs_map: int (2*N, 2*N) - operator map in binary representation.
    ps_map: int (2*N) - phase indicators associated to target operators.

    Returns:
    gs_out: int (L, 2*N) - output binary representation of Pauli strings.
    ps_out: int (L) - phase indicators of output operators."""
    # ps_map = np.array([0, 0, 0, 0]) # 2^(phase)
    # ps_in = np.array([0]) # 2^(phase)

    nqubit = gs_map.shape[0] // 2
    L = gs_in.shape[0]
    if ps_in is None:
        ps_in = np.zeros(L, dtype=np.int64)
    if ps_map is None:
        ps_map = np.zeros(2 * nqubit, dtype=np.int64)

    gs_out, ps_out = pauli_combine(gs_in, gs_map, ps_map)
    ps_out = (ps_in + ps0(gs_in) + ps_out) % 4
    return gs_out, ps_out


@njit
def z2inv(mat: NDArray[np.int64]):
    """Calculate Z2 inversion of a binary matrix."""
    assert mat.shape[0] == mat.shape[1]  # assuming matrix is square
    n = mat.shape[0]  # get matrix dimension
    a = np.zeros((n, 2 * n), dtype=mat.dtype)  # prepare a workspace
    a[:, :n] = mat  # copy matrix to the left part
    # create a diagonal matrix on the right part
    for i in range(n):
        a[i, i + n] = 1
    # forward pass
    for i in range(n):  # run through cols
        if a[i, i] == 0:  # need to find pivot
            found = False  # set a flag
            for k in range(i + 1, n):
                if a[k, i]:  # a[k, i] nonzero
                    found = True  # pivot found at k
                    break
            if found:  # if pivot found at k
                # swap rows i, k
                for j in range(i, 2 * n):
                    tmp = a[k, j]
                    a[k, j] = a[i, j]
                    a[i, j] = tmp
            else:  # if pivot not found, matrix not invertable
                raise ValueError("binary matrix not invertable.")
        # pivot has moved to a[i, i], perform GE
        for j in range(i + 1, n):
            if a[j, i]:  # a[j, i] nonzero
                a[j, i:] = (a[j, i:] + a[i, i:]) % 2
    # backward pass
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if a[j, i]:  # a[j, i] nonzero
                a[j, i:] = (a[j, i:] + a[i, i:]) % 2
    return a[:, n:]


def mapping_inverse(gs: NDArray, ps: NDArray = None):
    """
    Returns the inverse of this Clifford map, (such that it composes with
        its inverse results in identity map).
    Parameters:
    gs: int (2 * N, 2*N) - array of Pauli strings in binary repr (x0, z0, ..., xn, zn).
    ps: int (2 * N) - array of phase indicators (i powers).

    Return:
        gs_inv: int (2*N, 2*N)
        ps_inv: int (2*N)
    """
    gs_inv = z2inv(gs)
    if ps is None:
        ps = np.zeros(gs_inv.shape[0], dtype=np.int64)
    gs_iden, ps_mis = pauli_combine(gs_inv, gs, ps)
    ps_inv = (-ps_mis - ps0(gs_inv)) % 4

    return gs_inv, ps_inv
