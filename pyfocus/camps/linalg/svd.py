import warnings
import numpy as np
import scipy.linalg

from numpy.typing import NDArray

# copy from renormalizer/mps/svd_qn.py
def optimized_svd(a: NDArray, full_matrices: bool = True, opt_full_matrices: bool = True):
    # optimize performance when ``full_matrices = opt_full_matrices = True``
    # and the shape of ``a`` is extremely unbalanced
    # The idea is to construct only a limited number of orthogonal basis rather than all of them
    # (which are not necessary in most cases)
    m, n = a.shape
    if not full_matrices:
        opt_full_matrices = False

    # whether do the optimization
    # here 1/3 and 3 are only empirical
    opt = opt_full_matrices and not (1 / 3 < m / n < 3)

    # if opt, always set ``full_matrices=False``
    try:
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesdd",
        )
    except scipy.linalg.LinAlgError:
        warnings.warn("SVD failed to converge", category=RuntimeWarning)
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesvd",
        )
    if not opt:
        return U, S, Vt

    # if opt, add n additional basis assuming  2 * n < m
    if m < n:
        Vt = add_orthonormal_basis(Vt.T).T
    elif n < m:
        U = add_orthonormal_basis(U)
    else:
        assert False
    return U, S, Vt


def add_orthonormal_basis(u):
    # add `n` basis. `n` is empirical
    m, n = u.shape
    assert 2 * n < m
    assert np.allclose(u.T.conj() @ u, np.eye(n))
    a = np.random.rand(m, n)
    a = a - u @ (u.T.conj() @ a)
    q, _ = scipy.linalg.qr(a, mode="economic")
    res = np.concatenate([u, q], axis=1)

    assert np.allclose(res.T.conj() @ res, np.eye(2 * n))
    return res
