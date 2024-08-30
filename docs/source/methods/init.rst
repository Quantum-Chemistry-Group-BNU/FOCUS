
Initialization
##############

Basics
======

MPS can be initialized in different ways. One way is from a SCI wavefunction.

Canonicalization
----------------

Canonicalization for spin-adapted MPS is always carried out with the singlet embedding.

Final orthonormalization
------------------------

Orthonormalization of ``rwfuns`` used in the function ``updateRWFuns`` in ``sadmrg/ctns_tosu2_update.h``:

.. math::
   |\psi_i\rangle = \sum_{n}|n\rangle W_{in}

then using the SVD for :math:`\mathbf{W}=\mathbf{U\Lambda V}^\dagger`:

.. math::

   \langle\psi_i|\psi_j\rangle = \sum_m W_{im}^* W_{jm} = (\mathbf{W}^*\mathbf{W}^T)_{ij} = (\mathbf{U}^*\mathbf{\Lambda}^2\mathbf{U}^T)_{ij}

The Lowdin orthonormalization suggests to define

.. math::
   |\tilde{\psi}_i\rangle = \sum_{j}|\psi_j\rangle X_{ji},\quad \mathbf{X}=\mathbf{U}^*\mathbf{\Lambda}^{-1}\mathbf{U}^T

such that

.. math::
   |\tilde{\psi}_i\rangle = \sum_{n}|n\rangle W_{jn} X_{ji} = \sum_n |n\rangle (\mathbf{X}^T\mathbf{W})_{in} = \sum_n |n\rangle (\mathbf{UV}^\dagger)_{in}

That is, the new coefficient can be simply obtained from :math:`\mathbf{UV}^\dagger`.

Final orthonormalization in KA-DMRG
-----------------------------------

In case of KA-DMRG, ``rwfuns`` is obtained via diagonalization of the Kramers symmetry projected density matrix.

