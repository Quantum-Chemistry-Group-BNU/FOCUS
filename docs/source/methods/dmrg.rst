
DMRG
####

Density Matrix Renormalization Group (DMRG) algorithm

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Overview
========

The density matrix renormalization group (DMRG) algorithm is an algorithm aiming to (approximately) find the low-lying eigenstate of a Hamiltonian

.. math::

   H\Psi = E\Psi

by a special low-rank approximation, called matrix product state (MPS), where the 
many-body wavefunction is approximated as

.. math::

   \Psi^{n_1n_2\cdots n_K} \approx \sum_{\alpha_k} A^{n_1}_{\alpha_1}[1]
   A^{n_2}_{\alpha_1\alpha_2}[2]\cdots 
   A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\cdots
   A^{n_K}_{\alpha_{K-1}}[K].

Here, :math:`K` is the number of spatial orbitals determined by
the system under study, :math:`n_k\in\{0,1,2,3\}`, 
and :math:`\alpha_k\in\{1,2,\cdots,D_k\}`. In practice,
we set a cutoff for :math:`D_k`, such that the maximal
value is :math:`D`, usually referred as the bond dimension,
which is given as an input for the DMRG algorithm.

The set of tensors :math:`\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\}`
is to be found by minimizing the energy function

.. math::

   E[\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\}]=
   \langle \Psi(\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\}|H|
   \Psi(\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\})\rangle

subject to the normalization condition :math:`\langle \Psi(\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\}|\Psi(\{A^{n_k}_{\alpha_{k-1}\alpha_k}[k]\})\rangle=1`.
Since the number of parameters in each tensor
:math:`A^{n_k}_{\alpha_{k-1}\alpha_k}[k]` is :math:`O(D^2)`,
the total of parameters scale as :math:`O(KD^2)`.
The set of tensors is found by a sequential optimization,
which means other tensors are kept fixed,
when :math:`A[k]` is being optimized.

Pseudocode for DMRG algorithm
=============================

.. code-block::

   Prepare an initial set of tensors [save into rcanon.info]
   
   DMRG sweep optimization [see ctns_sweep.h]
   for isweep=0, maxsweep [determined by schedule]
       for ibond=0, maxbond [fixed, =2(K-3)]
           [see sweep_onedot.h or sweep_twodot.h]
           step 1. load operators (from disk to memory)
           step 2. solve H[k]c=Ec using Davidson algorithm
           step 3. from the lowest eigenvector c update A[k] (e.g., via SVD truncated by D)
           step 4. update all operators and save to disk
   
   Save new tensors into rcanon_new.info

The update of :math:`A[k]` in step 3 is controlled by the given bond dimension.
Typically, we do not directly set it to the target value. To save the computational cost,
it is gradually increase for each sweep. So there is a schedule for the parameters.

.. code-block::

   isweep, dots, bond dimension, noise
   0 2  500 1.e-4 0.0
   2 2 1000 1.e-4 0.0
   5 2 3000 1.e-4 0.0
   ...

The design for a schedule is based on experience, as a trade-off between
computational cost and accuracy.

Formation of Hamiltonian-wavefunction multiplication
====================================================

The formation of Hamiltonian-wavefunction multiplication in step 2 is usually the bottleneck
in a DMRG calculation.

Formula generation via symbolic derivation
------------------------------------------
In FOCUS, we decompose :math:`H` via a recursive bipartition for :math:`H^{L\|C_1|C_2\|R}`, such that

.. math::

   H = \sum_{\mu}\hat{O}^L_\mu \hat{O}^C_\mu \hat{O}^R_\mu

and

.. math::
   H = \sum_{\mu}\hat{O}^L_\mu \hat{O}^{C_1}_\mu\hat{O}^{C_2}_\mu \hat{O}^R_\mu

for one-dot and two-dot optimization algorithms, respectively.
For quantum chemistry Hamiltonian, the number of terms (:math:`\mu`) is :math:`O(K^2)`.

Implementation of the application of Hamiltonian on trial wavefunctions
-----------------------------------------------------------------------

Considering the one-dot case for simplicity, applying the Hamiltonian on trial wavefunctions reads

.. math::

   \sigma^{l'c'r'}=\langle l'c'r'|\hat{O}^L\hat{O}^C\hat{O}^R|lcr\rangle\Psi^{lcr}=sgn*O^L_{l'l}O^C_{c'c}O^{R}_{r'r}\Psi^{lcr}

where :math:`sgn=(-1)^{p(\hat{O}^R)(p(l)+p(c))+p(\hat{O}^C)p(l)}` accounts for the
antisymmetric principle of fermions. Here, :math:`p(l)` is the parity of the left
renormalized state.

Actual implementation of the operators acting on wavefunctions takes a sequential multiplications

.. math::

   \sigma^{l'c'r'}
   =
   \sum_{l}O^L_{l'l}
   \left(\sum_{c}O^C_{c'c}(-1)^{p(O^C)p(l)}
   \left(\sum_{r}O^{R}_{r'r}(-1)^{p(O^R)(p(l)+p(c))}\Psi^{lcr}\right)
   \right)

The computational cost for this formula scales as :math:`O(D^3)`.
Thus, the computational cost for $H\psi$ scales as :math:`O(K^2D^3)`.
In total, the total computational cost per sweep is :math:`O(K^3D^3)`.


Data structure
==============

The central data structures are block-sparse tensors (see files in the directory src/ctns/qtensor)

* stensor2: operators such as :math:`O^L_{ll'}`
* stensor3: onedot wavefunction :math:`\psi^{lcr}`
* stensor4: twodot wavefunction :math:`\psi^{lc_1c_2r}`
* oper_dict: dictionary of operators: 

.. math::

   \texttt{lqops}=\{O^L\}=\{H^L,C_p^L,S_p^L,A_{pq}^L,B_{pq}^L,P_{pq}^L,Q_{pq}^L\}

   
Other topics
============

Example for formula
-------------------

.. code-block::

   idx=2 bipart_oper=Clc1_Sc2r[0] size(lop,rop)=1,8
    symbolic_task=lop : size=1
     i=0 {1*C[l](0)} # type[space](index)
    symbolic_task=rop : size=8
     i=0 {1*S[c2](0)}
     i=1 {1*S[r](0)}
     i=2 {0*C[c2](8) + 8.69e-06*C[c2](9)} * {1*Ad[r](10,11)}
     i=3 {1*Cd[c2](8)} * {1*Q[r](0,8)}
     i=4 {1*Cd[c2](9)} * {1*Q[r](0,9)}
     i=5 {1*Ad[c2](8,9)} * {0*C[r](10) + 1e-05*C[r](11)}
     i=6 {1*Q[c2](0,10)} * {1*Cd[r](10)}
     i=7 {1*Q[c2](0,11)} * {1*Cd[r](11)}

corresponds to

.. math::
   (C^L_0)_{lop}\cdot (S^{C_2}_0 + S^R_0 + \cdots)_{rop}


Storage of two-index operators
------------------------------

* :math:`A_{pq}` (:math:`p<q`) and :math:`A_{p\bar{q}}` (:math:`p\le q`)

* :math:`B_{ps}` (:math:`p\le q`) and :math:`B_{p\bar{s}}` (:math:`p\le q`)

