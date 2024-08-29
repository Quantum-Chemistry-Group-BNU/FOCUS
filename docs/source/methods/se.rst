
Singlet embedding
#################

Basics
======

In spin-adapted DMRG, singlet embedding couples an fictious systems with the physical system such
that the total wavefunction is a singlet.

.. math::

   |\Psi\rangle = [|\Psi\rangle_{aux}\rangle \times |\Psi_{phys}\rangle]^0

This turns out to be very useful in DMRG and OO-DMRG, since many 6j or 9j coefficients get dramatically simplified.

Changes due to singlet embedding
================================

- ``init_phys.h``: get_left_bsite

- ``sweep_rcanon.h``: get_boundary_coupling for initial guess of wavefunction

- ``oper_env.h``: oper_init_dotSE for the leftmost dot (sadmrg/oper_dot_su2.h)

- ``sweep_onedot.h`` and ``sweep_twodot.h``: get_qsym_state change symmetry

- ``ctns_entropy.h``

