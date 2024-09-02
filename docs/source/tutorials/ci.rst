
CI
##

Configuration Interaction (CI)

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

exactdiag.x
===========

This will use functions in ``core`` to construct the Hamiltonian matrix using Slater-Condon rule and then diagonalize. It is just for benchmark purpose.

Input:

.. code-block::

   dtype 0
   nelec 5
   twom -1
   integral_file moleinfo/rmole.info
   scratch ./scratch
   
   $ci
   nroots 5
   schedule # fake
   0 1.e-5
   end
   cthrd 1.e-1
   rdm 0 0
   $end

Output:

.. code-block::
   
   Generate Hilbert space for (k,n)=12,5 dim=792
   fock::get_Hmat dim=792
   check ||H-H.dagger||=1.686e-14
   
   summary of FCI energies:
    state 0 energy = -2.895651558950
    state 1 energy = -2.895651558950
    state 2 energy = -2.743907667892
    state 3 energy = -2.743907667892
    state 4 energy = -2.631676155198
   
   state 0 energy = -2.895651558950
   fock::coeff_population dim=792 thresh=0.100000000000 iop=0
     i-th  /  idx  /  coeff  /  pop  /  rank  /  onstate  /  nelec
          0 :        1   +9.579e-0191.749  0  000000101111 (5)
          1 :       23   +1.318e-01 1.738  2  000010011011 (5)
          2 :       12   -1.282e-01 1.643  1  000001101011 (5)
   psum=0.951302 psum0=1.000000 Sd=0.525775
   <Ne>=5.000000 std=0.000000
   fock::coeff_analysis dim=792
    |c[i]| in 10^+0-10^-1 : pop=0.951 accum=0.951 counts=3
    |c[i]| in 10^-1-10^-2 : pop=0.047 accum=0.999 counts=43
    |c[i]| in 10^-2-10^-3 : pop=0.001 accum=1.000 counts=80
    |c[i]| in 10^-3-10^-4 : pop=0.000 accum=1.000 counts=22
    |c[i]| in 10^-4-10^-5 : pop=0.000 accum=1.000 counts=2

   ...

   fock:get_rdm12 k=12
   
   fock:get_rdm1 rdm1.shape=12,12
   ----- TIMING FOR fock:get_rdm1 : 1.012e-02 S -----
   
   fock:get_rdm2 rdm2.shape=66,66
   ----- TIMING FOR fock:get_rdm2 : 8.292e-02 S -----
   
   Check: I,J=0,0 H(I,J)=-2.895651558950
   save matrix into fname = rdm1.0.0.txt
   save matrix into fname = rdm2.0.0.txt
   ----- TIMING FOR fock:get_rdm12 : 9.479e-02 S -----
   
fci.x
=====

This will use ``fci`` related functions in ``ci`` to perform iterative diagonalization

Input: (some keywords like ``dets``, ``schedule``, ``maxiter`` work for SCI)

.. code-block::

   dtype 0
   nelec 6
   twom 0
   integral_file moleinfo/fmole.info
   scratch ./scratch
   
   $ci
   dets
   0 1 2 3 4 5
   end
   checkms
   nroots 2
   schedule
   0 1.e-5
   end
   maxiter 3
   rdm 0 0
   $end

Output:

.. code-block::

   ...

      48   0 -      -3.236066279760 -6.459e-11  1.088e-05    3    83  3.200e-04
      48   1 +      -3.062519335925  0.000e+00  7.678e-06    3    83  3.200e-04
      49   0 +      -3.236066279804 -4.382e-11  7.750e-06    3    84  3.220e-04
      49   1 +      -3.062519335925  8.882e-16  7.678e-06    3    84  3.220e-04
   ----- TIMING FOR solve_iter : 2.774e-02 S -----
   ----- TIMING FOR fci::ci_solver : 9.664e-02 S -----
   
   fci::ci_save fname=./scratch/ci.info
   
   sparse_hamiltonian::dump fname=./scratch/sparseH.bin
   
   fci:get_rdm12 Htype=0 k=12
   
   sparse_hamiltonian::get_hamiltonian dim0=0 dim=400
   ----- TIMING FOR pspace : 2.890e-04 S -----
   coupling_table::get_Cmn : dim=20 timing : 5.7e-05 s
   coupling_table::get_Cmn : dim=20 timing : 5.2e-05 s
   ----- TIMING FOR ctabA : 6.300e-05 S -----
   ----- TIMING FOR ctabB : 5.400e-05 S -----
   sparse_hamiltonian::make_hamiltonian
   ----- TIMING FOR diagonal : 3.230e-04 S -----
   ----- TIMING FOR get_HIJ_A1122_B00 : 1.830e-03 S -----
   ----- TIMING FOR get_HIJ_A00_B1122 : 1.533e-03 S -----
   ----- TIMING FOR get_HIJ_A11_B11 : 8.109e-03 S -----
   ----- TIMING FOR sparse_hamiltonian::get_hamiltonian : 1.182e-02 S -----
   ----- TIMING FOR get_hamiltonian : 1.224e-02 S -----
   
   fci:get_rdm1 rdm1.shape=12,12
   coupling_table::get_Cmn : dim=20 timing : 4.3e-05 s
   coupling_table::get_Cmn : dim=20 timing : 5.1e-05 s
   tr(rdm1)=6 normalized to N
   ----- TIMING FOR fci:get_rdm1 : 2.182e-03 S -----
   
   fci:get_rdm2 rdm2.shape=66,66
   tr(rdm2)=30 normalized to N(N-1)
   ----- TIMING FOR fci:get_rdm2 : 2.260e-03 S -----
   
   Check: I,J=0,0 H(I,J)=-3.236066279804
   save matrix into fname = rdm1.0.0.txt
   save matrix into fname = rdm2.0.0.txt
   ----- TIMING FOR fci:get_rdm12 : 1.782e-02 S -----
   
sci.x
=====

This will use ``sci`` related functions in ``ci`` to perform iterative diagonalization with selection

With the same input as that for ``fci.x``, performing SCI calculation leads to the following output:

.. code-block::
   
      43   0 +      -3.236066267492  4.441e-16  9.781e-05    3    76  3.190e-04
      43   1 -      -3.062519306544 -6.158e-09  1.038e-04    3    76  3.190e-04
      44   0 +      -3.236066267492  0.000e+00  9.781e-05    3    77  3.200e-04
      44   1 +      -3.062519311638 -5.094e-09  9.042e-05    3    77  3.200e-04
   
   sci: iter=2 eps1=1.000e-05 nsub=399 i=0 e=-3.23606626749 de=-8.934e-03 conv=0 SvN=4.917e+00
   fock::coeff_analysis dim=399
    |c[i]| in 10^+0-10^-1 : pop=0.434 accum=0.434 counts=24
    |c[i]| in 10^-1-10^-2 : pop=0.564 accum=0.997 counts=264
    |c[i]| in 10^-2-10^-3 : pop=0.003 accum=1.000 counts=90
    |c[i]| in 10^-3-10^-4 : pop=0.000 accum=1.000 counts=20
    |c[i]| in 10^-5-10^-6 : pop=0.000 accum=1.000 counts=1
   
   sci: iter=2 eps1=1.000e-05 nsub=399 i=1 e=-3.06251931164 de=-4.373e-03 conv=0 SvN=4.695e+00
   fock::coeff_analysis dim=399
    |c[i]| in 10^+0-10^-1 : pop=0.496 accum=0.496 counts=22
    |c[i]| in 10^-1-10^-2 : pop=0.502 accum=0.998 counts=256
    |c[i]| in 10^-2-10^-3 : pop=0.002 accum=1.000 counts=78
    |c[i]| in 10^-3-10^-4 : pop=0.000 accum=1.000 counts=24
   
   sci convergence failure: out of maxiter=3 for threshsold deltaE=1.000e-10
   
   ----- TIMING FOR sci::ci_solver : 1.378e-01 S -----
   
   fci::ci_save fname=./scratch/ci.info
   
   fci:get_rdm12 Htype=0 k=12
   
   sparse_hamiltonian::get_hamiltonian dim0=0 dim=399
   ----- TIMING FOR pspace : 2.950e-04 S -----
   coupling_table::get_Cmn : dim=20 timing : 5.5e-05 s
   coupling_table::get_Cmn : dim=20 timing : 4.9e-05 s
   ----- TIMING FOR ctabA : 6.100e-05 S -----
   ----- TIMING FOR ctabB : 5.100e-05 S -----
   sparse_hamiltonian::make_hamiltonian
   ----- TIMING FOR diagonal : 3.190e-04 S -----
   ----- TIMING FOR get_HIJ_A1122_B00 : 1.827e-03 S -----
   ----- TIMING FOR get_HIJ_A00_B1122 : 1.407e-03 S -----
   ----- TIMING FOR get_HIJ_A11_B11 : 8.088e-03 S -----
   ----- TIMING FOR sparse_hamiltonian::get_hamiltonian : 1.167e-02 S -----
   ----- TIMING FOR get_hamiltonian : 1.208e-02 S -----
   
   fci:get_rdm1 rdm1.shape=12,12
   coupling_table::get_Cmn : dim=20 timing : 3.9e-05 s
   coupling_table::get_Cmn : dim=20 timing : 5e-05 s
   tr(rdm1)=6 normalized to N
   ----- TIMING FOR fci:get_rdm1 : 2.180e-03 S -----
   
   fci:get_rdm2 rdm2.shape=66,66
   tr(rdm2)=30 normalized to N(N-1)
   ----- TIMING FOR fci:get_rdm2 : 2.257e-03 S -----
   
   Check: I,J=0,0 H(I,J)=-3.236066267492
   save matrix into fname = rdm1.0.0.txt
   save matrix into fname = rdm2.0.0.txt
   ----- TIMING FOR fci:get_rdm12 : 1.769e-02 S -----
   
