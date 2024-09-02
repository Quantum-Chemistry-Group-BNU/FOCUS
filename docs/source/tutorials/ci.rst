
CI
##

Configuration Interaction (CI)

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

exactdiag.x
===========

Input:

.. code-block::

   dtype 0
   nelec 5
   twom -1
   integral_file moleinfo/rmole.info
   scratch ./scratch
   
   $sci
   nroots 5
   schedule # fake
   0 1.e-5
   end
   cthrd 1.e-1
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

fci.x
=====

sci.x
=====

