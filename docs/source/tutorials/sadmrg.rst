
sadmrg.x
########

Spin-Adapted Density Matrix Renormalization Group (SA-DMRG)

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Convert a nonsu2 MPS to a su2 MPS
=================================

.. code-block::
   :emphasize-lines: 3,4,5,14,15

   $ctns
   verbose 1
   qkind rNS
   tosu2
   thresh_tosu2 1.e-8
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 1000 1.e-4 0.e0
   end
   maxsweep 80
   task_init
   rcanon_file d100
   $end

Suppose an MPS exists in ``scratch/d100.info``. Use the command

.. code-block::

   sadmrg.x input.dat > output

will convert it to a spin-adapted MPS saved in ``scratch/d100_su2.info``.


Convert a CSF to a su2 mps and nonsu2 mps 
=========================================

This works also for open-shell case.

.. code-block::
   :emphasize-lines: 3,4,19,20
   
   dtype 0
   nelec 113
   twoms 3
   twos 3
   integral_file moleinfo/fmole.info
   scratch ./scratch
   
   $ctns
   verbose 1
   qkind rNS
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 1000 1.e-4 0.e0
   end
   maxsweep 80
   fromconf 222000u222222222u2uduuuuud22dddd2u22u22222222duu2uu2uuuddud22222222222ddddd2
   task_tononsu2
   $end
  
This will generate ``rcanon_csf.info`` and ``rcanon_csf_nonsu2.info`` in scratch.

Convert a CSF into a nonsu2 mps, and then sample dets
=====================================================

Input:

.. code-block::

   $ctns
   verbose 1
   qkind rNS
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 1 1.e-4 0.e0
   end
   maxsweep 1
   alg_hvec 4
   alg_renorm 4
   fromconf 222000u222222222u2duuuuuu2dd2ddd2u2u222222222duu2uu2uuuddud2222222222d2ddd2d
   task_tononsu2
   nsample 10000
   pthrd 1.e-2
   $end
   
Output:

.. code-block::
   :emphasize-lines: 25,26,27,28,29,30

   comb::display_shape qkind=qNS
    idx=0 node=(75,0) shape(l,r,c)=(3,1,3) shapeU1(l,r,c)=(4,1,4)
    …
   idx=75 node=(0,0) shape(l,r,c)=(1,1,3) shapeU1(l,r,c)=(4,5,4)
   qtensor2: wf2 own=1 _data=0x600002cd8820
   qinfo2su2: wf2 sym=(0,0) dir=0,1
    qbond: qrow nsym=1 dimAll=1
    (113,3):1
    qbond: qcol nsym=1 dimAll=1
    (113,3):1
   total no. of nonzero blocks=1 nblocks=1 size=1:7.63e-06MB
   
   ctns::rcanon_Sdiag_sample: ifab=1 iroot=0 nsample=10000 pthrd=-0.01
   <CTNS[i]|CTNS[i]> = 1
    i=999 Sdiag=12.6399 std=0.170818 IPR=0.00516701 timing=2.87926 s
    i=1999 Sdiag=12.8018 std=0.1213 IPR=0.0049276 timing=2.19442 s
    i=2999 Sdiag=12.8562 std=0.0993659 IPR=0.00494432 timing=2.17673 s
    i=3999 Sdiag=12.7986 std=0.0864741 IPR=0.00521172 timing=2.19415 s
    i=4999 Sdiag=12.7751 std=0.0773553 IPR=0.00508331 timing=2.17008 s
    i=5999 Sdiag=12.7624 std=0.0705867 IPR=0.00510909 timing=2.18695 s
    i=6999 Sdiag=12.7693 std=0.065721 IPR=0.00523013 timing=2.20559 s
    i=7999 Sdiag=12.7902 std=0.0616358 IPR=0.0052288 timing=2.16293 s
    i=8999 Sdiag=12.7866 std=0.0580374 IPR=0.00521176 timing=2.1962 s
    i=9999 Sdiag=12.8091 std=0.0550559 IPR=0.0051644 timing=2.15708 s
   sampled important csf/det: pop.size=7355
    i=0 222000a222222222a2baaaaaa2bb2bbb2a2a222222222baa2aa2aaabbab2222222222b2bbb2b counts=682 p_i(sample)=0.0682 p_i(exact)=0.0654545 c_i(exact)=-0.255841
    i=1 222000b222222222a2aaaaaaa2bb2bbb2a2a222222222baa2aa2aaabbab2222222222b2bbb2b counts=168 p_i(sample)=0.0168 p_i(exact)=0.0163636 c_i(exact)=0.12792
    i=2 222000a222222222b2aaaaaaa2bb2bbb2a2a222222222baa2aa2aaabbab2222222222b2bbb2b counts=166 p_i(sample)=0.0166 p_i(exact)=0.0163636 c_i(exact)=0.12792
    i=3 222000a222222222a2baaaaaa2bb2bbb2a2b222222222aaa2aa2aaabbab2222222222b2bbb2b counts=46 p_i(sample)=0.0046 p_i(exact)=0.00409091 c_i(exact)=0.0639602
    i=4 222000a222222222a2baaaaaa2bb2bbb2b2a222222222aaa2aa2aaabbab2222222222b2bbb2b counts=45 p_i(sample)=0.0045 p_i(exact)=0.00409091 c_i(exact)=0.0639602
   accumulated counts=10000 nsample=10000 per=1
   estimated Sdiag[MC]=12.8091 Sdiag[pop]=8.22603
   estimated IPR[MC]=0.0051644 IPR[pop]=0.00545804
   
Expand a CSF to dets
====================

Input: using keywork ``task_expand``. This works only for small systems.

.. code-block::

   $ctns
   verbose 2
   qkind rNS
   nroots 1
   maxdets 1
   thresh_proj 1.e-15
   thresh_ortho 1.e-8
   topology_file topology/topo1
   schedule
   0 2 1 1.e-5 0.0
   end
   maxsweep 4
   alg_hvec 4
   alg_renorm 4
   task_expand
   fromconf ududud
   $end
  
Output:

.. code-block::
   :emphasize-lines: 14
         
   ctns::rcanon_Sdiag_exact(su2): ifab=0 iroot=0 type=csf pthrd=0.01
   ctns::rcanon_expand_csfspace: ifab=0 iroot=0 pthrd=0.01
   ctns::get_csfspace: ifab=0 iroot=0
   fock::get_csf_space (k,n,ts)=6,6,0
   ks=6 sym=(6,0) dimFCI[csf]=175
   ovlp=1
    i=0 idx=95 state=ududud pop=1 coeff=1
   dim=175 ovlp=1 Sdiag(exact)=0 IPR=1
   
   ctns::rcanon_Sdiag_exact(su2): ifab=0 iroot=0 type=det pthrd=0.01
   ctns::rcanon_expand_onspace: ifab=0 iroot=0 pthrd=0.01
   fock::get_csf_space (k,n,ts)=6,6,0
   ovlp=1
    i=0 idx=114 state=bababa pop=0.125 coeff=0.353553
    i=1 idx=133 state=babaab pop=0.125 coeff=-0.353553
    i=2 idx=152 state=baabba pop=0.125 coeff=-0.353553
    i=3 idx=171 state=baabab pop=0.125 coeff=0.353553
    i=4 idx=228 state=abbaba pop=0.125 coeff=-0.353553
    i=5 idx=247 state=abbaab pop=0.125 coeff=0.353553
    i=6 idx=266 state=ababba pop=0.125 coeff=0.353553
    i=7 idx=285 state=ababab pop=0.125 coeff=-0.353553
   dim=400 ovlp=1 Sdiag(exact)=2.07944 IPR=0.125

Directly sample dets from a spin-adapted MPS
============================================

just add a keyword ``detbasis``

Input:

.. code-block::

   $ctns
   verbose 1
   qkind rNS
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 1 1.e-4 0.e0
   end
   maxsweep 1
   alg_hvec 4
   alg_renorm 4
   fromconf 222000u222222222u2duuuuuu2dd2ddd2u2u222222222duu2uu2uuuddud2222222222d2ddd2d
   task_sdiag
   nsample 10000
   detbasis
   $end

Compute diagonal element of Hamiltonian for a CSF
=================================================

.. code-block::

   $ctns
   verbose 1
   qkind rNS
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 10 1.e-4 0.e0
   end
   maxsweep 1
   task_ham
   alg_hvec 4
   alg_renorm 4
   fromconf 222000u222222222u2uduuuuud22dddd2u22u22222222duu2uu2uuuddud22222222222ddddd2
   $end  
  
Start SA-DMRG with an input CSF
===============================

This can work with singlet embedding.

.. code-block::

   $ctns
   verbose 1
   qkind rNS
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topoA
   thresh_proj 1.e-16
   thresh_ortho 5.e-8
   schedule
   0  2 1 1.e-4 0.e0
   end
   maxsweep 1
   task_dmrg
   alg_hvec 4
   alg_renorm 4
   fromconf 222000u222222222u2duuuuuu2dd2ddd2u2u222222222duu2uu2uuuddud2222222222d2ddd2d
   singlet
   $end
 
Start OO-DMRG from a CSF
========================

Input:

.. code-block::

   $ctns
   verbose 2
   qkind rNS
   nroots 1
   maxdets 1
   thresh_proj 1.e-15
   thresh_ortho 1.e-8
   topology_file topology/topo1
   schedule
   0 2 5 1.e-4 0.0
   end
   maxsweep 4
   alg_hvec 4
   alg_renorm 4
   fromconf ududud
   task_oodmrg
   oo_maxiter 5
   $end
   
