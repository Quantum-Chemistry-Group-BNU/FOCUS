
ctns.x
######

Comb Tensor Network States (CTNS)

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Start from a DMRG calculation from a det
========================================

.. code-block::

   $ctns
   verbose 2
   qkind rNSz
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
   task_dmrg
   fromconf ababab
   $end
  
Sample configurations
=====================

Input:

.. code-block::

   $ctns
   verbose 1
   qkind rNSz
   nroots 1
   topology_file topology/topo0
   task_sdiag
   rcanon_file tmp
   saveconfs confs
   nsample 10000
   $end

Output:

.. code-block::

   ctns::rcanon_Sdiag_sample: ifab=1 iroot=0 nsample=10000 pthrd=-0.01 nprt=10 saveconfs=confs
   <CTNS[i]|CTNS[i]> = 1
    i=999 Sdiag=1.38561 std=0.0847186 IPR=0.66701 std=0.0100522 TIMING=0.444309 S
    i=1999 Sdiag=1.35197 std=0.0594285 IPR=0.673722 std=0.00698977 TIMING=0.448038 S
    i=2999 Sdiag=1.35679 std=0.0482542 IPR=0.671975 std=0.00573266 TIMING=0.445844 S
    i=3999 Sdiag=1.35257 std=0.0419221 IPR=0.673522 std=0.00494447 TIMING=0.438327 S
    i=4999 Sdiag=1.32478 std=0.0370398 IPR=0.675843 std=0.00439341 TIMING=0.43287 S
    i=5999 Sdiag=1.31787 std=0.0337102 IPR=0.676653 std=0.00400176 TIMING=0.433444 S
    i=6999 Sdiag=1.31739 std=0.0311726 IPR=0.676123 std=0.00370947 TIMING=0.433685 S
    i=7999 Sdiag=1.30401 std=0.0289804 IPR=0.677441 std=0.003457 TIMING=0.434131 S
    i=8999 Sdiag=1.30546 std=0.0273452 IPR=0.677432 std=0.00325961 TIMING=0.433249 S
    i=9999 Sdiag=1.30324 std=0.0259468 IPR=0.67815 std=0.0030865 TIMING=0.432594 S
   estimated Sdiag[MC]=1.30324 IPR[MC]=0.67815
   sampled unique csf/det: pop.size=378 psum=0.980751 Sdiag[pop]=1.2585 IPR[pop]=0.680458
    i=0 0000000aaa2222 counts=8238 p_i(sample)=0.8238 p_i(exact)=0.82085 c_i(exact)=0.906008
    i=1 0000000a2aa222 counts=380 p_i(sample)=0.038 p_i(exact)=0.0416167 c_i(exact)=0.204002
    i=2 000000a0a2a222 counts=151 p_i(sample)=0.0151 p_i(exact)=0.0145588 c_i(exact)=-0.12066
    i=3 00000002aa2a22 counts=40 p_i(sample)=0.004 p_i(exact)=0.00469544 c_i(exact)=-0.0685233
    i=4 00000a00a22a22 counts=46 p_i(sample)=0.0046 p_i(exact)=0.00462515 c_i(exact)=-0.0680084
    i=5 000000aabaa222 counts=32 p_i(sample)=0.0032 p_i(exact)=0.00321036 c_i(exact)=0.0566601
    i=6 000000aa022a22 counts=24 p_i(sample)=0.0024 p_i(exact)=0.00312546 c_i(exact)=0.0559058
    i=7 000000aa0a2222 counts=34 p_i(sample)=0.0034 p_i(exact)=0.00255267 c_i(exact)=0.050524
    i=8 00000a002aa222 counts=28 p_i(sample)=0.0028 p_i(exact)=0.00231436 c_i(exact)=0.0481078
    i=9 0000a000a222a2 counts=17 p_i(sample)=0.0017 p_i(exact)=0.00211367 c_i(exact)=-0.0459747
   accumulated counts for listed confs=9012 nsample=10000 per=0.9012
   save to file confs.txt
   
Configurations are saved in ``confs.txt``

.. code-block::

   size= 378 psum= 0.980751123663
   0000000aaa2222 0.906007955131
   0000000a2aa222 0.204001677820
   000000a0a2a222 -0.120659974486
   00000002aa2a22 -0.068523298585
   00000a00a22a22 -0.068008423928
   000000aabaa222 0.056660060832
   000000aa022a22 0.055905839494
   000000aa0a2222 0.050523970695
   00000a002aa222 0.048107802839
   0000a000a222a2 -0.045974720762
   000000a2a0a222 0.045323383669
   0000002aaa0222 -0.042999052956
   000000022aaa22 -0.042782423910
   00000a0a0222a2 0.041522375355
   000000aa202a22 -0.039362575047
   00000a0aba2a22 0.038099588217
   00000aa00a2222 -0.037358089526

