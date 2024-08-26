
CTNS
####

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
   
