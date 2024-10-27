
Orbital optimization
####################

We can find less entangled orbitals by compressing entanglement.

.. code-block::

   $ctns
   verbose 2
   qkind rNS
   nroots 1
   maxdets 10000
   thresh_proj 1.e-15
   thresh_ortho 1.e-8
   topology_file topology/topo1
   schedule
   0 2 3 1.e-4 0.0
   end
   maxsweep 2
   alg_hvec 4
   alg_renorm 4
   task_oodmrg
   oo_maxiter 50
   fromconf uuudud
   #rcfprefix new_
   #rcanon_file oo_rcanon_iter1_su2
   #oo_urot urot_iter1
   $end
