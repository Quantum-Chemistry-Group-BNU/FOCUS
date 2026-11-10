
pyfocus
#######

Some useful scripts 

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Load MPS without symmetry
=========================

.. code-block::

   from pyfocus import mps_nosym

   ne = mol. nelectron
   tm = 0
   qbonds, mps_sites, wfuns = mps_nosym.sweep_projection(sites2,ne, tm, debug=False)
   mps_nosym. dumpMPSforFOCUS (qbonds, rmps_sites, wfuns, prefix='./rmps', debug=True)

Load

.. code-block::

   $ctns
   qkind rNSz
   nroots 1
   maxdets 100
   thresh_proj 1.e-14
   thresh_ortho 1.e-8
   topology_file topo
   schedule
   0 2 10 1.e-4 1.e-8
   end
   maxsweep 4
   task_dmrg
   alg_hvec 4
   alg_renorm 4
   fromnosym
   $end

This can be used to compute overlap between an MPS from quantum circuit and the original uncompressed MPS.

