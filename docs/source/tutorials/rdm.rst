
rdm.x
#####

Properties of MPS

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Convert MPS to binary format
============================

Use ``savebin`` to convert MPS to binary format, which can be later load using functions in ``pyutils``.

.. code-block::

   $ctns
   qkind rNSz
   nroots 2
   maxdets 100000
   thresh_proj 1.e-15
   thresh_ortho 1.e-8
   topology_file topology/topo1
   schedule
   0 2 60 1.e-5 0.0
   end
   maxsweep 0
   task_ham
   alg_renorm 4
   savebin
   #rcanon_file rcanon_ci
   rcanon2_file rcanon_ci2
   $end
  
Output:

.. code-block::

   ctns::comb_load fname=rcanon_ci2
   
   comb::display_shape qkind=qNSz
    idx=0 node=(5,0) shape(l,r,c)=(4,1,4)
    idx=1 node=(4,0) shape(l,r,c)=(16,4,4)
    idx=2 node=(3,0) shape(l,r,c)=(63,16,4)
    idx=3 node=(2,0) shape(l,r,c)=(32,63,4)
    idx=4 node=(1,0) shape(l,r,c)=(8,32,4)
    idx=5 node=(0,0) shape(l,r,c)=(2,8,4)
   qtensor2: wf2 own=1 _data=0x600002faa300
   qinfo2: wf2 sym=(0,0) dir=0,1
    qbond: qrow nsym=1 dimAll=2
    (6,0):2
    qbond: qcol nsym=1 dimAll=2
    (6,0):2
   total no. of nonzero blocks=1 nblocks=1 size=4:3.051758e-05MB
   
   ctns::rcanon_savebin fname=./scratch/rcanon_ci2

 
