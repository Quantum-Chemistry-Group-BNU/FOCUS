
Basics and keywords
###################

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Basic usages
************

* use ``fci.x`` or ``exactdiag.x`` only for small systems [for benchmark]

* use ``sci.x`` to generate initial guess

* use ``ctns.x`` to optimize CTNS

* use ``sadmrg.x`` to optimize MPS with SU(2) symmetry

* use ``rdm.x`` to savebin / compute RDMs (slow)

A typical input file
********************

.. code-block::

   dtype 0
   nelec 30
   twom 0
   twos 0
   integral_file moleinfo/fmole.info
   scratch ./scratch
   perfcomm 27

   $ci
   dets
   0 2 4 6 8 10 12 14 16 18 20 22 24 36 38   1 3 15 17 19 21 23 25 27 29 31 33 35 37 39
   end
   checkms
   flip
   nroots 2
   schedule
   0 1.e-3
   #3 5.e-4
   #6 1.e-4
   #9 5.e-5
   end
   maxiter 3
   $end

   $ctns
   verbose 1
   qkind rNSz
   maxdets 1000 # keep dets as much as possible
   topology_file topology/topo0
   thresh_proj 1.e-14
   thresh_ortho 1.e-10
   schedule
   0 2 1000 1.e-4 0.e0
   2 2 2000 1.e-4 0.e0
   4 2 3000 1.e-4 0.e0
   6 2 4000 1.e-4 0.e0
   8 2 5000 1.e-4 0.e0
   10 2 6000 1.e-4 0.e0
   12 2 7000 1.e-4 0.e0
   14 2 8000 1.e-4 0.e0
   16 2 9000 1.e-4 0.e0
   18 2 10000 1.e-4 0.e0
   end
   maxsweep 20
   maxcycle 2
   task_dmrg
   alg_hvec 7
   batchgemm 3
   batchmem 5
   alg_renorm 2
   $end

Keywords
********

Common settings
===============

* ``dtype``: data type used in calculations =0 double, =1 complex
* ``nelec``: :math:`N_{\alpha}+N_{\beta}` - no. of electrons
* ``twom``: :math:`2*(N_{\alpha}-N_{\beta}`) [used in CTNS]
* ``twos``: :math:`2*S` [used in SA-DMRG]
* ``integral_file``: directory of the integral file
* ``scratch``: scratch directory
* ``perfcomm``: performance analysis with file size ``1ULL<<27``

CI (exactdiag.x, fci.x, sci.x)
==============================

* ``dets ... end``: occupied orbitals (abab ordering) in the starting determinant 
  multiple determinants are allowed by separate lines
* ``checkms``: check consistency of dets with twom
* ``flip``: useful for twom=0, which will add flip determinant
* ``nroots``: no. of states to be computed
* ``schedule``: iteration & tolerance for selecting determinants :math:`|H_{AI}*c_I|>\epsilon_1`
* ``maxiter``: max no of SCI iterations

DMRG (ctns.x, sadmrg.x, rdm.x)
==============================

* ``verbose``: 0,1,2 - print level for debugging
* ``qkind``: rNSz - symmetry of the calculation: rNSz - real, N, Sz
* ``topology_file``: topology file for CTNS
* ``maxdets``: max no. of det used in initialization
* ``thresh_proj``: threshold for projection in initialization
* ``thresh_ortho``: threshold for checking right canonical form in initialization
* ``schedule``:

.. code-block::
 
   schedule: sweep schdule
   0 2 1000 1.e-4 0.e0 # correspond to iter, dots, dcut, tol (in Davidson), noise
   end

* ``maxsweep``: max no. of sweeps
* ``maxcycle``: max no. of iteration in Davidson iteration
* ``task_dmrg``: perform dmrg optimization
 
* ``alg_hvec``: algorithm for :math:`H\psi` (>=4 for su2 case)

   * =0: oldest version
   * Algorithm based on symbolic formulae:
      * =1:  dynamic allocation of memory
      * =2:  preallocation of workspace: ``worktot = maxthreads*(opsize + 3*wfsize)``
      * =3:  (factorized) + preallocation of workspace: ``worktot = maxthreads*(opsize + 4*wfsize)``
      * =4:  preallocation of workspace + hintermediates [Hxlst]: ``worktot = maxthreads*(blksize*2 + ndim)`` [local]
      * =5:  preallocation of workspace + hintermediates [Hxlst2]: ``worktot = maxthreads*(blksize*3)`` [local]
      * Batched contractions: preallocation of workspace + Hxlst + Hmmtasks
         * =6: hintermediates [CPU]: ``worktot = batchsize*(blksize*2)``
         * =7: hintermediates [CPU + direct]
         * =8: hintermediates [CPU + single list]
         * =9: hintermediates [CPU + direct + single list]
         * =16: hintermediates [GPU]
         * =17: hintermediates [GPU + direct inter]
         * =18: hintermediates [GPU + single list]
         * =19: hintermediates [GPU + direct + single list]

* ``alg_renorm``: algorithm for renormalization (>=4 for su2 case)
   
   * =0: oldest version
   * Algorithms based on symbolic formulae:
      * =1: dynamic allocation of memory
      * =2: preallocation of workspace
      * =4: preallocation of workspace + rintermediates [Rlst]
      * Batched contractions: preallocation of workspace + Rlist + Rmmtasks
         * =6: rintermediates [CPU]
         * =7: rintermediates [CPU + direct]
         * =8: rintermediates [CPU + single]
         * =9: rintermediates [CPU + direct + single]
         * =16: rintermediates [GPU]
         * =17: rintermediates [GPU + direct]
         * =18: rintermediates [GPU + single]
         * =19: rintermediates [GPU + direct + single]

* ``alg_hcoper``: see ``preprocess_hmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: NSz/NS: only contract op[c1/c2] when it is purely on dot [see preprocess_hmu.h]
   * =2: NSz/NS: no contract for op[c1/c2] - but to perform reduction, x needs to be copied to workspace [see preprocess_hmmtask.h]

* ``alg_rcoper``: see ``preprocess_rmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: the same with alg_hcoper, but in rmmtask there is no need to copy x, because in renorm psi* will contract with sigma

* ``ioasync``: =0 (default: sequential)

Some keyworks useful for debugging
==================================

* ``maxbond``: stop the sweep after this bond and ``maxsweep``
* ``save_formulae``: save formulae for Hx & renorm
* ``save_mmtasks``: save mmtasks for checking

