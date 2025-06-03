
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

* use ``prop.x`` to compute properties (savebin, overlap, RDMs)

A typical input file
********************

.. code-block::

   dtype 0
   sorb 40
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
   alg_hvec 4
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

DMRG (ctns.x, sadmrg.x, prop.x)
===============================

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
      * =4:  preallocation of workspace + hintermediates [Hxlst]: ``worktot = maxthreads*(blksize*2)`` [local]
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

* ``alg_hinter``: hintermediates (sum of operators), see ``preprocess_hinter.h``. Note that for direct algorithm (7,9,17,19), alg_hinter must be specified.

   * =0: cpu axpy + openmp [work for alg_hvec=4,5,6,8] 
   * =1: cpu batchgemv [work for alg_hvec=4,5,6,7,8,9]
   * =2: gpu batchgemv [work for alg_hvec=16,17,18,19]

* ``alg_rinter``: rintermediates (sum of operators), see ``preprocess_rinter.h``

* ``alg_hcoper``: see ``preprocess_hmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: NSz/NS: only contract op[c1/c2] when it is purely on dot [see preprocess_hmu.h]
   * =2: NSz/NS: no contract for op[c1/c2] - but to perform reduction, x needs to be copied to workspace [see preprocess_hmmtask.h]

* ``alg_rcoper``: see ``preprocess_rmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: the same with alg_hcoper, but in rmmtask there is no need to copy x, because in renorm psi* will contract with sigma

* ``batchhvec``: specifying options for hinter, gemm, reduction in batch GPU algorithm

   * =2: magma (gemv/gemm)
   * =3: cublas batch (gemv/gemm) 
   * =4: cublas batch (gemv/gemm) + stream

* ``batchrenorm``: specifying options for rinter, gemm, reduction in batch GPU algoirthm

* ``ioasync``: =false (default: sequential)
   
* ``ifdist1``: distributed computation of C*S terms for Hx

* ``ifdistc``: whether to treat dot in dmrg specially. This is possible because dot operators are stored in all processes.

* ``ifdists``: whether to store opS distributedly. This must be used with ifdist1,
  otherwise, some C*S terms are missing, leading to wrong results!

* ``ifab2pq``: switch from A,B to P,Q in renormalization: 

* ``alg_a2p,alg_b2q``: control the algorithms for A,B to P,Q:

  * =0: CPU - xaxpy
  * =1: CPU - gemm
  * =2: CPU - gemm [saving memory]
  * =3: GPU - gemm [saving memory]

Some keyworks useful for debugging
==================================

* ``maxbond``: stop the sweep after this bond and ``maxsweep``
* ``save_formulae``: save formulae for Hx & renorm
* ``save_mmtasks``: save mmtasks for checking

Further examples
****************

Example for dmrg cpu
====================

.. code-block::

   task_dmrg
   alg_hvec 4
   alg_renorm 4

Example for dmrg gpu
====================

.. code-block::

   task_dmrg
   alg_hvec 19
   alg_hinter 1
   alg_coper 2
   alg_renorm 19
   alg_rinter 1
   alg_roper 1
   batchhvec 4 4 4
   batchrenorm 4 4 4

