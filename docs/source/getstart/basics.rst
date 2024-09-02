
Basics
######

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

Common settings
***************

* ``dtype``: data type used in calculations =0 double, =1 complex
* ``nelec``: :math:`N_{\alpha}+N_{\beta}` - no. of electrons
* ``twom``: :math:`2*(N_{\alpha}-N_{\beta}`) [used in CTNS]
* ``twos``: :math:`2*S` [used in SA-DMRG]
* ``integral_file``: directory of the integral file
* ``scratch``: scratch directory
* ``perfcomm``: performance analysis with file size ``1ULL<<27``

SCI
***

* ``dets ... end``: occupied orbitals (abab ordering) in the starting determinant 
  multiple determinants are allowed by separate lines
* ``checkms``: check consistency of dets with twom
* ``flip``: useful for twom=0, which will add flip determinant
* ``nroots``: no. of states to be computed
* ``schedule``: iteration & tolerance for selecting determinants :math:`|H_{AI}*c_I|>\epsilon_1`
* ``maxiter``: max no of SCI iterations

CTNS
****
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
 
* ``alg_hvec``: algorithm for :math:`H\psi`

   * =0: oldest version
   * =1: symbolic formulae + dynamic allocation of memory
   * =2: symbolic formulae + preallocation of workspace: ``worktot = maxthreads*(opsize + 3*wfsize)``
   * =3: symbolic formulae (factorized) + preallocation of workspace: ``worktot = maxthreads*(opsize + 4*wfsize)``
   * =4: symbolic formulae + preallocation of workspace + intermediates [Hxlst]: ``worktot = maxthreads*(blksize*2 + ndim)`` [local]
   * =5: symbolic formulae + preallocation of workspace + intermediates [Hxlst2]: ``worktot = maxthreads*(blksize*3)`` [local]
   * =6: symbolic formulae + preallocation of workspace + intermediates [BatchGEMM]: ``worktot = batchsize*(blksize*2)``
   * =7: symbolic formulae + preallocation of workspace + intermediates [direct]
   * =8: symbolic formulae + preallocation of workspace + intermediates [single]
   * =9: symbolic formulae + preallocation of workspace + intermediates [direct + single]
   * =16: symbolic formulae + preallocation of workspace + intermediates [GPU]
   * =17: symbolic formulae + preallocation of workspace + intermediates [GPU + direct inter]
   * =18: symbolic formulae + preallocation of workspace + intermediates [GPU + single]
   * =19: symbolic formulae + preallocation of workspace + intermediates [GPU + direct + single]

* ``alg_renorm``: algorithm for renormalization
   
   * =0: oldest version
   * =1: symbolic formulae + dynamic allocation of memory
   * =2: symbolic formulae + preallocation of workspace
   * =4: symbolic formulae + preallocation of workspace + intermediates [Rlst]
   * =6: symbolic formulae + preallocation of workspace + intermediates [BatchGEMM]
   * =7: symbolic formulae + preallocation of workspace + intermediates [direct]
   * =8: symbolic formulae + preallocation of workspace + intermediates [single]
   * =9: symbolic formulae + preallocation of workspace + intermediates [direct + single]
   * =16: symbolic formulae + preallocation of workspace + intermediates [GPU]
   * =17: symbolic formulae + preallocation of workspace + intermediates [GPU + direct]
   * =18: symbolic formulae + preallocation of workspace + intermediates [GPU + single]
   * =19: symbolic formulae + preallocation of workspace + intermediates [GPU + direct + single]

* ``alg_hcoper``: see ``preprocess_hmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: NSz/NS: only contract op[c1/c2] when it is purely on dot [see preprocess_hmu.h]
   * =2: NSz/NS: no contract for op[c1/c2] - but to perform reduction, x needs to be copied to workspace [see preprocess_hmmtask.h]

* ``alg_rcoper``: see ``preprocess_rmu.h``

   * =0: always contract op[c1/c2] - most general case
   * =1: the same with alg_hcoper, but in rmmtask there is no need to copy x, because in renorm psi* will contract with sigma

* ``ioasync``: =0 (default: sequential)

Some keyworks useful for debugging
**********************************

* ``maxbond``: stop the sweep after this bond and ``maxsweep``
* ``save_formulae``: save formulae for Hx & renorm

