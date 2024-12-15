
Benchmarks
##########

These executives help to identiy performance problems on supercomputers.

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

benchmark_blas.x
================

Benchmark the performance of GEMV_BATCH and GEMM_BATCH on CPU and GPU.

dell2@20230601 - A100 40G

.. code-block::

   === GEMV_BATCH ===
    i=1 M,N=300,300 nbatch=10 time=0.0444808 flops=4.04669e+07
    i=3 M,N=900,900 nbatch=10 time=0.00387504 flops=4.1806e+09
    i=5 M,N=1500,1500 nbatch=10 time=0.0128143 flops=3.51171e+09
    i=7 M,N=2100,2100 nbatch=10 time=0.026401 flops=3.34078e+09
    i=9 M,N=2700,2700 nbatch=10 time=0.0455472 flops=3.20107e+09
   
   === GEMM_BATCH ===
    i=1 M,N,K=300,300,1 nbatch=10 time=0.000895519 flops=2.01001e+09
    i=3 M,N,K=900,900,1 nbatch=10 time=0.00299933 flops=5.4012e+09
    i=5 M,N,K=1500,1500,1 nbatch=10 time=0.0108046 flops=4.1649e+09
    i=7 M,N,K=2100,2100,1 nbatch=10 time=0.0252583 flops=3.49193e+09
    i=9 M,N,K=2700,2700,1 nbatch=10 time=0.0431005 flops=3.38279e+09
   
   === GEMM_BATCH ===
    i=1 M,N,K=300,300,300 nbatch=10 time=0.00153829 flops=3.51039e+11
    i=3 M,N,K=900,900,900 nbatch=10 time=0.0400837 flops=3.63739e+11
    i=5 M,N,K=1500,1500,1500 nbatch=10 time=0.195221 flops=3.45761e+11
    i=7 M,N,K=2100,2100,2100 nbatch=10 time=0.498141 flops=3.71822e+11
    i=9 M,N,K=2700,2700,2700 nbatch=10 time=0.952389 flops=4.1334e+11
   gpu_init
   rank=0 num_gpus=2 device_id=0 magma_queue=0x1670fc0
   
   === GEMV_BATCH_GPU: magma ===
    i=1 M,N=300,300 nbatch=10 time=0.000630941 flops=2.85288e+09
    i=3 M,N=900,900 nbatch=10 time=0.000710352 flops=2.28056e+10
    i=5 M,N=1500,1500 nbatch=10 time=0.00093322 flops=4.82201e+10
    i=7 M,N=2100,2100 nbatch=10 time=0.00117672 flops=7.49541e+10
    i=9 M,N=2700,2700 nbatch=10 time=0.00145084 flops=1.00494e+11
   
   === GEMV_BATCH_GPU: cublas ===
    i=1 M,N=300,300 nbatch=10 time=0.000406639 flops=4.42653e+09
    i=3 M,N=900,900 nbatch=10 time=0.000201602 flops=8.03563e+10
    i=5 M,N=1500,1500 nbatch=10 time=0.000303903 flops=1.48074e+11
    i=7 M,N=2100,2100 nbatch=10 time=0.000404702 flops=2.17938e+11
    i=9 M,N=2700,2700 nbatch=10 time=0.000552815 flops=2.63741e+11
   
   === GEMM_BATCH: magma ===
    i=1 M,N,K=300,300,1 nbatch=10 time=0.000462502 flops=3.89188e+09
    i=3 M,N,K=900,900,1 nbatch=10 time=0.000397677 flops=4.07366e+10
    i=5 M,N,K=1500,1500,1 nbatch=10 time=0.000516525 flops=8.71207e+10
    i=7 M,N,K=2100,2100,1 nbatch=10 time=0.000704115 flops=1.25264e+11
    i=9 M,N,K=2700,2700,1 nbatch=10 time=0.000869853 flops=1.67615e+11
   
   === GEMM_BATCH: cublas ===
    i=1 M,N,K=300,300,1 nbatch=10 time=0.000271722 flops=6.62442e+09
    i=3 M,N,K=900,900,1 nbatch=10 time=0.0002368 flops=6.84122e+10
    i=5 M,N,K=1500,1500,1 nbatch=10 time=0.000306763 flops=1.46693e+11
    i=7 M,N,K=2100,2100,1 nbatch=10 time=0.000414057 flops=2.13014e+11
    i=9 M,N,K=2700,2700,1 nbatch=10 time=0.00055779 flops=2.61389e+11
   
   === GEMM_BATCH: magma ===
    i=1 M,N,K=300,300,300 nbatch=10 time=0.000504213 flops=1.07098e+12
    i=3 M,N,K=900,900,900 nbatch=10 time=0.00481992 flops=3.02495e+12
    i=5 M,N,K=1500,1500,1500 nbatch=10 time=0.0220727 flops=3.05808e+12
    i=7 M,N,K=2100,2100,2100 nbatch=10 time=0.0381243 flops=4.85831e+12
    i=9 M,N,K=2700,2700,2700 nbatch=10 time=0.0851567 flops=4.62277e+12
   
   === GEMM_BATCH: cublas ===
    i=1 M,N,K=300,300,300 nbatch=10 time=0.00138115 flops=3.90979e+11
    i=3 M,N,K=900,900,900 nbatch=10 time=0.00127414 flops=1.1443e+13
    i=5 M,N,K=1500,1500,1500 nbatch=10 time=0.00461177 flops=1.46365e+13
    i=7 M,N,K=2100,2100,2100 nbatch=10 time=0.0127986 flops=1.44719e+13
    i=9 M,N,K=2700,2700,2700 nbatch=10 time=0.0375398 flops=1.04865e+13

benchmark_mathlib.x
===================

Benchmark the performance of loops and some lapack functions used in DMRG.

mac machine - ZL@20230601

.. code-block::

   ----------------------------------------------------------------------
   tests::test_mathlib maxthreads=4
   ndim=50000000 rows=2000 cols=2000
   ----------------------------------------------------------------------
   time for test_loop = 0.203004 S
   time for test_xnrm2 = 0.0417343 S c=7071.07
   time for test_xcopy = 0.034611 S
   time for test_xgemm = 0.0885083 S FLOPS=168.359 G/s
   time for test_ortho = 2.64287 S
   time for test_eig = 0.390302 S
   time for test_svd = 1.69179 S

jiageng - mkl with OMP_NUM_THREADS=1 MKL_NUM_THREADS=8

.. code-block::

   ----------------------------------------------------------------------
   tests::test_mathlib maxthreads=1
   ndim=50000000 rows=2000 cols=2000
   ----------------------------------------------------------------------
   time for test_loop = 0.0698263 S
   time for test_xnrm2 = 0.0123547 S c=7071.07
   time for test_xcopy = 0.00999965 S
   time for test_xgemm = 0.0311599 S FLOPS=478.216 G/s
   time for test_ortho = 1.96348 S
   time for test_eig = 0.265356 S
   time for test_svd = 0.610317 S

jiageng - mkl with OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

.. code-block::

   ----------------------------------------------------------------------
   tests::test_mathlib maxthreads=8
   ndim=50000000 rows=2000 cols=2000
   ----------------------------------------------------------------------
   data type: DOUBLE
   time for test_loop = 0.0693973 S
   time for test_xnrm2 = 0.00985157 S c=7071.07
   time for test_xcopy = 0.00978466 S
   time for test_xscal = 0.00677777 S
   time for test_xgemm = 0.0288668 S FLOPS=516.205 G/s
   time for test_ortho = 1.611 S
   time for test_eig = 0.221419 S
   time for test_svd = 0.608259 S
   data type: DOUBLE_COMPLEX
   time for test_loop = 0.140277 S
   time for test_xnrm2 = 0.00969873 S c=7071.07
   time for test_xcopy = 0.0194603 S
   time for test_xscal = 0.0138964 S
   time for test_xgemm = 0.10724 S FLOPS=138.952 G/s
   time for test_ortho = 2.00555 S
   time for test_eig = 0.576223 S
   time for test_svd = 1.25792 S

benchmark_lapack.x
==================

Benchmark eig and svd speed for CPU and GPU.

benchmark_io.x
==============

Benchmark the speed of IO.

benchmark_mpi.x
===============

Benchmark the speed of communication using MPI.

Platform: jinan [8 A100 SXM 40GB per node]

1 node:

.. code-block::

        reduce data_size= 8192MB:8GB
         rank=3 data_count: 1073741824  t_reduce=10.452 speed=0.765404GB/S
         rank=1 data_count: 1073741824  t_reduce=10.4236 speed=0.767487GB/S
         rank=5 data_count: 1073741824  t_reduce=10.4273 speed=0.767215GB/S
         rank=7 data_count: 1073741824  t_reduce=10.337 speed=0.77392GB/S
         rank=2 data_count: 1073741824  t_reduce=9.08497 speed=0.880576GB/S
         rank=4 data_count: 1073741824  t_reduce=9.13537 speed=0.875717GB/S
         rank=0 data_count: 1073741824  t_reduce=9.15253 speed=0.874075GB/S
         rank=6 data_count: 1073741824  t_reduce=9.11729 speed=0.877453GB/S

8 node:

.. code-block::

        reduce data_size= 8192MB:8GB
         rank=1 data_count: 1073741824  t_reduce=16.9427 speed=0.472181GB/S
         rank=5 data_count: 1073741824  t_reduce=17.0238 speed=0.46993GB/S
         rank=7 data_count: 1073741824  t_reduce=17.03 speed=0.46976GB/S
         rank=3 data_count: 1073741824  t_reduce=17.0069 speed=0.470397GB/S
         rank=4 data_count: 1073741824  t_reduce=15.7025 speed=0.509474GB/S
         rank=2 data_count: 1073741824  t_reduce=15.7402 speed=0.508252GB/S
         rank=6 data_count: 1073741824  t_reduce=15.6525 speed=0.511101GB/S
         rank=0 data_count: 1073741824  t_reduce=15.6508 speed=0.511155GB/S
         rank=37 data_count: 1073741824  t_reduce=17.1357 speed=0.46686GB/S
         rank=39 data_count: 1073741824  t_reduce=17.1244 speed=0.467169GB/S
         rank=52 data_count: 1073741824  t_reduce=15.6654 speed=0.510679GB/S
         rank=23 data_count: 1073741824  t_reduce=16.8709 speed=0.474189GB/S
         rank=11 data_count: 1073741824  t_reduce=16.6753 speed=0.47975GB/S
         rank=59 data_count: 1073741824  t_reduce=17.0823 speed=0.46832GB/S
         rank=27 data_count: 1073741824  t_reduce=16.6177 speed=0.481415GB/S
         rank=44 data_count: 1073741824  t_reduce=15.8081 speed=0.50607GB/S
         rank=33 data_count: 1073741824  t_reduce=17.097 speed=0.467919GB/S
         rank=50 data_count: 1073741824  t_reduce=15.761 speed=0.507581GB/S
         rank=20 data_count: 1073741824  t_reduce=15.833 speed=0.505273GB/S
         rank=15 data_count: 1073741824  t_reduce=16.6475 speed=0.480553GB/S
         rank=58 data_count: 1073741824  t_reduce=17.0954 speed=0.467963GB/S
         rank=29 data_count: 1073741824  t_reduce=16.6706 speed=0.479887GB/S
         rank=47 data_count: 1073741824  t_reduce=17.1077 speed=0.467626GB/S
         rank=34 data_count: 1073741824  t_reduce=15.8181 speed=0.505749GB/S
         rank=49 data_count: 1073741824  t_reduce=17.1256 speed=0.467138GB/S
         rank=21 data_count: 1073741824  t_reduce=16.9318 speed=0.472485GB/S
         rank=10 data_count: 1073741824  t_reduce=15.9719 speed=0.500881GB/S
         rank=57 data_count: 1073741824  t_reduce=17.1366 speed=0.466838GB/S
         rank=31 data_count: 1073741824  t_reduce=16.6707 speed=0.479882GB/S
         rank=41 data_count: 1073741824  t_reduce=17.1494 speed=0.466489GB/S
         rank=38 data_count: 1073741824  t_reduce=15.8651 speed=0.504251GB/S
         rank=55 data_count: 1073741824  t_reduce=17.1017 speed=0.467791GB/S
         rank=19 data_count: 1073741824  t_reduce=16.9149 speed=0.472956GB/S
         rank=9 data_count: 1073741824  t_reduce=16.7228 speed=0.478388GB/S
         rank=56 data_count: 1073741824  t_reduce=17.1289 speed=0.467048GB/S
         rank=30 data_count: 1073741824  t_reduce=15.8412 speed=0.505012GB/S
         rank=43 data_count: 1073741824  t_reduce=17.1424 speed=0.46668GB/S
         rank=36 data_count: 1073741824  t_reduce=15.8487 speed=0.504773GB/S
         rank=53 data_count: 1073741824  t_reduce=17.1635 speed=0.466104GB/S
         rank=17 data_count: 1073741824  t_reduce=16.8405 speed=0.475046GB/S
         rank=13 data_count: 1073741824  t_reduce=16.7098 speed=0.47876GB/S
         rank=60 data_count: 1073741824  t_reduce=17.1248 speed=0.467158GB/S
         rank=25 data_count: 1073741824  t_reduce=16.6188 speed=0.481384GB/S
         rank=45 data_count: 1073741824  t_reduce=17.0704 speed=0.468647GB/S
         rank=35 data_count: 1073741824  t_reduce=17.0525 speed=0.469141GB/S
         rank=51 data_count: 1073741824  t_reduce=17.1535 speed=0.466377GB/S
         rank=16 data_count: 1073741824  t_reduce=15.813 speed=0.505914GB/S
         rank=8 data_count: 1073741824  t_reduce=15.9396 speed=0.501894GB/S
         rank=61 data_count: 1073741824  t_reduce=17.1468 speed=0.46656GB/S
         rank=28 data_count: 1073741824  t_reduce=15.8296 speed=0.505383GB/S
         rank=42 data_count: 1073741824  t_reduce=15.8523 speed=0.504659GB/S
         rank=32 data_count: 1073741824  t_reduce=15.8338 speed=0.50525GB/S
         rank=54 data_count: 1073741824  t_reduce=15.6917 speed=0.509825GB/S
         rank=18 data_count: 1073741824  t_reduce=15.8357 speed=0.505187GB/S
         rank=12 data_count: 1073741824  t_reduce=15.9831 speed=0.500529GB/S
         rank=62 data_count: 1073741824  t_reduce=17.1375 speed=0.466813GB/S
         rank=24 data_count: 1073741824  t_reduce=15.7948 speed=0.506497GB/S
         rank=46 data_count: 1073741824  t_reduce=15.8391 speed=0.505081GB/S
         rank=48 data_count: 1073741824  t_reduce=15.7107 speed=0.509206GB/S
         rank=22 data_count: 1073741824  t_reduce=15.7921 speed=0.506582GB/S
         rank=14 data_count: 1073741824  t_reduce=15.9392 speed=0.501908GB/S
         rank=63 data_count: 1073741824  t_reduce=17.1174 speed=0.46736GB/S
         rank=26 data_count: 1073741824  t_reduce=15.8097 speed=0.50602GB/S
         rank=40 data_count: 1073741824  t_reduce=15.72 speed=0.508905GB/S

benchmark_nccl.x
================

Benchmark the speed of communication for GPU using NCCL.

.. code-block::

   nvidia-smi topo --matrix

output:

.. code-block::

        	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	mlx5_0	mlx5_1	CPU Affinity	NUMA Affinity
        GPU0	 X 	NV12	NV12	NV12	NV12	NV12	NV12	NV12	PXB	SYS	0-31,64-95	0
        GPU1	NV12	 X 	NV12	NV12	NV12	NV12	NV12	NV12	PXB	SYS	0-31,64-95	0
        GPU2	NV12	NV12	 X 	NV12	NV12	NV12	NV12	NV12	NODE	SYS	0-31,64-95	0
        GPU3	NV12	NV12	NV12	 X 	NV12	NV12	NV12	NV12	NODE	SYS	0-31,64-95	0
        GPU4	NV12	NV12	NV12	NV12	 X 	NV12	NV12	NV12	SYS	PXB	32-63,96-127	1
        GPU5	NV12	NV12	NV12	NV12	NV12	 X 	NV12	NV12	SYS	PXB	32-63,96-127	1
        GPU6	NV12	NV12	NV12	NV12	NV12	NV12	 X 	NV12	SYS	NODE	32-63,96-127	1
        GPU7	NV12	NV12	NV12	NV12	NV12	NV12	NV12	 X 	SYS	NODE	32-63,96-127	1
        mlx5_0	PXB	PXB	NODE	NODE	SYS	SYS	SYS	SYS	 X 	SYS
        mlx5_1	SYS	SYS	SYS	SYS	PXB	PXB	NODE	NODE	SYS	 X
        
        Legend:
        
          X    = Self
          SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
          NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
          PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
          PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
          PIX  = Connection traversing at most a single PCIe bridge
          NV#  = Connection traversing a bonded set of # NVLinks

1 node:

.. code-block::

        reduce data_size= 8192MB:8GB
         rank=4 data_count: 536870912  t_cpu2gpu=0.529247 speed=7.55791GB/S  t_reduce=0.0305075 speed=131.115GB/S  t_gpu2cpu=0.384506 speed=10.403GB/S  t_alloc=0.00459244 speed=870.996GB/S  t_bcast=0.0211663 speed=188.979GB/S  t_dealloc=0.0182338 speed=219.373GB/S
         rank=1 data_count: 536870912  t_cpu2gpu=0.486313 speed=8.22515GB/S  t_reduce=0.0286371 speed=139.679GB/S  t_gpu2cpu=0.357879 speed=11.177GB/S  t_alloc=0.0138547 speed=288.71GB/S  t_bcast=0.0204485 speed=195.613GB/S  t_dealloc=0.0274912 speed=145.501GB/S
         rank=6 data_count: 536870912  t_cpu2gpu=0.573355 speed=6.97648GB/S  t_reduce=0.0312007 speed=128.202GB/S  t_gpu2cpu=0.391629 speed=10.2137GB/S  t_alloc=0.00466877 speed=856.756GB/S  t_bcast=0.0212709 speed=188.05GB/S  t_dealloc=0.0222663 speed=179.643GB/S
         rank=2 data_count: 536870912  t_cpu2gpu=0.502387 speed=7.96199GB/S  t_reduce=0.0292858 speed=136.585GB/S  t_gpu2cpu=0.367484 speed=10.8848GB/S  t_alloc=0.00485972 speed=823.093GB/S  t_bcast=0.0208343 speed=191.991GB/S  t_dealloc=0.0328553 speed=121.746GB/S
         rank=1 data_count: 1073741824  t_cpu2gpu=0.972735 speed=8.22423GB/S  t_reduce=0.0572062 speed=139.845GB/S  t_gpu2cpu=0.702471 speed=11.3884GB/S  t_alloc=0.010644 speed=751.596GB/S  t_bcast=0.0412921 speed=193.742GB/S  t_dealloc=0.00720822 speed=1109.84GB/S
         rank=3 data_count: 1073741824  t_cpu2gpu=0.985433 speed=8.11826GB/S  t_reduce=0.0584496 speed=136.87GB/S  t_gpu2cpu=0.842119 speed=9.49984GB/S  t_alloc=0.0275579 speed=290.298GB/S  t_bcast=0.0419607 speed=190.654GB/S  t_dealloc=0.0711628 speed=112.418GB/S
         rank=5 data_count: 1073741824  t_cpu2gpu=0.986536 speed=8.10918GB/S  t_reduce=0.0596062 speed=134.214GB/S  t_gpu2cpu=0.654303 speed=12.2268GB/S  t_alloc=0.00943906 speed=847.542GB/S  t_bcast=0.0420886 speed=190.075GB/S  t_dealloc=0.0491058 speed=162.913GB/S
         rank=0 data_count: 1073741824  t_cpu2gpu=1.04614 speed=7.64715GB/S  t_reduce=0.0602483 speed=132.784GB/S  t_gpu2cpu=0.768757 speed=10.4064GB/S  t_alloc=0.00992388 speed=806.136GB/S  t_bcast=0.0408677 speed=195.754GB/S  t_dealloc=0.0323032 speed=247.654GB/S
         rank=6 data_count: 1073741824  t_cpu2gpu=1.05711 speed=7.56777GB/S  t_reduce=0.0601292 speed=133.047GB/S  t_gpu2cpu=0.788997 speed=10.1395GB/S  t_alloc=0.00939856 speed=851.194GB/S  t_bcast=0.0421302 speed=189.888GB/S  t_dealloc=0.021885 speed=365.547GB/S
         rank=2 data_count: 1073741824  t_cpu2gpu=1.12424 speed=7.11594GB/S  t_reduce=0.0578542 speed=138.279GB/S  t_gpu2cpu=0.803517 speed=9.95623GB/S  t_alloc=0.00954284 speed=838.325GB/S  t_bcast=0.0416646 speed=192.009GB/S  t_dealloc=0.092915 speed=86.1002GB/S
         rank=4 data_count: 1073741824  t_cpu2gpu=1.07115 speed=7.46858GB/S  t_reduce=0.0590314 speed=135.521GB/S  t_gpu2cpu=0.777144 speed=10.2941GB/S  t_alloc=0.0093461 speed=855.972GB/S  t_bcast=0.0420031 speed=190.462GB/S  t_dealloc=0.0139185 speed=574.775GB/S
         rank=7 data_count: 1073741824  t_cpu2gpu=0.98147 speed=8.15104GB/S  t_reduce=0.0601954 speed=132.901GB/S  t_gpu2cpu=0.682104 speed=11.7284GB/S  t_alloc=0.0265505 speed=301.313GB/S  t_bcast=0.0421495 speed=189.801GB/S  t_dealloc=0.115285 speed=69.3932GB/S
        
8 node:

.. code-block::

        reduce data_size= 8192MB:8GB
         rank=6 data_count: 1073741824  t_cpu2gpu=1.09226 speed=7.32426GB/S  t_reduce=0.480898 speed=16.6355GB/S  t_gpu2cpu=0.769347 speed=10.3984GB/S  t_alloc=0.00924193 speed=865.62GB/S  t_bcast=0.352723 speed=22.6807GB/S  t_dealloc=0.0899489 speed=88.9393GB/S
         rank=3 data_count: 1073741824  t_cpu2gpu=0.966465 speed=8.27759GB/S  t_reduce=0.48744 speed=16.4123GB/S  t_gpu2cpu=0.685037 speed=11.6782GB/S  t_alloc=0.00954509 speed=838.127GB/S  t_bcast=0.357563 speed=22.3737GB/S  t_dealloc=0.00749564 speed=1067.29GB/S
         rank=5 data_count: 1073741824  t_cpu2gpu=0.955632 speed=8.37143GB/S  t_reduce=0.481149 speed=16.6269GB/S  t_gpu2cpu=0.598153 speed=13.3745GB/S  t_alloc=0.009078 speed=881.252GB/S  t_bcast=0.35303 speed=22.661GB/S  t_dealloc=0.0325735 speed=245.599GB/S
         rank=1 data_count: 1073741824  t_cpu2gpu=0.978946 speed=8.17206GB/S  t_reduce=0.487643 speed=16.4054GB/S  t_gpu2cpu=0.742608 speed=10.7728GB/S  t_alloc=0.00940522 speed=850.592GB/S  t_bcast=0.357616 speed=22.3704GB/S  t_dealloc=0.01444 speed=554.017GB/S
         rank=7 data_count: 1073741824  t_cpu2gpu=0.976065 speed=8.19618GB/S  t_reduce=0.480441 speed=16.6514GB/S  t_gpu2cpu=0.602532 speed=13.2773GB/S  t_alloc=0.00922888 speed=866.844GB/S  t_bcast=0.352412 speed=22.7007GB/S  t_dealloc=0.0224113 speed=356.963GB/S
         rank=4 data_count: 1073741824  t_cpu2gpu=1.09865 speed=7.28166GB/S  t_reduce=0.487352 speed=16.4152GB/S  t_gpu2cpu=0.786196 speed=10.1756GB/S  t_alloc=0.00937475 speed=853.356GB/S  t_bcast=0.357533 speed=22.3756GB/S  t_dealloc=0.0678635 speed=117.884GB/S
         rank=2 data_count: 1073741824  t_cpu2gpu=0.951292 speed=8.40961GB/S  t_reduce=0.487531 speed=16.4092GB/S  t_gpu2cpu=0.644469 speed=12.4133GB/S  t_alloc=0.00921451 speed=868.196GB/S  t_bcast=0.357595 speed=22.3717GB/S  t_dealloc=0.0466996 speed=171.308GB/S
         rank=0 data_count: 1073741824  t_cpu2gpu=1.05297 speed=7.59759GB/S  t_reduce=0.487701 speed=16.4035GB/S  t_gpu2cpu=0.653201 speed=12.2474GB/S  t_alloc=0.00903734 speed=885.216GB/S  t_bcast=0.352093 speed=22.7212GB/S  t_dealloc=0.111516 speed=71.7386GB/S
         rank=36 data_count: 1073741824  t_cpu2gpu=0.976475 speed=8.19274GB/S  t_reduce=0.484166 speed=16.5233GB/S  t_gpu2cpu=0.734101 speed=10.8977GB/S  t_alloc=0.0189466 speed=422.239GB/S  t_bcast=0.355351 speed=22.5129GB/S  t_dealloc=0.00711249 speed=1124.78GB/S
         rank=63 data_count: 1073741824  t_cpu2gpu=0.971096 speed=8.23811GB/S  t_reduce=0.487016 speed=16.4266GB/S  t_gpu2cpu=0.606897 speed=13.1818GB/S  t_alloc=0.0753386 speed=106.187GB/S  t_bcast=0.357282 speed=22.3913GB/S  t_dealloc=0.0139196 speed=574.73GB/S
         rank=14 data_count: 1073741824  t_cpu2gpu=1.03459 speed=7.73252GB/S  t_reduce=0.481776 speed=16.6052GB/S  t_gpu2cpu=0.753533 speed=10.6167GB/S  t_alloc=0.0115697 speed=691.46GB/S  t_bcast=0.353565 speed=22.6267GB/S  t_dealloc=0.0901658 speed=88.7255GB/S
         rank=21 data_count: 1073741824  t_cpu2gpu=0.980383 speed=8.16008GB/S  t_reduce=0.482927 speed=16.5656GB/S  t_gpu2cpu=0.667044 speed=11.9932GB/S  t_alloc=0.0184685 speed=433.171GB/S  t_bcast=0.354579 speed=22.562GB/S  t_dealloc=0.0892647 speed=89.6211GB/S
         rank=53 data_count: 1073741824  t_cpu2gpu=0.968774 speed=8.25786GB/S  t_reduce=0.48654 speed=16.4427GB/S  t_gpu2cpu=0.653393 speed=12.2438GB/S  t_alloc=0.0101027 speed=791.865GB/S  t_bcast=0.356894 speed=22.4156GB/S  t_dealloc=0.0625724 speed=127.852GB/S
         rank=42 data_count: 1073741824  t_cpu2gpu=0.980046 speed=8.16288GB/S  t_reduce=0.485165 speed=16.4892GB/S  t_gpu2cpu=0.660812 speed=12.1063GB/S  t_alloc=0.00912898 speed=876.33GB/S  t_bcast=0.355928 speed=22.4765GB/S  t_dealloc=0.00714771 speed=1119.24GB/S
         rank=24 data_count: 1073741824  t_cpu2gpu=0.981065 speed=8.15441GB/S  t_reduce=0.48336 speed=16.5508GB/S  t_gpu2cpu=0.629527 speed=12.7079GB/S  t_alloc=0.00888367 speed=900.529GB/S  t_bcast=0.355018 speed=22.5341GB/S  t_dealloc=0.0728147 speed=109.868GB/S
         rank=34 data_count: 1073741824  t_cpu2gpu=1.07976 speed=7.40905GB/S  t_reduce=0.48439 speed=16.5156GB/S  t_gpu2cpu=0.737216 speed=10.8516GB/S  t_alloc=0.0185539 speed=431.175GB/S  t_bcast=0.355419 speed=22.5086GB/S  t_dealloc=0.0239668 speed=333.795GB/S
         rank=59 data_count: 1073741824  t_cpu2gpu=0.96214 speed=8.3148GB/S  t_reduce=0.48671 speed=16.4369GB/S  t_gpu2cpu=0.70597 speed=11.3319GB/S  t_alloc=0.102139 speed=78.3246GB/S  t_bcast=0.35715 speed=22.3995GB/S  t_dealloc=0.0225039 speed=355.494GB/S
         rank=9 data_count: 1073741824  t_cpu2gpu=0.980773 speed=8.15683GB/S  t_reduce=0.481463 speed=16.616GB/S  t_gpu2cpu=0.727985 speed=10.9892GB/S  t_alloc=0.00974289 speed=821.112GB/S  t_bcast=0.353456 speed=22.6337GB/S  t_dealloc=0.0226994 speed=352.433GB/S
         rank=20 data_count: 1073741824  t_cpu2gpu=0.984128 speed=8.12903GB/S  t_reduce=0.482114 speed=16.5936GB/S  t_gpu2cpu=0.702412 speed=11.3893GB/S  t_alloc=0.00945641 speed=845.987GB/S  t_bcast=0.353934 speed=22.6031GB/S  t_dealloc=0.0322703 speed=247.906GB/S
         rank=49 data_count: 1073741824  t_cpu2gpu=0.959843 speed=8.3347GB/S  t_reduce=0.486109 speed=16.4572GB/S  t_gpu2cpu=0.726747 speed=11.0079GB/S  t_alloc=0.0274205 speed=291.753GB/S  t_bcast=0.356762 speed=22.4239GB/S  t_dealloc=0.0844203 speed=94.7639GB/S
         rank=40 data_count: 1073741824  t_cpu2gpu=0.982936 speed=8.13888GB/S  t_reduce=0.485315 speed=16.4841GB/S  t_gpu2cpu=0.626523 speed=12.7689GB/S  t_alloc=0.00937286 speed=853.528GB/S  t_bcast=0.355994 speed=22.4723GB/S  t_dealloc=0.0420256 speed=190.36GB/S
         rank=30 data_count: 1073741824  t_cpu2gpu=0.987052 speed=8.10494GB/S  t_reduce=0.483698 speed=16.5392GB/S  t_gpu2cpu=0.748358 speed=10.6901GB/S  t_alloc=0.0095499 speed=837.705GB/S  t_bcast=0.355083 speed=22.53GB/S  t_dealloc=0.0144647 speed=553.071GB/S
         rank=32 data_count: 1073741824  t_cpu2gpu=1.01323 speed=7.89557GB/S  t_reduce=0.484575 speed=16.5093GB/S  t_gpu2cpu=0.767266 speed=10.4266GB/S  t_alloc=0.00918364 speed=871.114GB/S  t_bcast=0.355492 speed=22.504GB/S  t_dealloc=0.014723 speed=543.368GB/S
         rank=57 data_count: 1073741824  t_cpu2gpu=0.976865 speed=8.18946GB/S  t_reduce=0.48684 speed=16.4325GB/S  t_gpu2cpu=0.699488 speed=11.4369GB/S  t_alloc=0.0666102 speed=120.102GB/S  t_bcast=0.357217 speed=22.3954GB/S  t_dealloc=0.0931467 speed=85.886GB/S
         rank=11 data_count: 1073741824  t_cpu2gpu=0.991511 speed=8.0685GB/S  t_reduce=0.481285 speed=16.6222GB/S  t_gpu2cpu=0.715608 speed=11.1793GB/S  t_alloc=0.0229302 speed=348.885GB/S  t_bcast=0.353391 speed=22.6378GB/S  t_dealloc=0.0143199 speed=558.661GB/S
         rank=19 data_count: 1073741824  t_cpu2gpu=0.986121 speed=8.11259GB/S  t_reduce=0.48224 speed=16.5893GB/S  t_gpu2cpu=0.756981 speed=10.5683GB/S  t_alloc=0.00960182 speed=833.175GB/S  t_bcast=0.35397 speed=22.6008GB/S  t_dealloc=0.0134437 speed=595.074GB/S
         rank=48 data_count: 1073741824  t_cpu2gpu=1.04659 speed=7.64386GB/S  t_reduce=0.486202 speed=16.4541GB/S  t_gpu2cpu=0.645694 speed=12.3898GB/S  t_alloc=0.0101305 speed=789.693GB/S  t_bcast=0.356795 speed=22.4218GB/S  t_dealloc=0.0326412 speed=245.089GB/S
         rank=44 data_count: 1073741824  t_cpu2gpu=0.984089 speed=8.12934GB/S  t_reduce=0.484989 speed=16.4952GB/S  t_gpu2cpu=0.705141 speed=11.3453GB/S  t_alloc=0.0102312 speed=781.924GB/S  t_bcast=0.355863 speed=22.4805GB/S  t_dealloc=0.0811743 speed=98.5533GB/S
         rank=29 data_count: 1073741824  t_cpu2gpu=0.952397 speed=8.39986GB/S  t_reduce=0.484095 speed=16.5257GB/S  t_gpu2cpu=0.617217 speed=12.9614GB/S  t_alloc=0.00911541 speed=877.635GB/S  t_bcast=0.355118 speed=22.5277GB/S  t_dealloc=0.0234699 speed=340.862GB/S
         rank=38 data_count: 1073741824  t_cpu2gpu=1.17673 speed=6.79852GB/S  t_reduce=0.484721 speed=16.5043GB/S  t_gpu2cpu=0.82344 speed=9.71534GB/S  t_alloc=0.00934477 speed=856.094GB/S  t_bcast=0.355558 speed=22.4998GB/S  t_dealloc=0.0362233 speed=220.853GB/S
         rank=61 data_count: 1073741824  t_cpu2gpu=0.957601 speed=8.35421GB/S  t_reduce=0.487284 speed=16.4175GB/S  t_gpu2cpu=0.62287 speed=12.8438GB/S  t_alloc=0.100093 speed=79.9259GB/S  t_bcast=0.357351 speed=22.387GB/S  t_dealloc=0.0475709 speed=168.17GB/S
         rank=15 data_count: 1073741824  t_cpu2gpu=1.00174 speed=7.98614GB/S  t_reduce=0.481699 speed=16.6079GB/S  t_gpu2cpu=0.631353 speed=12.6712GB/S  t_alloc=0.0273679 speed=292.314GB/S  t_bcast=0.353532 speed=22.6288GB/S  t_dealloc=0.00722073 speed=1107.92GB/S
         rank=16 data_count: 1073741824  t_cpu2gpu=0.999107 speed=8.00715GB/S  t_reduce=0.482563 speed=16.5781GB/S  t_gpu2cpu=0.647195 speed=12.361GB/S  t_alloc=0.0172316 speed=464.263GB/S  t_bcast=0.354069 speed=22.5945GB/S  t_dealloc=0.0661793 speed=120.884GB/S
         rank=50 data_count: 1073741824  t_cpu2gpu=1.02079 speed=7.83706GB/S  t_reduce=0.485985 speed=16.4614GB/S  t_gpu2cpu=0.649757 speed=12.3123GB/S  t_alloc=0.018805 speed=425.419GB/S  t_bcast=0.356729 speed=22.426GB/S  t_dealloc=0.022466 speed=356.094GB/S
         rank=47 data_count: 1073741824  t_cpu2gpu=0.992448 speed=8.06088GB/S  t_reduce=0.485424 speed=16.4804GB/S  t_gpu2cpu=0.657623 speed=12.165GB/S  t_alloc=0.013267 speed=602.998GB/S  t_bcast=0.356025 speed=22.4703GB/S  t_dealloc=0.0131981 speed=606.15GB/S
         rank=27 data_count: 1073741824  t_cpu2gpu=1.00529 speed=7.95789GB/S  t_reduce=0.483084 speed=16.5603GB/S  t_gpu2cpu=0.706133 speed=11.3293GB/S  t_alloc=0.0131062 speed=610.398GB/S  t_bcast=0.35492 speed=22.5403GB/S  t_dealloc=0.0954163 speed=83.8432GB/S
         rank=33 data_count: 1073741824  t_cpu2gpu=0.982864 speed=8.13948GB/S  t_reduce=0.484493 speed=16.5121GB/S  t_gpu2cpu=0.742776 speed=10.7704GB/S  t_alloc=0.0127741 speed=626.265GB/S  t_bcast=0.355458 speed=22.5062GB/S  t_dealloc=0.0999869 speed=80.0105GB/S
         rank=56 data_count: 1073741824  t_cpu2gpu=0.972776 speed=8.22389GB/S  t_reduce=0.486934 speed=16.4293GB/S  t_gpu2cpu=0.634401 speed=12.6103GB/S  t_alloc=0.0606177 speed=131.975GB/S  t_bcast=0.357249 speed=22.3934GB/S  t_dealloc=0.00716804 speed=1116.07GB/S
         rank=8 data_count: 1073741824  t_cpu2gpu=0.966815 speed=8.27459GB/S  t_reduce=0.481609 speed=16.611GB/S  t_gpu2cpu=0.605802 speed=13.2056GB/S  t_alloc=0.0181961 speed=439.655GB/S  t_bcast=0.353493 speed=22.6313GB/S  t_dealloc=0.0466526 speed=171.48GB/S
         rank=23 data_count: 1073741824  t_cpu2gpu=0.995244 speed=8.03823GB/S  t_reduce=0.482665 speed=16.5746GB/S  t_gpu2cpu=0.644689 speed=12.4091GB/S  t_alloc=0.00953033 speed=839.425GB/S  t_bcast=0.3541 speed=22.5925GB/S  t_dealloc=0.00709493 speed=1127.57GB/S
         rank=54 data_count: 1073741824  t_cpu2gpu=0.968874 speed=8.25701GB/S  t_reduce=0.486369 speed=16.4484GB/S  t_gpu2cpu=0.708362 speed=11.2937GB/S  t_alloc=0.0198539 speed=402.943GB/S  t_bcast=0.35686 speed=22.4178GB/S  t_dealloc=0.00744156 speed=1075.04GB/S
         rank=45 data_count: 1073741824  t_cpu2gpu=0.965576 speed=8.28521GB/S  t_reduce=0.485744 speed=16.4696GB/S  t_gpu2cpu=0.645202 speed=12.3992GB/S  t_alloc=0.0207208 speed=386.086GB/S  t_bcast=0.356328 speed=22.4512GB/S  t_dealloc=0.0206004 speed=388.342GB/S
         rank=31 data_count: 1073741824  t_cpu2gpu=0.988014 speed=8.09705GB/S  t_reduce=0.483468 speed=16.5471GB/S  t_gpu2cpu=0.62699 speed=12.7594GB/S  t_alloc=0.0221799 speed=360.687GB/S  t_bcast=0.355051 speed=22.5319GB/S  t_dealloc=0.0517331 speed=154.64GB/S
         rank=37 data_count: 1073741824  t_cpu2gpu=0.968891 speed=8.25687GB/S  t_reduce=0.484917 speed=16.4977GB/S  t_gpu2cpu=0.697175 speed=11.4749GB/S  t_alloc=0.0268954 speed=297.449GB/S  t_bcast=0.355589 speed=22.4979GB/S  t_dealloc=0.0753104 speed=106.227GB/S
         rank=62 data_count: 1073741824  t_cpu2gpu=0.970736 speed=8.24117GB/S  t_reduce=0.487107 speed=16.4235GB/S  t_gpu2cpu=0.720057 speed=11.1102GB/S  t_alloc=0.103846 speed=77.0371GB/S  t_bcast=0.357317 speed=22.3891GB/S  t_dealloc=0.0340474 speed=234.966GB/S
         rank=10 data_count: 1073741824  t_cpu2gpu=0.986281 speed=8.11128GB/S  t_reduce=0.48139 speed=16.6185GB/S  t_gpu2cpu=0.613114 speed=13.0481GB/S  t_alloc=0.0139306 speed=574.275GB/S  t_bcast=0.353425 speed=22.6356GB/S  t_dealloc=0.0679429 speed=117.746GB/S
         rank=22 data_count: 1073741824  t_cpu2gpu=0.985374 speed=8.11874GB/S  t_reduce=0.482742 speed=16.572GB/S  t_gpu2cpu=0.701802 speed=11.3992GB/S  t_alloc=0.00975614 speed=819.997GB/S  t_bcast=0.354267 speed=22.5818GB/S  t_dealloc=0.0459086 speed=174.259GB/S
         rank=55 data_count: 1073741824  t_cpu2gpu=0.968016 speed=8.26433GB/S  t_reduce=0.486273 speed=16.4517GB/S  t_gpu2cpu=0.64333 speed=12.4353GB/S  t_alloc=0.0202506 speed=395.051GB/S  t_bcast=0.356827 speed=22.4198GB/S  t_dealloc=0.0456029 speed=175.427GB/S
         rank=43 data_count: 1073741824  t_cpu2gpu=0.965316 speed=8.28744GB/S  t_reduce=0.485089 speed=16.4918GB/S  t_gpu2cpu=0.687698 speed=11.633GB/S  t_alloc=0.0271353 speed=294.819GB/S  t_bcast=0.355894 speed=22.4786GB/S  t_dealloc=0.0592441 speed=135.035GB/S
         rank=28 data_count: 1073741824  t_cpu2gpu=0.986993 speed=8.10543GB/S  t_reduce=0.482999 speed=16.5632GB/S  t_gpu2cpu=0.748372 speed=10.6899GB/S  t_alloc=0.00916481 speed=872.904GB/S  t_bcast=0.354888 speed=22.5423GB/S  t_dealloc=0.00735677 speed=1087.43GB/S
         rank=35 data_count: 1073741824  t_cpu2gpu=0.975203 speed=8.20342GB/S  t_reduce=0.484317 speed=16.5181GB/S  t_gpu2cpu=0.719607 speed=11.1172GB/S  t_alloc=0.0204174 speed=391.822GB/S  t_bcast=0.355383 speed=22.5109GB/S  t_dealloc=0.0531276 speed=150.581GB/S
         rank=60 data_count: 1073741824  t_cpu2gpu=0.986531 speed=8.10923GB/S  t_reduce=0.486608 speed=16.4403GB/S  t_gpu2cpu=0.705387 speed=11.3413GB/S  t_alloc=0.0380481 speed=210.26GB/S  t_bcast=0.357118 speed=22.4015GB/S  t_dealloc=0.0696526 speed=114.856GB/S
         rank=13 data_count: 1073741824  t_cpu2gpu=0.992956 speed=8.05675GB/S  t_reduce=0.482039 speed=16.5962GB/S  t_gpu2cpu=0.633198 speed=12.6343GB/S  t_alloc=0.0117364 speed=681.641GB/S  t_bcast=0.353598 speed=22.6245GB/S  t_dealloc=0.03303 speed=242.204GB/S
         rank=17 data_count: 1073741824  t_cpu2gpu=0.975703 speed=8.19922GB/S  t_reduce=0.48247 speed=16.5813GB/S  t_gpu2cpu=0.714688 speed=11.1937GB/S  t_alloc=0.0095543 speed=837.319GB/S  t_bcast=0.354035 speed=22.5966GB/S  t_dealloc=0.0215316 speed=371.547GB/S
         rank=52 data_count: 1073741824  t_cpu2gpu=1.03778 speed=7.70879GB/S  t_reduce=0.485825 speed=16.4668GB/S  t_gpu2cpu=0.762863 speed=10.4868GB/S  t_alloc=0.00915444 speed=873.893GB/S  t_bcast=0.356661 speed=22.4303GB/S  t_dealloc=0.0146504 speed=546.061GB/S
         rank=41 data_count: 1073741824  t_cpu2gpu=1.00021 speed=7.99836GB/S  t_reduce=0.485233 speed=16.4869GB/S  t_gpu2cpu=0.735858 speed=10.8717GB/S  t_alloc=0.00944112 speed=847.357GB/S  t_bcast=0.35596 speed=22.4744GB/S  t_dealloc=0.0292465 speed=273.537GB/S
         rank=26 data_count: 1073741824  t_cpu2gpu=1.0269 speed=7.79042GB/S  t_reduce=0.483182 speed=16.5569GB/S  t_gpu2cpu=0.665133 speed=12.0277GB/S  t_alloc=0.00937832 speed=853.031GB/S  t_bcast=0.354953 speed=22.5382GB/S  t_dealloc=0.034973 speed=228.748GB/S
         rank=39 data_count: 1073741824  t_cpu2gpu=0.990321 speed=8.07819GB/S  t_reduce=0.484643 speed=16.507GB/S  t_gpu2cpu=0.614007 speed=13.0292GB/S  t_alloc=0.00897989 speed=890.879GB/S  t_bcast=0.355524 speed=22.502GB/S  t_dealloc=0.122409 speed=65.3549GB/S
         rank=58 data_count: 1073741824  t_cpu2gpu=0.959121 speed=8.34097GB/S  t_reduce=0.486782 speed=16.4345GB/S  t_gpu2cpu=0.624871 speed=12.8026GB/S  t_alloc=0.0824051 speed=97.0814GB/S  t_bcast=0.357184 speed=22.3974GB/S  t_dealloc=0.114671 speed=69.7649GB/S
         rank=12 data_count: 1073741824  t_cpu2gpu=0.964212 speed=8.29693GB/S  t_reduce=0.481217 speed=16.6245GB/S  t_gpu2cpu=0.68797 speed=11.6284GB/S  t_alloc=0.00915293 speed=874.037GB/S  t_bcast=0.353361 speed=22.6398GB/S  t_dealloc=0.111981 speed=71.441GB/S
         rank=18 data_count: 1073741824  t_cpu2gpu=1.01702 speed=7.86613GB/S  t_reduce=0.482352 speed=16.5854GB/S  t_gpu2cpu=0.697138 speed=11.4755GB/S  t_alloc=0.0185733 speed=430.726GB/S  t_bcast=0.354004 speed=22.5986GB/S  t_dealloc=0.112881 speed=70.8711GB/S
         rank=51 data_count: 1073741824  t_cpu2gpu=1.00696 speed=7.94472GB/S  t_reduce=0.485903 speed=16.4642GB/S  t_gpu2cpu=0.708887 speed=11.2853GB/S  t_alloc=0.0104914 speed=762.529GB/S  t_bcast=0.356692 speed=22.4283GB/S  t_dealloc=0.107933 speed=74.1201GB/S
         rank=46 data_count: 1073741824  t_cpu2gpu=1.04337 speed=7.66743GB/S  t_reduce=0.485526 speed=16.477GB/S  t_gpu2cpu=0.738089 speed=10.8388GB/S  t_alloc=0.00952683 speed=839.734GB/S  t_bcast=0.356059 speed=22.4682GB/S  t_dealloc=0.103706 speed=77.1413GB/S
         rank=25 data_count: 1073741824  t_cpu2gpu=0.983922 speed=8.13073GB/S  t_reduce=0.483239 speed=16.555GB/S  t_gpu2cpu=0.717417 speed=11.1511GB/S  t_alloc=0.0273093 speed=292.94GB/S  t_bcast=0.354985 speed=22.5361GB/S  t_dealloc=0.117553 speed=68.0544GB/S
        
