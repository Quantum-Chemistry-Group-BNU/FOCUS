#!/bin/bash

export PATH=/data/home/scv7260/run/xiangchunyang/ctags-install/bin:$PATH
export PATH=/data/home/scv7260/run/xiangchunyang/valgrind-install/bin:$PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/boost_1_80_0_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/magma_2_6_1_install/lib:$LD_LIBRARY_PATH
export CUDADIR=/data/apps/cuda/11.4 
module load oneAPI/2022.2-mpi
module load oneAPI/2022.2
module load gcc/9.3
module load cuda/11.4
