#!/bin/bash
#SBATCH -J N2_n16
#SBATCH -t 300:00:00 
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH -n 16
#SBATCH -N 2
#SBATCH -o DCU16_Node2.o
#SBATCH -e DCU16_Node2.e

export PATH=/data/home/scv7260/run/xiangchunyang/ctags-install/bin:$PATH
export PATH=/data/home/scv7260/run/xiangchunyang/valgrind-install/bin:$PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/boost_1_80_0_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/magma_2_6_1_install/lib:$LD_LIBRARY_PATH
export CUDADIR=/data/apps/cuda/11.4 
module load oneAPI/2022.2-mpi
module load oneAPI/2022.2
module load gcc/9.3
module load cuda/11.4

export OMP_NUM_THREADS=10
#export MKL_NUM_THREADS=1
#export MKL_DYNAMIC=FALSE

srun  ../../bin/ctns.x input.dat

