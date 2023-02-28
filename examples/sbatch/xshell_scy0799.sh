#!/bin/bash
#SBATCH -J N2_n16
#SBATCH -t 80:00:00 
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8
#SBATCH --qos=gpugpu
#SBATCH -n 16
#SBATCH -N 2
#SBATCH -o DCU16_Node2.o
#SBATCH -e DCU16_Node2.e
####SBATCH --exclude=g[0006-0011,0013-0014]

export LD_LIBRARY_PATH=/data01/home/scy0799/run/xiangchunyang/project/magma-2.6.1-install:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data01/home/scy0799/run/xiangchunyang/project/boost_1_80_0_install:$LD_LIBRARY_PATH
export CUDADIR=/data/apps/cuda/11.2 
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
module load intel/oneapi/mpi/2022.1
module load intel/oneapi/compiler/2022.1
module load gcc/9.3
module load cuda/11.2
module load openmpi/4.0.5
export UCX_NET_DEVICES=mlx5_1:1

export OMP_NUM_THREADS=10
#export MKL_NUM_THREADS=1
#export MKL_DYNAMIC=FALSE

srun  ../../bin/ctns.x input.dat

