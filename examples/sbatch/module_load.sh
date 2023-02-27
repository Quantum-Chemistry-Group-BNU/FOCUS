#!/bin/bash

platform="scy0799" # scy0799  #scv720  #DCU_419 #DCU_zhengzhou

if [ $platform == "scy0799" ]; then
module purge 
export LD_LIBRARY_PATH=/data01/home/scy0799/run/xiangchunyang/project/magma-2.6.1-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data01/home/scy0799/run/xiangchunyang/project/boost_1_80_0_install/lib:$LD_LIBRARY_PATH
export CUDADIR=/data/apps/cuda/11.2 
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
module load intel/oneapi/mpi/2022.1
module load intel/oneapi/compiler/2022.1
module load gcc/9.3
module load cuda/11.2
#########################################
#interactive job run
# $ salloc -N 1 -n 1 --gres=gpu:1

# multile nodes by openmpi and intel mpi 
module load openmpi/4.0.5
export UCX_NET_DEVICES=mlx5_1:1
# multile nodes by nvidia nccl
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5
# export NCCL_IB_HCA=mlx5_1:1
# export NCCL_IB_GID_INDEX=3
########################################

elif [ $platform == "scv720" ]; then
module purge
export PATH=/data/home/scv7260/run/xiangchunyang/ctags-install/bin:$PATH
export PATH=/data/home/scv7260/run/xiangchunyang/valgrind-install/bin:$PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/boost_1_80_0_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/home/scv7260/run/xiangchunyang/magma_2_6_1_install/lib:$LD_LIBRARY_PATH
export CUDADIR=/data/apps/cuda/11.4 
module load oneAPI/2022.2-mpi
module load oneAPI/2022.2
module load gcc/9.3
module load cuda/11.4
#########################################
#interactive job run
# $ salloc -N 1 -n 1 --gres=gpu:1
########################################

elif [ $platform == "DCU_419" ]; then
module purge
export LD_LIBRARY_PATH=/public/software/mathlib/magma/magma-rocm_3.3_develop/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/ictapp_j/xiangchunyang/boost-1.80.0-install/lib:$LD_LIBRARY_PATH
export MKL_DEBUG_CPU_TYPE=5
source /public/software/compiler/intel/oneapi/setvars.sh
module load compiler/devtoolset/9.3.1
module load mathlib/magma/rocm_3.3_develop/3.3
module load compiler/rocm/3.3
#########################################
#interactive job run
# $ salloc -N 1 -n 1 --gres=dcu:1
########################################
elif [ $platform == "DCU_zhengzhou" ]; then
module purge
export LD_LIBRARY_PATH=/public/software/mathlib/magma/magma-rocm_3.3_develop/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/ictapp_j/xiangchunyang/boost-1.80.0-install/lib:$LD_LIBRARY_PATH
export MKL_DEBUG_CPU_TYPE=5
source /public/software/compiler/intel/oneapi/setvars.sh
module load compiler/devtoolset/9.3.1
module load mathlib/magma/rocm_3.3_develop/3.3
module load compiler/rocm/3.3
#########################################
#interactive job run
# $ salloc -N 1 -n 1 --gres=dcu:1 -p fat
########################################

else
    echo "not support yet"
fi

