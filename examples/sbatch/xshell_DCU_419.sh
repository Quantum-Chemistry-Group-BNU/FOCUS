#!/bin/bash
#SBATCH -J 8DCU_2Node
#SBATCH -t 0-12:00:00 
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH -o DCU8_Node2.o
#SBATCH -e DCU8_Node2.e


module purge
export LD_LIBRARY_PATH=/public/software/mathlib/magma/magma-rocm_3.3_develop/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/ictapp_j/xiangchunyang/boost-1.80.0-install/lib:$LD_LIBRARY_PATH
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so
export MKL_DEBUG_CPU_TYPE=5
source /public/software/compiler/intel/oneapi/setvars.sh
module load compiler/devtoolset/9.3.1
module load mathlib/magma/rocm_3.3_develop/3.3
module load compiler/rocm/3.3


srun   ../../bin/ctns.x input.dat

