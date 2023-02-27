#!/bin/bash
#SBATCH -p fat
#SBATCH -J 1DCU_1Node
#SBATCH -t 50:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --partition=fat
#SBATCH -o DCU4_Node1.o
#SBATCH -e DCU4_Node1.e

# -J task_name
# -t runting time
# -N node number
# -n core number
#

export LD_LIBRARY_PATH=/public/home/tangm/xiangchunyang/magma-2.7.0-install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/tangm/xiangchunyang/boost-1.80.0-install/lib:$LD_LIBRARY_PATH
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so
module purge
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.3.0
module load compiler/devtoolset/7.3.1
#module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/3.9.0


srun -N 1 -n 1 -p fat  ../../bin/ctns.x input.dat

