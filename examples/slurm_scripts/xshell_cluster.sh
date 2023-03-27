#!/bin/sh
#SBATCH --partition=a100
#SBATCH --job-name=qublic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1

module purge
export PATH=/data/home/scv7260/run/xiangchunyang/ctags-install/bin:$PATH
export PATH=/data/home/scv7260/run/xiangchunyang/valgrind-install/bin:$PATH
export LD_LIBRARY_PATH=/share/home/xiangchunyang/software/gcc-9.5.0-install/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/home/xiangchunyang/software/boost-1.80.0-install-64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/home/xiangchunyang/software/magma-2.7.1-install-64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/share/app/oneapi2022/compiler/2022.0.2/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH
export CUDADIR=/share/app/cuda/cuda-11.6
module load icc/latest
module load mkl/latest
module load mpi/latest
module load compiler-rt/latest
module load cuda/11.6

export OMP_NUM_THREADS=10

#mpirun -np $SLURM_NPROCS -iface ib0 PWmat | tee output
mpirun -np $SLURM_NPROCS  ../../bin/ctns.x input.dat | tee output_N1_mpi4_openmp10
