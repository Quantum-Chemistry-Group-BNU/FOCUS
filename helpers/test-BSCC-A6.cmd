#!/bin/bash

#SBATCH --partition=amd_256
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
##SBATCH --mem=250000
#SBATCH -t 100

export I_MPI_PMI_LIBRARY="/usr/lib64/libpmi2.so"
export OMP_NUM_THREADS=64 
srun /public1/home/scb3124/lzd/FOCUS/bin/ctns.x input.dat > test1b.out

