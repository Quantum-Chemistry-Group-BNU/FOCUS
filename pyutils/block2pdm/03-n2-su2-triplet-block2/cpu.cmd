#!/bin/bash
#SBATCH -J fe4nn
#SBATCH -t 48:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

python 00-focus-save-mps-triplet.py > output

#source /public/home/bnulizdtest/add_pynqs.sh
#python lcanon_block2pdm_su2.py > output2

