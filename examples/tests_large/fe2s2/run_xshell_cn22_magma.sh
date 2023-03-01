#!/bin/bash
#sbatch --gpus=1 ./xshell_cn22.sh
#sbatch --gpus=1 ./xshell_cn22_magma.sh
#sbatch --gpus=4 ./xshell_cn22_magma.sh
sbatch --qos=gpugpu -N 4 --gres=gpu:8 ./xshell_cn22_magma.sh
