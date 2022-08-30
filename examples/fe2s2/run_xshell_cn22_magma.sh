#!/bin/bash
#sbatch --gpus=1 ./xshell_cn22.sh
#sbatch --gpus=1 ./xshell_cn22_magma.sh
sbatch --gpus=8 ./xshell_cn22_magma.sh
