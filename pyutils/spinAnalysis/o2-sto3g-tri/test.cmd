#!/bin/bash
#SBATCH --job-name=o2singlet
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

let "total_tasks_per_node=$SLURM_NTASKS_PER_NODE"

srun --nodes=${SLURM_NNODES} bash -c 'hostname' | sort | uniq | awk -v slots=$SLURM_NTASKS_PER_NODE '{print $0, "slots="slots}' > hostfile

cat hostfile

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
#mpirun --hostfile hostfile --mca btl_openib_allow_ib true --mca btl_openib_warn_default_gid_prefix 0 -mca btl vader,tcp,openib,self -bind-to numa -x NCCL_IB_HCA=NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_NVLS_ENABLE=1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -np ${SLURM_NPROCS} sadmrg.x sadmrg.dat > output

QUDA_ENABLE_P2P=3 /yeesuanAI03/yeesuan/apps/hpcx-v2.14-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.16-x86_64/ompi/bin/mpirun --hostfile hostfile -bind-to numa -map-by slot -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH -mca pml ucx -x NCCL_DEBUG=INFO -x NCCL_TREE_THRESHOLD=0 -x UCX_LOG_LEVEL=info -mca btl vader,tcp,smcuda,self -x NCCL_IB_HCA=NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_NVLS_ENABLE=1 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x NCCL_IB_GID_INDEX=3 -x NCCL_NET_GDR_LEVEL=2 -np ${SLURM_NPROCS} sadmrg.x input.dat > test.out

