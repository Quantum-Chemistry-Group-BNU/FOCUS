#!/bin/bash

#DSUB -n 8n_nccl
#DSUB -A root.zhongkyjssuo
#DSUB -q root.default
#DSUB -N 8
#DSUB -R cpu=128;gpu=4
#DSUB --job_type cosched
#DSUB -oo output_openmpi_openblas_%J.log
#DSUB -eo errlog_openmpi_openblas_%J.log

##kblas hmpi
#export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/kblas-install-ilp64/omp:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/boost_1.80.0_install_hmpi_64/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/magma-2.7.1-install-kblas64-lapack64/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/lapack-3.11.0-install-kblas64:$LD_LIBRARY_PATH

#openblas openmpi
export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/OpenBLAS-0.3.23-install-ilp64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/boost_1.80.0_install_openmpi_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/software/magma-2.7.1-install-openblas64/lib:$LD_LIBRARY_PATH

export CUDADIR=/home/HPCBase/compilers/cuda/11.4.0
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
#module load compilers/gcc/9.3.0
module load compilers/kgcc/9.3.1
module load compilers/cuda/11.4.0
module load libs/nccl/2.16.5-cuda11.4 

##hmpi
#module load mpi/hmpi/1.2.0_kgcc9.3.1

##openmpi
module load mpi/openmpi/4.1.2_gcc9.3.0  

JOB_ID=${BATCH_JOB_ID}
cat ${CCS_ALLOC_FILE}|grep ^whshare|awk '{print $1,"slots="$2}' > ${JOB_ID}.nodefile


##openmpi
#mpirun -N 4 -x OMP_NUM_THREADS=32  -hostfile ${JOB_ID}.nodefile --mca btl ^vader,tcp,openib,uct  --mca pml ucx -x UCX_TLS=self,sm,rc --bind-to numa -x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 -x PATH -x LD_LIBRARY_PATH --mca plm_rsh_agent /opt/batch/agent/tools/dstart  /home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/project/qubic_20230521/qubic/bin/ctns.x input.dat

##nccl
mpirun -N 4 -x OMP_NUM_THREADS=32  -hostfile ${JOB_ID}.nodefile --mca btl ^vader,tcp,openib,uct  --mca pml ucx -x UCX_TLS=self,sm,rc --bind-to numa -x NCCL_IB_HCA=mlx5_0,mlx5_2 -x NCCL_IB_GID_INDEX=3 -x PATH -x LD_LIBRARY_PATH --mca plm_rsh_agent /opt/batch/agent/tools/dstart  /home/share/zhongkyjssuo/home/jiaweile/xiangchunyang/project/qubic_kunpeng/qubic/bin/ctns.x input.dat

