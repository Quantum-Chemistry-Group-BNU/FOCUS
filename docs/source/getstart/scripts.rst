
Scripts
#######

Some scripts for submitting jobs using slurm.

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Example
=======

On BSCC-A6

.. code-block::

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
  
Using local disk

.. code-block::

   #!/bin/bash
   
   #SBATCH --partition=amd_sata
   #SBATCH -N 4
   #SBATCH --ntasks-per-node=1
   #SBATCH -c 64
   ##SBATCH --mem=250000
   #SBATCH -t 100
   
   export I_MPI_PMI_LIBRARY="/usr/lib64/libpmi2.so"
   export OMP_NUM_THREADS=64
   
   srun hostname |sort
   export SCRATCH="/disk1/fe4s4"
   srun mkdir -p $SCRATCH
   srun rm -r $SCRATCH
   srun mkdir -p $SCRATCH
   cp -r scratch $SCRATCH
   
   srun /public1/home/scb3124/lzd/FOCUS/bin/ctns.x local.dat > local4.out
   
   rm -r  $SCRATCH/scratch/*op
   mv $SCRATCH/scratch scratch_new
   srun rm -r $SCRATCH
   
