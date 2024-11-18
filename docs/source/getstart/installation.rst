
Installation
#############

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Requirements
*************

* c++ (gcc>=9.3)

* nvcc (for gpu)

* blas and lapack

Libraries
**********

1. openmpi (optional, if use mpi)
---------------------------------

Download `openmpi <https://www.open-mpi.org/software/ompi/v5.0/>`_, and build MPI libraries for 64-bit integers:

.. code-block::
   
   ./configure --prefix=XXX/openmpi/install64 --with-slurm --with-pmix FFLAGS="-m64 -fdefault-integer-8" FCFLAGS="-m64 -fdefault-integer-8" CFLAGS=-m64 CXXFLAGS=-m64 --enable-mpi-fortran=usempi

   make -j N && make install

On some platforms, you need to specify more information:

.. code-block::

   ./configure --prefix=/GLOBALFS/bnu_pp_1/openmpi/install64  --with-ucx=/APP/u22/ai_x86/mpi/ucx-1.15.0-gcc-11.4.0-cuda12.2-gdrcopy2.4 --with-cuda=/APP/u22/ai_x86/CUDA/12.2 --with-cuda-libdir=/APP/u22/ai_x86/CUDA/12.2/lib64/stubs --with-pmix=/usr/local/pmix4 --enable-mca-no-build=btl-uct --with-slurm FFLAGS="-m64 -fdefault-integer-8" FCFLAGS="-m64 -fdefault-integer-8" CFLAGS=-m64 CXXFLAGS=-m64 --enable-mpi-fortran=usempi

Add path:

.. code-block::
   
   export PATH="XXX/openmpi/install64/bin:$PATH"
   export LD_LIBRARY_PATH="XXX/openmpi/install64/lib:$LD_LIBRARY_PATH"

2. boost (required)
-------------------

Download `boost <https://www.boost.org/users/download/>`_.

.. code-block::

   ./bootstrap.sh --with-toolset=intel-linux

Edit the file ``project-config.jam`` to add mpi information (get from ``which mpi``)

.. code-block::

   using mpi : /opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin/mpiicpc ; 

Compile

.. code-block::

   ./b2 toolset=intel-linux --layout=tagged link=static,shared threading=multi install --prefix=../install -j 10 —address-model=64

After this, one can find include and lib in the directory specified by prefix. Add the path in ``.bashrc``.

.. code-block::

   export LD_LIBRARY_PATH="XXX/boost/install/lib:$LD_LIBRARY_PATH"


3. FOCUS/extlibs/magma (optional, if use gpu)
---------------------------------------------

Copy a makefile template from ``make.inc-examples`` (e.g., ``make.inc.mkl-gcc-ilp64``) to ``make.inc`` and edit, then make

.. code-block::

   make lib [no need for sparse-lib]
   make test

After the installation, one can find ``libmagma.a`` and ``libmagma.so`` in the ``lib`` directory. 
Add to path.

.. code-block::

   export LD_LIBRARY_PATH="XXX/magma-2.6.1/lib:$LD_LIBRARY_PATH"

4. FOCUS/extlibs/nccl (optional, if use nccl)
---------------------------------------------

Edit ``makefiles/common.mk``, then make

.. code-block::
   
   make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

Add to path

.. code-block::

   export LD_LIBRARY_PATH="XXX/nccl/build/lib:$LD_LIBRARY_PATH"

5. FOCUS/extlibs/gsl (required, for SA-DMRG)
--------------------------------------------

`GNU Scientific Library (GSL) <https://www.gnu.org/software/gsl>`_

.. code-block::

   ./configure prefix=XXX
   make -j
   make install

Add to path

6. FOCUS/extlibs/nlopt (required, for OO-DMRG)
----------------------------------------------

`NLopt <https://nlopt.readthedocs.io/en/latest/>`_ for nonlinear optimization.

.. code-block::
   
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=directory_to_install ..
   make
   make install

For more information, see `installation guide <https://nlopt.readthedocs.io/en/latest/NLopt_Installation/>`_.

7. FOCUS/extlibs/zquatev (required, for KA-DMRG)
------------------------------------------------

The library `zquatev <https://github.com/sunqm/zquatev>`_ is used for diagonalizing quaternion matrix, used in Kramers-adapted DMRG.

Edit ``Makefile`` and make.


Install FOCUS
**************

1. FOCUS/src/ctns/gpu_kernel (optional, if use gpu)
---------------------------------------------------

Edit ``Makefile`` and make.

2. FOCUS
---------

Edit ``Makefile`` and make.

.. warning::

   * If GPU is used, be careful about Cuda toolkit & runtime version.

   * magma needs to be compiled with the same compiler as FOCUS,
     otherwise, there will be problems in using openmp.

The final ``.bashrc`` file should include the following paths for dynamic libraries required for FOCUS: 

.. code-block::

   . /opt/intel/oneapi/setvars.sh
   
   export CUDADIR="/usr/local/cuda-12.2"
   
   MPI_HOME=/usr/local/openmpi
   export PATH=${MPI_HOME}/bin:$PATH:/usr/local/cuda-12.2/bin/
   export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
   export MANPATH=${MPI_HOME}/share/man:$MANPATH
  
   # Required for running FOCUS 
   export LD_LIBRARY_PATH="/storage/boost/install/lib:$LD_LIBRARY_PATH"
   export LD_LIBRARY_PATH="/storage/FOCUS/extlibs/gsl-install/lib:$LD_LIBRARY_PATH"
   export LD_LIBRARY_PATH="/storage/FOCUS/extlibs/nlopt-2.7.1/build/lib:$LD_LIBRARY_PATH"
   export LD_LIBRARY_PATH="/storage/FOCUS/extlibs/magma-2.8.0/lib:$LD_LIBRARY_PATH"
   export LD_LIBRARY_PATH="/root/nccl_apps/nccl220/lib:$LD_LIBRARY_PATH"
   
   export PATH="/storage/FOCUS/bin:${PATH}"
   
