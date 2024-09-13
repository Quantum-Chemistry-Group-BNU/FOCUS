.. FOCUS documentation master file, created by
   sphinx-quickstart on Mon Aug 26 15:04:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FOCUS
#####

**FOCUS** (FermiOniC qUantum Simulation) is developed to explore the complexity of electronic structure problems.
It support FCI/SCI and CTNS/DMRG algorithms for both nonrelativistic and relativistic Hamiltonians.
Currently, it support both MPI/OpenMP and MPI/GPU parallelizations for non-spin-adapted
and spin-adapted DMRG with nonrelativistic Hamiltonian.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getstart/installation
   getstart/basics
   getstart/benchmark
   getstart/scripts
   getstart/problems

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/ci
   tutorials/ctns
   tutorials/sadmrg
   tutorials/rdm

.. toctree::
   :maxdepth: 1
   :caption: Theory and Implementation

   methods/ci
   methods/csf
   methods/ordering
   methods/init
   methods/dmrg
   methods/se
   methods/kr
   methods/rdm
   methods/oodmrg

