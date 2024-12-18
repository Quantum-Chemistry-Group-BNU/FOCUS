
Known problems
##############

* ``mpi::reduce`` hangs on intel MPI (https://github.com/boostorg/mpi/issues/1204): edit ``BOOST_MPI_USE_IMPROBE`` in ``boost/mpi/config.hpp``

* ``std::filesystem`` may cause problems for old compilers

* The truncation ``thresh_sig2=1.e-14`` may lead to some differences on the number of renormalized states, such that the optimized energies can be slightly different with ``maxcycle`` truncated.

* 2024/12/12: When SCI wavefunction is bad, initialization must be careful. Different runs with large dcut at the initial step
  can lead to different results, as the calculations are affected by renormalized states with very small weights. This can also be affected by the settings for ``thresh_sig2``. Initialization with small dcut can help.



