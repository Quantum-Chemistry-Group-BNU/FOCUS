
Known Problems
##############

* ``mpi::reduce`` hangs on intel MPI (https://github.com/boostorg/mpi/issues/1204): edit ``BOOST_MPI_USE_IMPROBE`` in ``boost/mpi/config.hpp``

* ``std::filesystem`` may cause problems for old compilers


* ``thresh_proj`` & ``thresh_ortho`` sensitivity in SCI projection [theoretical/numerical problem?]: implement canonicalization with help!

* The truncation ``thresh_sig2=1.e-14`` may lead to some differences on the number of renormalized states, such that the optimized energies can be slightly different with ``maxcycle`` truncated.


