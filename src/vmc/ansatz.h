#ifndef ANSATZ_H
#define ANSATZ_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include <cmath>
#include "../core/serialization.h"
#include "../core/matrix.h"

namespace vmc{

   struct BaseAnsatz{
      public:
         // value
         virtual std::complex<double> lnpsi(const fock::onstate& state) const = 0;
         // grad with respect to parameters
         virtual std::vector<std::complex<double>> dlnpsiC(const fock::onstate& state) const = 0;
      public:
         int nparam;
         std::vector<double> params;
   };

} // vmc

#endif
