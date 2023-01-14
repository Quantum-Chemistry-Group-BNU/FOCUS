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
         // initialization
         virtual void init(const int _nqubits, const int _nhiden, const double scale) = 0;
         // value
         virtual std::complex<double> psi(const fock::onstate& state) const = 0;
         // grad with respect to parameters
         virtual std::vector<std::complex<double>> dlnpsiC(const fock::onstate& state) = 0;
         // save
         virtual void save(const std::string fname) const = 0;
         // load
         virtual void load(const std::string fname) = 0;
      public:
         int nqubits;
         int nhiden; // rbm
         int nparam;
         std::vector<double> params;
   };

} // vmc

#endif
