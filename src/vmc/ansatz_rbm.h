#ifndef ANSATZ_RBM_H
#define ANSATZ_RBM_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include <cmath>
#include "../core/serialization.h"
#include "../core/matrix.h"

namespace vmc{

   class irbm{
      private:
         // serialize [for MPI] in src/drivers/ctns.cpp
         friend class boost::serialization::access;	   
         template <class Archive>
            void save(Archive & ar, const unsigned int version) const{
               ar & nqubits
                  & nhiden
                  & size 
                  & data;
            }
      public:
         irbm(){}
         irbm(const int _nqubits, const int _nhiden, const double scale=1.e-3){
            nqubits = _nqubits;
            nhiden = _nhiden;
            size = nqubits + nhiden + nhiden*nqubits;
            auto tmp = linalg::random_matrix<double>(size,1)*scale;
            data.resize(size);
            linalg::xcopy(size, tmp.data(), data.data());
         }
         // value
         std::complex<double> lnpsi(const fock::onstate& state) const;
         // grad
         std::vector<std::complex<double>> dlnpsiC(const fock::onstate& state) const;
      public:
         int nqubits;
         int nhiden;
         int size;
         std::vector<double> data;
   }; // irbm

   // psi = sum_{ha} exp(1j * (ai*zi + ba*ha + ha*Wai*zi))  
   //     = exp(1j * (ai*zi)) * prod_a 2 cos(ba+Wai*zi)
   // lnpsi = 1j*(ai*zi) + sum_a ln[2*cos(ba+Wai*zi)
   std::complex<double> irbm::lnpsi(const fock::onstate& state) const{
      const double alpha = 1.0, beta = 1.0;
      const int INCX = 1, INCY = 1; 
      double lnpsiR = 0.0, lnpsiI = 0.0;
      // get zvec from onstate
      auto zvec = state.get_zvec();
/*
      std::cout << "state=" << state << std::endl;
      tools::print_vector(zvec,"z");
*/
      // lnpsiI
      lnpsiI = linalg::xdot(nqubits,&data[0],zvec.data());
      // Wai*zvec
      std::vector<double> Wz(nhiden);
      linalg::xcopy(nhiden, &data[nqubits], Wz.data());
      linalg::xgemv("N",&nhiden,&nqubits,&alpha,&data[nqubits+nhiden],&nhiden,
            zvec.data(),&INCX,&beta,Wz.data(),&INCY);
/*      
      tools::print_vector(Wz,"lzdWz");
*/
      // lnpsiR
      for(int a=0; a<nhiden; a++){
         lnpsiR += std::log(std::cos(Wz[a]));
      }
      return std::complex(lnpsiR,lnpsiI);
   }

   // d ln psi_theta*(x) / d theta_i
   std::vector<std::complex<double>> irbm::dlnpsiC(const fock::onstate& state) const{
      const double alpha = 1.0, beta = 1.0;
      const int INCX = 1, INCY = 1; 
      // get zvec from onstate
      auto zvec = state.get_zvec(); 
      // Wai*zvec
      std::vector<double> Wz(nhiden,0.0);
      linalg::xcopy(nhiden, &data[nqubits], Wz.data());
      linalg::xgemv("N",&nhiden,&nqubits,&alpha,&data[nqubits+nhiden],&nhiden,
            zvec.data(),&INCX,&beta,Wz.data(),&INCY);
      // assemble gradient
      std::complex<double> iunit(0.0,1.0);
      std::vector<std::complex<double>> dlnpsiC(size);
      // ai
      std::transform(zvec.begin(), zvec.end(), dlnpsiC.begin(),
                     [&iunit](const auto& z){ return -iunit*z; });
      // ba
      std::transform(Wz.begin(), Wz.end(), &dlnpsiC[nqubits],
                     [](const auto& u){ return -std::tan(u); });
      // Wai
      for(int i=0; i<nqubits; i++){
         double zi = zvec[i];
         std::transform(&dlnpsiC[nqubits], &dlnpsiC[nqubits+nhiden],
                        &dlnpsiC[nqubits+nhiden+i*nqubits],
                        [&zi](const auto& x){ return x*zi; });
      }
      return dlnpsiC;
   }

} // vmc

#endif
