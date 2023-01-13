#ifndef ANSATZ_IRBM_H
#define ANSATZ_IRBM_H

#include "ansatz.h"

namespace vmc{

   class irbm : public BaseAnsatz {
      private:
         // serialize [for MPI] in src/drivers/ctns.cpp
         friend class boost::serialization::access;	   
         template <class Archive>
            void save(Archive & ar, const unsigned int version) const{
               ar & nqubits
                  & nhiden
                  & nparam 
                  & params;
            }
      public:
         irbm(){}
         irbm(const int _nqubits, const int _nhiden, const double scale=1.e-3){
            nqubits = _nqubits;
            nhiden = _nhiden;
            nparam = nqubits + nhiden + nhiden*nqubits;
            auto tmp = linalg::random_matrix<double>(nparam,1)*scale;
            params.resize(nparam);
            linalg::xcopy(nparam, tmp.data(), params.data());
         }
         // value
         std::complex<double> psi(const fock::onstate& state) const;
         // grad
         std::vector<std::complex<double>> dlnpsiC(const fock::onstate& state) const;
   }; // irbm

   // psi = sum_{ha} exp(1j * (ai*zi + ba*ha + ha*Wai*zi))  
   //     = exp(1j * (ai*zi)) * prod_a 2 cos(ba+Wai*zi)
   std::complex<double> irbm::psi(const fock::onstate& state) const{
      const double alpha = 1.0, beta = 1.0;
      const int INCX = 1, INCY = 1; 
      // get zvec from onstate
      auto zvec = state.get_zvec();
      // phase = ai*zi 
      double phase = linalg::xdot(nqubits,&params[0],zvec.data());
      // Wai*zvec
      std::vector<double> Wz(nhiden);
      linalg::xcopy(nhiden, &params[nqubits], Wz.data());
      linalg::xgemv("N",&nhiden,&nqubits,&alpha,&params[nqubits+nhiden],&nhiden,
            zvec.data(),&INCX,&beta,Wz.data(),&INCY);
      // amp
      double amp = 1.0;
      for(int a=0; a<nhiden; a++){
         amp *= 2.0*std::cos(Wz[a]);
      }
      return std::complex(amp*std::cos(phase),amp*std::sin(phase));
   }

   // d ln psi_theta*(x) / d theta_i
   std::vector<std::complex<double>> irbm::dlnpsiC(const fock::onstate& state) const{
      const double alpha = 1.0, beta = 1.0;
      const int INCX = 1, INCY = 1;
      std::vector<std::complex<double>> dlnpsiC(nparam,0.0);
      // get zvec from onstate
      auto zvec = state.get_zvec();
      // Wai*zvec
      std::vector<double> Wz(nhiden);
      linalg::xcopy(nhiden, &params[nqubits], Wz.data());
      linalg::xgemv("N",&nhiden,&nqubits,&alpha,&params[nqubits+nhiden],&nhiden,
            zvec.data(),&INCX,&beta,Wz.data(),&INCY);
      // assemble gradient
      std::complex<double> iunit(0.0,1.0);
      // ai
      std::transform(zvec.begin(), zvec.end(), dlnpsiC.begin(),
                     [&iunit](const auto& z){ return -iunit*z; });
      // ba
      std::transform(Wz.begin(), Wz.end(), &dlnpsiC[nqubits],
                     [](const auto& u){ return -std::tan(u); });
      // Wai
      for(int i=0; i<nqubits; i++){
         linalg::xaxpy(nhiden, zvec[i], &dlnpsiC[nqubits], &dlnpsiC[nqubits+nhiden+i*nhiden]);
      }
      /*
      // debug
      double eps = 1.e-4;
      for(int k=0; k<nparam; k++){
         auto val = this->psi(state);
         params[k] += eps;
         auto valp = this->psi(state);
         params[k] -= 2*eps;
         auto valm = this->psi(state);
         auto diff = tools::conjugate((valp - valm)/(2.0*eps*val));
         std::cout << "CHECK dlnpsiC: k=" << k << " d=" << dlnpsiC[k]
                   << " fd=" << diff << " error="
                   << dlnpsiC[k]-diff << std::endl;
         params[k] += eps;
      }
      */
      return dlnpsiC;
   }

} // vmc

#endif
