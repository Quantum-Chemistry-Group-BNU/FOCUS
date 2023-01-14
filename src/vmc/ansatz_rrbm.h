#ifndef ANSATZ_RRBM_H
#define ANSATZ_RRBM_H

#include "ansatz.h"

namespace vmc{

   class rrbm : public BaseAnsatz {
      private:
         friend class boost::serialization::access;	   
         template <class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & nqubits
                  & nhiden
                  & nparam 
                  & params;
            }
      public:
         rrbm(){}
         void init(const int _nqubits, const int _nhiden, const double scale=1.e-3){
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
         std::vector<std::complex<double>> dlnpsiC(const fock::onstate& state);
         // save
         void save(const std::string fname) const{
            std::cout << "\nvmc::rrbm::save fname=" << fname << std::endl;
            std::ofstream ofs(fname, std::ios::binary);
            boost::archive::binary_oarchive save(ofs);
            save << (*this);
            ofs.close();
         }
         // load
         void load(const std::string fname){
            std::cout << "\nvmc::rrbm::load fname=" << fname << std::endl;
            std::ifstream ifs(fname, std::ios::binary);
            boost::archive::binary_iarchive load(ifs);
            load >> (*this);
            ifs.close();
         }
   }; // rrbm

   // psi = exp(ai*zi) * sum_{ha} exp(ba*ha + ha*Wai*zi) 
   std::complex<double> rrbm::psi(const fock::onstate& state) const{
      const double alpha = 1.0, beta = 1.0;
      const int INCX = 1, INCY = 1; 
      // get zvec from onstate
      auto zvec = state.get_zvec();
      //--------------
      // phase = ai*zi 
      double phase = std::exp(linalg::xdot(nqubits,&params[0],zvec.data()));
      //--------------
      // Wai*zvec
      std::vector<double> Wz(nhiden);
      linalg::xcopy(nhiden, &params[nqubits], Wz.data());
      linalg::xgemv("N",&nhiden,&nqubits,&alpha,&params[nqubits+nhiden],&nhiden,
            zvec.data(),&INCX,&beta,Wz.data(),&INCY);
      // amp
      double amp = 1.0;
      for(int a=0; a<nhiden; a++){
         amp *= 2.0*std::cosh(Wz[a]);
      }
      return std::complex(amp*phase,0.0);
   }

   // d ln psi_theta*(x) / d theta_i
   std::vector<std::complex<double>> rrbm::dlnpsiC(const fock::onstate& state){
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
      // ai
      double fac = 1.0;
      std::transform(zvec.begin(), zvec.end(), dlnpsiC.begin(),
                     [&fac](const auto& z){ return fac*z; });
      // ba
      std::transform(Wz.begin(), Wz.end(), &dlnpsiC[nqubits],
                     [](const auto& u){ return std::tanh(u); });
      // Wai
      for(int i=0; i<nqubits; i++){
         linalg::xaxpy(nhiden, zvec[i], &dlnpsiC[nqubits], &dlnpsiC[nqubits+nhiden+i*nhiden]);
      }
      /*
      // debug [const needs to be removed]
      std::cout << std::setprecision(10);
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
      exit(1);
      */
      return dlnpsiC;
   }

} // vmc

#endif
