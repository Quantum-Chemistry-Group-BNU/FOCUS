#ifndef VMC_UPDATE_H
#define VMC_UPDATE_H

namespace vmc{

   template <typename Tm>
      void update_exact(BaseAnsatz& wavefun, 
            const fock::onspace& space, 
            const std::vector<Tm>& eloc, 
            const double emean){
         std::cout << "\nvmc::update_exact" << std::endl;
         const double lr = 1.e-3;
         const double eps = 1.e-2;
         int nparam = wavefun.nparam;
         std::vector<std::complex<double>> oi(nparam,0.0);
         std::vector<double> oii(nparam,0.0);
         std::vector<double> eoi(nparam,0.0);
         int nsample = space.size();
         for(int i=0; i<nsample; i++){
            auto dlnpsiC = wavefun.dlnpsiC(space[i]);
            double fac = 1.0/(i+1.0);
            for(int j=0; j<nparam; j++){
               oi[j]  += fac*(dlnpsiC[j] - oi[j]);
               oii[j] += fac*(std::pow(std::abs(dlnpsiC[j]),2) - oii[j]);
               eoi[j] += fac*((eloc[i]*dlnpsiC[j]).real() - eoi[j]);
            }
         }
         // update parameters
         double gnorm = 0.0;
         for(int j=0; j<nparam; j++){
            double sj = oii[j] - std::pow(std::abs(oi[j]),2); // variance
            double gj = eoi[j] - emean*oi[j].real();
            gnorm += std::pow(gj,2);
            //wavefun.params[j] -= lr*gj/(sj + eps);
            wavefun.params[j] -= lr*gj;
         }
         gnorm = std::sqrt(gnorm);
         std::cout << "||g||=" << gnorm << std::endl;
      }

   template <typename Tm>
      void update_sample(BaseAnsatz& wavefun, 
            const fock::onspace& space, 
            const std::vector<Tm>& eloc, 
            const double emean){
         std::cout << "\nvmc::update_sample" << std::endl;
         const double lr = 1.e-3;
         const double eps = 1.e-2;
         int nparam = wavefun.nparam;
         std::vector<std::complex<double>> oi(nparam,0.0);
         std::vector<double> oii(nparam,0.0);
         std::vector<double> eoi(nparam,0.0);
         int nsample = space.size();
         for(int i=0; i<nsample; i++){
            auto dlnpsiC = wavefun.dlnpsiC(space[i]);
            double fac = 1.0/(i+1.0);
            for(int j=0; j<nparam; j++){
               oi[j]  += fac*(dlnpsiC[j] - oi[j]);
               oii[j] += fac*(std::pow(std::abs(dlnpsiC[j]),2) - oii[j]);
               eoi[j] += fac*((eloc[i]*dlnpsiC[j]).real() - eoi[j]);
            }
         }
         // update parameters
         double gnorm = 0.0;
         for(int j=0; j<nparam; j++){
            double sj = oii[j] - std::pow(std::abs(oi[j]),2); // variance
            double gj = eoi[j] - emean*oi[j].real();
            gnorm += std::pow(gj,2);
            //wavefun.params[j] -= lr*gj/(sj + eps);
            wavefun.params[j] -= lr*gj;
         }
         gnorm = std::sqrt(gnorm);
         std::cout << "||g||=" << gnorm << std::endl;
      }

} // vmc

#endif
