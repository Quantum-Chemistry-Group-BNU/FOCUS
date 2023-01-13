#ifndef VMC_UPDATE_H
#define VMC_UPDATE_H

namespace vmc{

   template <typename Tm>
      double update_exact(BaseAnsatz& wavefun, 
            const fock::onspace& space, 
            //const std::vector<Tm>& eloc){ 
            const std::vector<Tm>& eloc,
            const double lr,
            const std::vector<double>& grad){ 
         std::cout << "\nvmc::update_exact" << std::endl;
         const double eps = 1.e-2;
         int nsample = space.size();
         // probablity
         std::vector<double> prob(nsample);
         for(int i=0; i<nsample; i++){
            auto psi = wavefun.psi(space[i]);
            prob[i] =  std::pow(std::abs(psi),2);
         }
         double fac = 1.0/std::accumulate(prob.begin(), prob.end(), 0.0);
         linalg::xscal(nsample, fac, prob.data());
         // emean
         double emean = nsample*get_mean(nsample, prob.data(), eloc.data()).real(); 
         // grad
         int nparam = wavefun.nparam;
         std::vector<std::complex<double>> oi(nparam,0.0);
         std::vector<double> oii(nparam,0.0);
         std::vector<double> eoi(nparam,0.0);
         for(int i=0; i<nsample; i++){
            auto dlnpsiC = wavefun.dlnpsiC(space[i]);
            for(int j=0; j<nparam; j++){
               oi[j]  += dlnpsiC[j]*prob[i];
               oii[j] += std::pow(std::abs(dlnpsiC[j]),2)*prob[i];
               eoi[j] += (eloc[i]*dlnpsiC[j]).real()*prob[i];
            }
         }
         // update parameters
         double gnorm = 0.0;
         for(int j=0; j<nparam; j++){
            double sj = oii[j] - std::pow(std::abs(oi[j]),2); // variance
            double gj = 2.0*(eoi[j] - emean*oi[j].real());
            gnorm += std::pow(gj,2);
            //wavefun.params[j] -= lr*gj/(sj + eps);
            wavefun.params[j] -= lr*gj;
            /*
            std::cout << std::setprecision(10);
            std::cout << "j=" << j 
                      << " oi=" << oi[j]
                      << " oii=" << oii[j]
                      << " eoi=" << eoi[j]
                      << " sj=" << sj
                      << " gj=" << gj
                      << " grad=" << grad[j]
                      << " diff=" << gj-grad[j]
                      << std::endl;
            */
         }
         gnorm = std::sqrt(gnorm);
         std::cout << std::setprecision(12);
         std::cout << "||g||=" << gnorm << " emean=" << emean << std::endl;
         return emean;
      }

   template <typename Tm>
      double update_sample(BaseAnsatz& wavefun, 
            const fock::onspace& space,
            const std::vector<Tm>& eloc,
            const double lr){
         std::cout << "\nvmc::update_sample" << std::endl;
         const double eps = 1.e-2;
         int nparam = wavefun.nparam;
         std::vector<std::complex<double>> oi(nparam,0.0);
         std::vector<double> oii(nparam,0.0);
         std::vector<double> eoi(nparam,0.0);
         int nsample = space.size();
         double emean = get_mean(nsample, eloc.data()).real();
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
            double gj = 2.0*(eoi[j] - emean*oi[j].real());
            gnorm += std::pow(gj,2);
            //wavefun.params[j] -= lr*gj/(sj + eps);
            wavefun.params[j] -= lr*gj;
         }
         gnorm = std::sqrt(gnorm);
         std::cout << "||g||=" << gnorm << " emean=" << emean << std::endl;
         return emean;
      }

} // vmc

#endif
