#ifndef VMC_UPDATE_H
#define VMC_UPDATE_H

namespace vmc{

   template <typename Tm>
      void update(irbm& wavefun, 
            const fock::onspace& space, 
            const std::vector<Tm>& eloc, 
            const double emean){
         std::cout << "\nvmc::update" << std::endl;
         const double lr = 1.e-3;
         const double eps = 1.e-2;
         /*
            int nsample = space.size();
            for(int i=0; i<wavefun.size(); i++){
            auto Oi = wavefun.dlnpsiC(space,i); // O(x,i)
            auto Oimean = get_mean(nsample,Oi.data());
            double si = linalg::xnrm2(nsample,Oi.data())/nsample;
            double di = si - std::pow(Oimean.norm(),2); // variance 
            double gi = get_mean(eloc,Oi) - emean*Oimean
            wavefun.data[i] -= lr*gi/(dii+eps)
            }
            */
         int nparam = wavefun.size;
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
            //wavefun.data[j] -= lr*gj/(sj + eps);
            wavefun.data[j] -= lr*gj;
         }
         gnorm = std::sqrt(gnorm);
         std::cout << "||g||=" << gnorm << std::endl;
      }

} // vmc

#endif
