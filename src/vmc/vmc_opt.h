#ifndef VMC_OPT_H
#define VMC_OPT_H

#include "ansatz.h"
#include "vmc_eloc.h"
#include "vmc_sample.h"
#include "vmc_mean.h"
#include "vmc_update.h"

namespace vmc{

   inline void opt_dump(const std::string fname, 
                        const std::vector<double>& ehis){
      std::cout << "\nvmc::opt_dump fname=" << fname << std::endl;
      std::ofstream ofs(fname, std::ios::binary);
      int dim = ehis.size();
      ofs.write((char*)(&dim), sizeof(dim));
      ofs.write((char*)(ehis.data()), sizeof(double)*dim);
      ofs.close();
   }

   template <typename Tm>
      void opt_exact(BaseAnsatz& wavefun,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool debug = (rank==0); 
         if(debug) std::cout << "\nvmc::opt_exact" << std::endl;
         auto t0 = tools::get_time();

         // generate FCI space       
         fock::onspace fci_space;
         if(tools::is_complex<Tm>()){
            fci_space = fock::get_fci_space(int1e.sorb, schd.nelec);
         }else{
            int na = (schd.nelec + schd.twoms)/2;
            int nb = (schd.nelec - schd.twoms)/2;
            fci_space = fock::get_fci_space(int1e.sorb/2, na, nb);
         }

         // set up head-bath table
         const double eps2 = 1.e-10;
         sci::heatbath_table<Tm> hbtab(int2e, int1e);

         int nsample = fci_space.size();
         std::vector<double> ehis(schd.vmc.maxiter);
         for(int iter=0; iter<schd.vmc.maxiter; iter++){
            // compute local energy
            auto eloc = get_eloc(wavefun, fci_space, int2e, int1e, ecore, hbtab, eps2);

            std::vector<double> grad(wavefun.nparam);
            /* 
               std::vector<double> prob(nsample);
               double eps = 1.e-4;
               for(int j=0; j<wavefun.nparam; j++){
               wavefun.params[j] += eps;
               auto eloc1 = get_eloc(wavefun, fci_space, int2e, int1e, ecore, hbtab, eps2);
               for(int i=0; i<nsample; i++){
               auto psi = wavefun.psi(fci_space[i]);
               prob[i] =  std::pow(std::abs(psi),2);
               }
               double fac = 1.0/std::accumulate(prob.begin(), prob.end(), 0.0);
               linalg::xscal(nsample, fac, prob.data());
               double emean1 = nsample*get_mean(nsample, prob.data(), eloc1.data()).real(); 

               wavefun.params[j] -= 2*eps;
               auto eloc2 = get_eloc(wavefun, fci_space, int2e, int1e, ecore, hbtab, eps2);
               for(int i=0; i<nsample; i++){
               auto psi = wavefun.psi(fci_space[i]);
               prob[i] =  std::pow(std::abs(psi),2);
               }
               fac = 1.0/std::accumulate(prob.begin(), prob.end(), 0.0);
               linalg::xscal(nsample, fac, prob.data());
               double emean2 = nsample*get_mean(nsample, prob.data(), eloc2.data()).real(); 
               grad[j] = (emean1-emean2)/(2.0*eps); 
               std::cout << "j=" << j
               << " fdiff=" << grad[j]
               << std::endl; 
               wavefun.params[j] += eps;
               } // j
               */

            // update parameters
            double emean = update_exact(wavefun, fci_space, eloc, schd.vmc.lr, grad);
            ehis[iter] = emean;

            std::cout << "iter=" << iter << " emean=" << emean << std::endl;
         } // iter
         opt_dump(schd.vmc.history, ehis);
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("vmc::opt_exact", t0, t1);
         }
      }

   template <typename Tm>
      void opt_sample(BaseAnsatz& wavefun,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const fock::onspace& sci_space){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool debug = (rank==0); 
         if(debug) std::cout << "\nvmc::opt_sample" << std::endl;
         auto t0 = tools::get_time();

         // set up head-bath table
         const double eps2 = 1.e-10;
         sci::heatbath_table<Tm> hbtab(int2e, int1e);

         int nsample = schd.vmc.nsample;
         std::vector<double> ehis(schd.vmc.maxiter);
         for(int iter=0; iter<schd.vmc.maxiter; iter++){
            // sample determinants
            auto samples = get_sample(wavefun, nsample, sci_space[0]);
            // compute local energy
            auto eloc = get_eloc(wavefun, samples, int2e, int1e, ecore, hbtab, eps2);
            // update parameters
            double emean = update_sample(wavefun, samples, eloc, schd.vmc.lr);
            ehis[iter] = emean;
            std::cout << "iter=" << iter << " emean=" << emean << std::endl;
         }
         opt_dump(schd.vmc.history, ehis);
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("vmc::opt_sample", t0, t1);
         }
      }

} // vmc

#endif
