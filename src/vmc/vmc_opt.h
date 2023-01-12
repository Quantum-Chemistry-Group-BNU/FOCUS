#ifndef VMC_OPT_H
#define VMC_OPT_H

#include "ansatz_rbm.h"
#include "vmc_eloc.h"
#include "vmc_sample.h"
#include "vmc_mean.h"
#include "vmc_update.h"

namespace vmc{

   template <typename Tm>
      void opt(irbm& wavefun,
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
         if(debug){ 
            std::cout << "\nvmc::opt" << std::endl;
         }
         auto t0 = tools::get_time();
         
         // set up head-bath table
         const double eps2 = 1.e-10;
         sci::heatbath_table<Tm> hbtab(int2e, int1e);
/*
         wavefun.sample_init(schd.vmc.nsample, sci_space[0]);
*/
         int nsample = schd.vmc.nsample;

         for(int iter=0; iter<schd.vmc.maxiter; iter++){

            // sample determinants
            auto samples = get_sample(wavefun, nsample, sci_space[0]);
            
            // compute local energy
            auto eloc = get_eloc(wavefun, samples, int2e, int1e, hbtab, eps2);
            // emean
            auto emean = get_mean(nsample, eloc.data()); 
            std::cout << "iter=" << iter << " emean=" << emean << std::endl;

            // update parameters
            update(wavefun, samples, eloc, emean.real());

         }
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("vmc::opt", t0, t1);
         }
      }

} // vmc

#endif
