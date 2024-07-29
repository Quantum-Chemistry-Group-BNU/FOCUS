#ifndef CTNS_OODMRG_H
#define CTNS_OODMRG_H

namespace ctns{

   template <typename Qm, typename Tm>
      void oodmrg(comb<Qm,Tm>& icomb, 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool debug = (rank==0);
         const int maxiter = schd.ctns.ooparams.maxiter; 
         if(debug){ 
            std::cout << "\nctns::oodmrg maxiter=" << maxiter 
               << " maxsweep=" << schd.ctns.maxsweep 
               << std::endl;
         }
         auto t0 = tools::get_time();

         const int norb = icomb.get_nphysical();
         std::vector<double> enew_history(maxiter);
         std::vector<double> emin_history(maxiter);
         double e_min = 1.e20;
         auto urot_min = linalg::identity_matrix<Tm>(norb);
         for(int iter=0; iter<maxiter; iter++){
           
            // minimize entanglement
            if(rank == 0){
            // we assume that icomb has already been available, 
            // which is usually the case with a initial MPS from
            // SCI or a previous optimization.
            
            /*
            // randomized swap layer
            urot,icomb_new = minimize_entropy(icomb)
            */
            }
            /*
#ifndef SERIAL
   if(size > 1){
      mpi_wrapper::broadcast(schd.world, icomb, 0);
      icomb.world = schd.world;
   }
#endif
            */

            auto urot = urot_min;

            // update integrals
            auto int1e_new = int1e.rotate_spatial(urot);
            //auto int2e_new = int1e.rotate_spatial(urot);
            exit(1);
            /*
            comb<Qm,Tm> icomb_new;
            */
            auto& icomb_new = icomb;
            //auto& int1e_new = int1e;
            auto& int2e_new = int2e;

            // prepare environment
            auto Hij = ctns::get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch); 

            // optimization
            auto result = ctns::sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
            double e_new = result.get_eminlast(0);
            if(rank == 0){
               std::cout << "iter=" << iter 
                  << " e_new=" << std::setprecision(12) << e_new 
                  << std::endl;
            }

            // check acceptance
            if(e_new < e_min){
               if(rank == 0) std::cout << "accept the move!" << std::endl;
               e_min = e_new;
            //   urot_min = urot;
            //   icomb = icomb_new;
            }else{
               if(rank == 0) std::cout << "reject the move!" << std::endl;
            }
            enew_history[iter] = e_new;
            emin_history[iter] = e_min;
            exit(1);
         } // iter

         // save e_min,urot_min,icomb?

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg", t0, t1);
         }
      }

   } // ctns

#endif
