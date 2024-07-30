#ifndef CTNS_OODMRG_H
#define CTNS_OODMRG_H

#include "../../core/integral_rotate.h"
#include "oodmrg_move.h"

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
         std::vector<bool> acceptance(maxiter,0);
         double e_min = 1.e20;
         auto urot_min = linalg::identity_matrix<Tm>(norb);

         for(int iter=0; iter<maxiter; iter++){

            if(rank == 0){
               std::cout << tools::line_separator2 << std::endl;
               std::cout << "OO-DMRG: iter=" << iter << std::endl;
               std::cout << tools::line_separator2 << std::endl;
            }
            
            // minimize entanglement only at rank 0
            auto icomb_new = icomb;
            auto urot = urot_min;
            if(rank == 0){
               // we assume that icomb has already been available, 
               // which is usually the case with a initial MPS from
               // SCI or a previous optimization.
               oodmrg_move(icomb_new, urot, schd);
            }
#ifndef SERIAL
            if(size > 1){
               mpi_wrapper::broadcast(schd.world, icomb_new, 0);
            }
#endif

            // update integrals
            integral::one_body<Tm> int1e_new;
            integral::two_body<Tm> int2e_new;
            if(rank == 0){
               rotate_spatial(int1e, int1e_new, urot);
               rotate_spatial(int2e, int2e_new, urot);
            }
#ifndef SERIAL
            std::cout << "rank=" << rank << " size=" << size << std::endl;
            if(size > 1){
               boost::mpi::broadcast(schd.world, int1e_new, 0);
               mpi_wrapper::broadcast(schd.world, int2e_new, 0);
            }
#endif

            // prepare environment
            auto Hij = ctns::get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
            // optimization
            auto result = ctns::sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);

            // accept or reject
            if(rank == 0){
               double e_new = result.get_eminlast(0);
               // print
               std::cout << std::endl;
               std::cout << tools::line_separator << std::endl;
               std::cout << "OO-DMRG: iter=" << iter << std::setprecision(12)
                  << " e_new=" << e_new
                  << " e_min=" << e_min 
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
               // check acceptance
               if(e_new < e_min){
                  std::cout << "accept the move!" << std::endl;
                  acceptance[iter] = true;
                  e_min = e_new;
                  urot_min = urot;
                  icomb = std::move(icomb_new); // move is defined, but copy is deleted
               }else{
                  std::cout << "reject the move!" << std::endl;
                  // urot_min and icomb in the next iter will 
                  // still be the old one without change.
               }
               enew_history[iter] = e_new;
               emin_history[iter] = e_min;
               // display results
               std::cout << "summary of oodmrg results:" << std::endl;
               for(int jter=0; jter<=iter; jter++){
                  std::cout << " iter=" << jter
                     << " accept=" << acceptance[jter]
                     << std::defaultfloat << std::setprecision(12)  
                     << " e_new=" << enew_history[jter]
                     << " e_min=" << emin_history[jter]
                     << " de_min=" << std::scientific << std::setprecision(2)
                     << (emin_history[jter]-emin_history[iter])
                     << std::endl; 
               }
               std::cout << tools::line_separator << std::endl;
               std::cout << std::endl;
            } // rank-0

         } // iter

         // save e_min,urot_min,icomb?

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg", t0, t1);
         }
      }

   } // ctns

#endif
