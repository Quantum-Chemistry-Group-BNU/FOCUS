#ifndef CTNS_OODMRG_H
#define CTNS_OODMRG_H

#include "../../core/integral_io.h"
#include "../../core/integral_rotate.h"
#include "oodmrg_move.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void oodmrg(comb<Qm,Tm>& icomb, 
            integral::two_body<Tm>& int2e,
            integral::one_body<Tm>& int1e,
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
         const bool acceptall = schd.ctns.ooparams.acceptall; 
         if(debug){ 
            std::cout << "\nctns::oodmrg maxiter=" << maxiter 
               << " maxsweep=" << schd.ctns.maxsweep
               << " acceptall=" << acceptall 
               << std::endl;
         }
         auto t0 = tools::get_time();
        
         // compute Hij first to get the initial Hij 
         auto Hij0 = get_Hmat(icomb, int2e, int1e, ecore, schd, scratch);
         double e0gs = std::real(Hij0(0,0));
         if(debug) Hij0.print("initial Hij", schd.ctns.outprec);
         
         // start oodmrg optimization   
         const int norb = icomb.get_nphysical();
         std::vector<double> enew_history(maxiter);
         std::vector<double> emin_history(maxiter+1);
         emin_history[0] = e0gs;
         double e_min = emin_history[0];
         std::vector<double> u_history(maxiter);
         std::vector<bool> acceptance(maxiter,0);
         auto urot_min = linalg::identity_matrix<Tm>(norb);
         double u_diff;
         integral::one_body<Tm> int1e_new;
         integral::two_body<Tm> int2e_new;
         for(int iter=0; iter<maxiter; iter++){

            if(rank == 0){
               std::cout << std::endl;
               std::cout << tools::line_separator2 << std::endl;
               std::cout << "OO-DMRG: maxiter=" << maxiter << " iter=" << iter;
               if(iter != maxiter-1){
                  std::cout << std::endl;
               }else{
                  std::cout << " last iter without orbital rotation" << std::endl;
               }
               std::cout << tools::line_separator2 << std::endl;
            }

            // minimize entanglement only at rank 0
            auto icomb_new = icomb; 
            auto urot = urot_min;
            if(rank == 0 and (iter != maxiter-1 and iter != 0)){
               // we assume that icomb has already been available, 
               // which is usually the case with a initial MPS from
               // SCI or a previous optimization.
               u_diff = oodmrg_move(icomb_new, urot, schd);
            }else{
               u_diff = 0.0;
            }
#ifndef SERIAL
            if(size > 1){
               mpi_wrapper::broadcast(schd.world, icomb_new, 0);
            }
#endif
            // update integrals
            if(rank == 0){
               rotate_spatial(int1e, int1e_new, urot);
               rotate_spatial(int2e, int2e_new, urot);
            }
#ifndef SERIAL
            if(size > 1){
               boost::mpi::broadcast(schd.world, int1e_new, 0);
               mpi_wrapper::broadcast(schd.world, int2e_new, 0);
            }
#endif

            // prepare environment
            auto Hij = get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
            if(rank == 0) Hij.print("Hij", schd.ctns.outprec);

            // optimization
            std::string rcfprefix = "oo_";
            auto result = sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch, rcfprefix);

            // accept or reject
            if(rank == 0){
               // check acceptance
               std::string status;
               double e_new = result.get_eminlast(0);
               double deltaE = e_new - e_min;
               if(deltaE < 0 or iter == maxiter-1 or acceptall){
                  status = "Accept the move!";
                  acceptance[iter] = true;
                  e_min = e_new;
                  urot_min = urot;
                  icomb = std::move(icomb_new); // move is defined, but copy is deleted
               }else{
                  // urot_min and icomb in the next iter will 
                  // still be the old one without change.
                  status = "Reject the move!";
               }
               enew_history[iter] = e_new;
               emin_history[iter+1] = e_min;
               u_history[iter] = u_diff;

               // display results
               std::cout << std::endl;
               std::cout << tools::line_separator2 << std::endl;
               std::cout << "OO-DMRG: iter=" << iter << std::setprecision(12)
                  << " dcut=" << schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut
                  << std::scientific << std::setprecision(2)
                  << " deltaE=" << deltaE << " " << status
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
               std::cout << "initial ground-state energy=" 
                  << std::defaultfloat << std::setprecision(12) << e0gs 
                  << std::endl;
               std::cout << "summary of oodmrg results:" << std::endl;
               for(int jter=0; jter<=iter; jter++){
                  if(jter == maxiter-1) std::cout << "final check:" << std::endl;
                  std::cout << " iter=" << jter
                     << " accept=" << acceptance[jter]
                     << std::defaultfloat << std::setprecision(12)  
                     << " e_new=" << enew_history[jter]
                     << " e_min=" << emin_history[jter+1]
                     << std::scientific << std::setprecision(2)
                     << " de_i=" << std::setw(9) << (emin_history[jter+1]-emin_history[jter])
                     << " de_0=" << std::setw(9) << (emin_history[jter+1]-emin_history[0])
                     << " u_diff=" << std::setw(8) << u_history[jter]
                     << std::endl;
               }
               std::cout << tools::line_separator2 << std::endl;
            } // rank-0
         } // iter
  
         // save integrals and urot_min
         if(rank == 0){
            std::cout << "save OO-DMRG results for later calculations:" << std::endl;
            urot_min.save_text("urot", schd.ctns.outprec);
            std::string fname = schd.integral_file+".new";
            integral::save(int2e_new, int1e_new, ecore, fname);
         }

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg", t0, t1);
         }
      }

   } // ctns

#endif
