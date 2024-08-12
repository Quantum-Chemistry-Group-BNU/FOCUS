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
         if(debug){ 
            std::cout << "\nctns::oodmrg maxiter=" << maxiter 
               << " maxsweep=" << schd.ctns.maxsweep 
               << std::endl;
         }
         auto t0 = tools::get_time();
        
         // compute Hij first to get the initial Hij 
         auto Hij0 = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch);
         if(debug) Hij0.print("initial Hij", schd.ctns.outprec);

         const int norb = icomb.get_nphysical();
         std::vector<double> enew_history(maxiter);
         std::vector<double> emin_history(maxiter+1);
         emin_history[0] = std::real(Hij0(0,0));
         double e_min = emin_history[0];
         std::vector<double> u_history(maxiter);
         std::vector<bool> acceptance(maxiter,0);
         auto urot_min = linalg::identity_matrix<Tm>(norb);
         integral::one_body<Tm> int1e_new;
         integral::two_body<Tm> int2e_new;
         for(int iter=0; iter<maxiter; iter++){

            if(rank == 0){
               std::cout << std::endl;
               std::cout << tools::line_separator2 << std::endl;
               std::cout << "OO-DMRG: iter=" << iter << std::endl;
               std::cout << tools::line_separator2 << std::endl;
            }
            
            // minimize entanglement only at rank 0
            auto icomb_new = icomb;
            auto urot = urot_min;
            if(rank == 0 and iter != maxiter-1){
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
            auto Hij = ctns::get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
            
            // optimization
            auto result = ctns::sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);

            // accept or reject
            if(rank == 0){
               double e_new = result.get_eminlast(0);
               // urot = urot_min*U => U = urot_min.H()*urot
               auto u = linalg::xgemm("C","N",urot_min,urot) - linalg::identity_matrix<Tm>(norb);
               double u_diff = linalg::xnrm2(u.size(), u.data());
               // print
               std::cout << std::endl;
               std::cout << tools::line_separator << std::endl;
               std::cout << "OO-DMRG: iter=" << iter << std::setprecision(12)
                  << " e_new=" << e_new
                  << " e_min=" << e_min
                  << " |U-I|_F=" << std::setprecision(2) << u_diff 
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
               // check acceptance
               if(e_new < e_min or iter == maxiter-1){
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
               emin_history[iter+1] = e_min;
               u_history[iter] = u_diff;
               // display results
               Hij0.print("initial Hij",schd.ctns.outprec);
               std::cout << "summary of oodmrg results:" << std::endl;
               for(int jter=0; jter<=iter; jter++){
                  std::cout << " iter=" << jter
                     << " accept=" << acceptance[jter]
                     << std::defaultfloat << std::setprecision(12)  
                     << " e_new=" << enew_history[jter]
                     << " e_min=" << emin_history[jter+1]
                     << " de_min=" << std::scientific << std::setprecision(2)
                     << (emin_history[jter+1]-emin_history[jter])
                     << " u_diff=" << u_history[jter]
                     << std::endl;
               }
               std::cout << tools::line_separator << std::endl;
            } // rank-0
         } // iter
  
         // save integrals and urot_min
         if(rank == 0){
            std::cout << "save urot_min to fname = urot.bin" << std::endl;
            urot_min.save("urot.bin");
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
