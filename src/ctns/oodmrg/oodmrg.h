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
         const int dcut = schd.ctns.maxsweep>0? schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut : -1;
         const bool acceptall = schd.ctns.ooparams.acceptall;
         const double alpha = schd.ctns.ooparams.alpha;
         if(debug){ 
            std::cout << "\nctns::oodmrg"
               << " maxiter=" << maxiter 
               << " maxsweep=" << schd.ctns.maxsweep
               << " dcut=" << dcut
               << " acceptall=" << acceptall
               << " alpha=" << alpha 
               << std::endl;
         }
         auto t0 = tools::get_time();

         // initialization
         const int norb = icomb.get_nphysical();
         std::vector<double> enew_history(maxiter);
         std::vector<double> emin_history(maxiter+1);
         std::vector<double> sdiag_history(maxiter+1,0);
         std::vector<double> srenyi_history(maxiter+1,0);
         double e_min;
         std::vector<double> u_history(maxiter);
         std::vector<bool> acceptance(maxiter,0);
         auto urot_min = linalg::identity_matrix<Tm>(norb);
         integral::one_body<Tm> int1e_new;
         integral::two_body<Tm> int2e_new;

         // check whether twodot_rotate is correct?
         const bool debug_rotate = true;
         if(debug_rotate){
            auto icomb_new = icomb; 
            auto urot = urot_min; 
            //auto Hij0 = get_Hmat(icomb_new, int2e, int1e, ecore, schd, scratch);
            //Hij0.print("Hij0_rank"+std::to_string(rank),10);
            if(rank == 0){
               const int dfac = schd.ctns.ooparams.dfac;
               const int dmax = dfac*icomb_new.get_dmax();
               //std::vector<int> gates = {64};
               /*
               std::vector<int> gates = {0,1,2,3,4,5,6,7,8,9,10,11,12,
               13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
               31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
               49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66};
               std::vector<int> gates = {31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
               49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66};
               */
               
               std::vector<int> gates = {
                  0,1,2,3,4,5,6,7,8,9,10,11,12,
                  13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                  31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
               49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66};
               
               double maxdwt = reduce_entropy_single(icomb_new, urot, "random", dmax, schd.ctns.ooparams, gates);
               std::cout << "maxdwt=" << maxdwt << std::endl;
               //reduce_entropy_single(icomb_new, urot, "opt", dmax, schd.ctns.ooparams);
              
               auto Sd1 = rcanon_Sdiag_sample(icomb, 0, schd.ctns.nsample, -1, 10);
               auto Sd2 = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, -1, 10);
              
               rcanon_lastdots(icomb_new);
               
               auto Sd3 = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, -1, 10);
               /*
               rotate_spatial(int1e, int1e_new, urot);
               rotate_spatial(int2e, int2e_new, urot);
               */ 
               for(int i=0; i<urot.rows(); i++){
                  for(int j=0; j<urot.cols(); j++){
                     if(std::abs(urot(i,j))>1.e-10 and i!=j){
                        std::cout << "urot i=" << i << " j=" << j << " uij=" << urot(i,j) << std::endl;
                     }
                  }
               }
               auto diff = linalg::check_orthogonality(urot);
               std::cout << "diff=" << diff << std::endl;

               auto pr = rcanon_random(icomb,0);
               auto pr2 = rcanon_random(icomb_new,0);
               auto state = pr.first;
               auto state2 = pr2.first;

               for(int i=0; i<urot.rows(); i++){
                  std::cout << "i=" << i << " state=" << state[2*i+1] << state[2*i]
                     << " state2=" << state2[2*i+1] << state[2*i]
                     << std::endl;
               } 
               exit(1);
               
            }
#ifndef SERIAL
            if(size > 1){
               mpi_wrapper::broadcast(schd.world, icomb_new, 0);
               boost::mpi::broadcast(schd.world, int1e_new, 0);
               mpi_wrapper::broadcast(schd.world, int2e_new, 0);
            }
#endif
/*
            auto Hij1 = get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
            Hij0.print("Hij0_rank"+std::to_string(rank),10);
            Hij1.print("Hij1_rank"+std::to_string(rank),10);
            auto Sij0 = get_Smat(icomb);
            auto Sij1 = get_Smat(icomb_new);
            Sij0.print("Sij0_rank"+std::to_string(rank),10);
            Sij1.print("Sij1_rank"+std::to_string(rank),10);
#ifndef SERIAL
            icomb.world.barrier();
#endif

            auto diffH = Hij1 - Hij0;
            diffH.print("diffH",10);
            auto Sd1 = rcanon_Sdiag_sample(icomb, 0, schd.ctns.nsample, -1, 10);
            auto Sd2 = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, -1, 10);
*/
            exit(1);
         }

         // start optimization
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
            if(iter != maxiter-1 and iter != 0){
               // we assume that icomb has already been available, 
               // which is usually the case with a initial MPS from
               // SCI or a previous optimization.
               if(rank == 0) u_history[iter] = oodmrg_move(icomb_new, urot, schd);
            }else{
               u_history[iter] = 0.0;
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
            if(rank == 0){
               Hij.print("Hij0_iter"+std::to_string(iter), schd.ctns.outprec);
               // save the initial ground-state energy
               if(iter == 0){
                  e_min = std::real(Hij(0,0));
                  emin_history[0] = e_min;
                  // compute initial entropy
                  auto Sd = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, -1, 10);
                  sdiag_history[0] = Sd;
                  auto Sr = rcanon_entropysum(icomb_new, alpha);
                  srenyi_history[0] = Sr;
               }
            }

            // optimization of MPS with new integrals
            std::string rcfprefix = "oo_";
            auto result = sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch, rcfprefix);
            if(rank == 0){   
               auto Sd = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, -1, 10);
               sdiag_history[iter+1] = Sd;
               auto Sr = rcanon_entropysum(icomb_new, alpha);
               srenyi_history[iter+1] = Sr; 
            }

            const bool debug_rotate2 = true;
            if(debug_rotate2){
               auto Hij0 = get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
               Hij0.print("Hij0_rank"+std::to_string(rank),10);
               if(rank == 0){
                  const int dfac = schd.ctns.ooparams.dfac;
                  const int dmax = dfac*icomb_new.get_dmax();
                  std::vector<int> gates; // = {0,1,2,3};
                  double maxdwt = reduce_entropy_single(icomb_new, urot, "random", dmax, schd.ctns.ooparams, gates);
                  std::cout << "maxdwt=" << maxdwt << std::endl;
                  //reduce_entropy_single(icomb_new, urot, "opt", dmax, schd.ctns.ooparams);
                  rcanon_lastdots(icomb_new);
                  rotate_spatial(int1e, int1e_new, urot);
                  rotate_spatial(int2e, int2e_new, urot);
               }
#ifndef SERIAL
               if(size > 1){
                  mpi_wrapper::broadcast(schd.world, icomb_new, 0);
                  boost::mpi::broadcast(schd.world, int1e_new, 0);
                  mpi_wrapper::broadcast(schd.world, int2e_new, 0);
               }
#endif
               auto Hij1 = get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
               Hij0.print("Hij0_rank"+std::to_string(rank),10);
               Hij1.print("Hij1_rank"+std::to_string(rank),10);
#ifndef SERIAL
               icomb.world.barrier();
#endif
               exit(1);
            }

            // accept or reject
            if(rank == 0){
               // check acceptance
               std::string status;
               double e_new = result.get_eminlast(0);
               double deltaE = e_new - e_min;
               if(deltaE < 0 or iter == maxiter-1 or acceptall){
                  status = "accept move!";
                  acceptance[iter] = true;
                  e_min = e_new;
                  urot_min = urot;
                  icomb = std::move(icomb_new); // move is defined, but copy is deleted
               }else{
                  // urot_min and icomb in the next iter will 
                  // still be the old one without change.
                  status = "reject move!";
               }
               enew_history[iter] = e_new;
               emin_history[iter+1] = e_min;

               // display results
               std::cout << std::endl;
               std::cout << tools::line_separator2 << std::endl;
               std::cout << "OO-DMRG results:"
                  << " iter=" << iter
                  << " alpha=" << alpha
                  << " dcut=" << dcut
                  << std::scientific << std::setprecision(2)
                  << " DE=" << deltaE << " " << status
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
               std::cout << "initial ground-state energy=" 
                  << std::defaultfloat << std::setprecision(10) << emin_history[0]
                  << std::scientific << std::setprecision(2) 
                  << " Sd=" << sdiag_history[0]
                  << " Sr=" << srenyi_history[0] 
                  << std::endl;
               for(int jter=0; jter<=iter; jter++){
                  if(jter == 1) std::cout << "oodmrg steps:" << std::endl;
                  if(jter == maxiter-1) std::cout << "final check:" << std::endl;
                  std::cout << " iter=" << jter << ":" << acceptance[jter]
                     << std::fixed << std::setprecision(10)  
                     << " E_i=" << enew_history[jter]
                     << " E_min=" << emin_history[jter+1]
                     << std::scientific << std::setprecision(2)
                     << " DE=" << std::setw(9) << (enew_history[jter]-emin_history[jter]) // deltaE
                     << " LE=" << std::setw(9) << (emin_history[jter+1]-emin_history[0]) // loweringE
                     << " Sd=" << std::setw(8) << sdiag_history[jter+1]
                     << " Sr=" << std::setw(8) << srenyi_history[jter+1]
                     << " |U_i-I|=" << std::setw(8) << u_history[jter]
                     << std::endl;
               }
               std::cout << tools::line_separator2 << std::endl;

               // save the current best results
               if(acceptance[iter]){
                  std::string rcanon_file = schd.scratch+"/oo_rcanon_d"+std::to_string(dcut); 
                  if(!Qm::ifabelian) rcanon_file += "_su2";
                  rcanon_save(icomb, rcanon_file);
                  urot_min.save_text("urot", schd.ctns.outprec);
               }
            } // rank-0
         } // iter

         // save integrals for urot_min
         if(rank == 0){
            rotate_spatial(int1e, int1e_new, urot_min);
            rotate_spatial(int2e, int2e_new, urot_min);
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
