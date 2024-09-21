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
         const double thrdeps = schd.ctns.ooparams.thrdeps;
         if(debug){ 
            std::cout << "\nctns::oodmrg"
               << " maxiter=" << maxiter 
               << " maxsweep=" << schd.ctns.maxsweep
               << " dcut=" << dcut
               << " acceptall=" << acceptall
               << " alpha=" << alpha
               << " thrdeps=" << thrdeps 
               << std::endl;
         }
         auto t0 = tools::get_time();

         // initialization
         const int norb = icomb.get_nphysical();
         std::vector<double> enew_history(maxiter);
         std::vector<double> sdnew_history(maxiter,0);
         std::vector<double> srnew_history(maxiter,0);
         std::vector<double> emin_history(maxiter+1);
         std::vector<double> sdiag_history(maxiter+1,0);
         std::vector<double> srenyi_history(maxiter+1,0);
         std::vector<double> u_history(maxiter);
         std::vector<bool> acceptance(maxiter);
         auto urot_min = linalg::identity_matrix<Tm>(norb);
         integral::one_body<Tm> int1e_new;
         integral::two_body<Tm> int2e_new;

         //----------------------------------------------
         // Debug rotation
         //----------------------------------------------
         const bool debug_rotate = false;
         if(debug_rotate){
            auto icomb_new = icomb; 
            auto urot = urot_min; 
            auto Hij0 = get_Hmat(icomb_new, int2e, int1e, ecore, schd, scratch);
            Hij0.print("Hij0_rank"+std::to_string(rank),10);
            if(rank == 0){
               const int dfac = schd.ctns.ooparams.dfac;
               const int dmax = dfac*icomb_new.get_dmax();
               std::vector<int> gates;
               double maxdwt = reduce_entropy_single(icomb_new, urot, "random", dmax, schd.ctns.ooparams, gates);
               std::cout << "maxdwt=" << maxdwt << std::endl;
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
            auto Sij0 = get_Smat(icomb);
            auto Sij1 = get_Smat(icomb_new);
            Sij0.print("Sij0_rank"+std::to_string(rank),10);
            Sij1.print("Sij1_rank"+std::to_string(rank),10);
#ifndef SERIAL
            icomb.world.barrier();
#endif
            auto diffH = Hij1 - Hij0;
            diffH.print("diffH",10);
            exit(1);
         }
         //----------------------------------------------

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
                  emin_history[0] = std::real(Hij(0,0));
                  // compute initial entropy
                  sdiag_history[0] = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, schd.ctns.pthrd, schd.ctns.nprt);
                  srenyi_history[0] = rcanon_entropysum(icomb_new, alpha);
               }
            }

            // optimization of MPS with new integrals
            std::string rcfprefix = "oo_";
            auto result = sweep_opt(icomb_new, int2e_new, int1e_new, ecore, schd, scratch, rcfprefix);
            if(rank == 0){
               sdnew_history[iter] = rcanon_Sdiag_sample(icomb_new, 0, schd.ctns.nsample, schd.ctns.pthrd, schd.ctns.nprt);
               srnew_history[iter] = rcanon_entropysum(icomb_new, alpha); 
            }

            //----------------------------------------------
            // Debug rotation
            //----------------------------------------------
            const bool debug_rotate2 = false;
            if(debug_rotate2){
               auto Hij0 = get_Hmat(icomb_new, int2e_new, int1e_new, ecore, schd, scratch);
               Hij0.print("Hij0_rank"+std::to_string(rank),10);
               if(rank == 0){
                  const int dfac = schd.ctns.ooparams.dfac;
                  const int dmax = dfac*icomb_new.get_dmax();
                  std::vector<int> gates; // = {0,1,2,3};
                  double maxdwt = reduce_entropy_single(icomb_new, urot, "random", dmax, schd.ctns.ooparams, gates);
                  std::cout << "maxdwt=" << maxdwt << std::endl;
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
               auto diffH = Hij1 - Hij0;
               diffH.print("diffH",10);
               exit(1);
            }
            //----------------------------------------------

            // accept or reject
            if(rank == 0){
               // check acceptance
               std::string status;
               enew_history[iter] = result.get_eminlast(0);
               double deltaE = enew_history[iter] - emin_history[iter];
               bool accept = (iter == maxiter-1) or (iter == 0) or 
                  acceptall or (deltaE <= -thrdeps) or 
                  (std::abs(deltaE)<thrdeps and srnew_history[iter]<srenyi_history[iter]);
               if(accept){
                  status = "accept move!";
                  acceptance[iter] = true;
                  emin_history[iter+1] = enew_history[iter]; 
                  sdiag_history[iter+1] = sdnew_history[iter];
                  srenyi_history[iter+1] = srnew_history[iter];
                  urot_min = urot;
                  icomb = std::move(icomb_new); // move is defined, but copy is deleted
               }else{
                  // urot_min and icomb in the next iter will 
                  // still be the old one without change.
                  status = "reject move!";
                  acceptance[iter] = false;
                  emin_history[iter+1] = emin_history[iter];
                  sdiag_history[iter+1] = sdiag_history[iter];
                  srenyi_history[iter+1] = srenyi_history[iter];
               }

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
               std::cout << "initial energy of MPS Hii= " 
                  << std::defaultfloat << std::setprecision(schd.ctns.outprec) << emin_history[0]
                  << std::scientific << std::setprecision(2) 
                  << " Sd= " << sdiag_history[0]
                  << " Sr= " << srenyi_history[0] 
                  << std::endl;
               std::cout << "  iter accept      enew            emin        deltaE    lowerE"
                  << "   sdnew    sdiag    srnew    srenyi   |Ui-I|" 
                  << std::endl; 
               for(int jter=0; jter<=iter; jter++){
                  std::cout << std::setw(5) << jter << "    " << acceptance[jter] << "   "
                     << std::fixed << std::setprecision(schd.ctns.outprec) 
                     << enew_history[jter] << " "
                     << emin_history[jter+1] << " "
                     << std::scientific << std::setprecision(2)
                     << std::setw(9) << (enew_history[jter]-emin_history[jter]) << " " // deltaE
                     << std::setw(9) << (emin_history[jter+1]-emin_history[1]) << " " // loweringE
                     << std::setw(8) << sdnew_history[jter] << " "
                     << std::setw(8) << sdiag_history[jter+1] << " "
                     << std::setw(8) << srnew_history[jter] << " "
                     << std::setw(8) << srenyi_history[jter+1] << " "
                     << std::setw(8) << u_history[jter]
                     << std::endl;
               }
               std::cout << tools::line_separator2 << std::endl;

               // save the current best results
               if(acceptance[iter]){
                  std::string rcanon_file = schd.scratch+"/oo_rcanon_iter"+std::to_string(iter); 
                  if(!Qm::ifabelian) rcanon_file += "_su2";
                  rcanon_save(icomb, rcanon_file);
                  urot_min.save_txt("urot", schd.ctns.outprec);
                  auto schmidt_file = schd.scratch+"/schmidt_values_iter"+std::to_string(iter);
                  ctns::rcanon_schmidt(icomb, schd.ctns.iroot, schmidt_file);
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
