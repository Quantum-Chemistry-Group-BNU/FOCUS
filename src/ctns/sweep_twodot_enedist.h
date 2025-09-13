#ifndef SWEEP_TWODOT_ENEDIST_H
#define SWEEP_TWODOT_ENEDIST_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "sweep_util.h"
#include "sweep_twodot_diag.h"
#include "sadmrg/sweep_twodot_diag_su2.h"
#include "sweep_twodot_renorm.h"
#include "sweep_hvec.h"
#include "../qtensor/plinear.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "sweep_twodot_diagGPU.h"
#include "sadmrg/sweep_twodot_diagGPU_su2.h"
#endif

namespace ctns{

   // local CI solver	
   template <typename Qm, typename Tm>
      void twodot_local_linear(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const double eps,
            const int parity,
            const size_t ndim,
            const int neig,
            const double* diag,
            HVec_type<Tm> HVec,
            std::vector<double>& eopt,
            linalg::matrix<Tm>& vsol,
            int& nmvp,
            qtensor4<Qm::ifabelian,Tm>& wf,
            const directed_bond& dbond,
            dot_timing& timing,
            const std::vector<Tm>& rhs,
            const int icase,
            const double omegaR=0.0,
            const double omegaI=0.0){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // without kramers restriction
         plinearSolver_nkr<Tm> solver(ndim, neig, eps, schd.ctns.maxcycle, icase);
         solver.iprt = schd.ctns.verbose;
         solver.damping = schd.ctns.damping;
         solver.precond = schd.ctns.precond;
         solver.ifnccl = schd.ctns.ifnccl;
         solver.Diag = const_cast<double*>(diag);
         solver.HVec = HVec;
         //-----------------------------------
         // specification for linear equation
         //-----------------------------------
         solver.RHS = const_cast<Tm*>(rhs.data()); // RHS of A*x=b
         solver.omegaR = omegaR;
         solver.omegaI = omegaI;
#ifndef SERIAL
         solver.world = icomb.world;
#endif

         if(schd.ctns.cisolver == 0){

            // full diagonalization for debug
            solver.solve_diag(eopt.data(), vsol.data(), false);

         }else if(schd.ctns.cisolver == 1){ 

            // davidson
            if(schd.ctns.guess == 0){
               // davidson without initial guess
               solver.solve_iter(eopt.data(), vsol.data()); 
            }else if(schd.ctns.guess == 1){     
               //------------------------------------
               // prepare initial guess     
               //------------------------------------
               auto t0 = tools::get_time();
               std::vector<Tm> v0;
               if(rank == 0){
                  assert(icomb.cpsi.size() == neig);
                  // specific to twodot 
                  twodot_guess_v0(icomb, dbond, ndim, neig, wf, v0);
                  // reorthogonalization
                  int nindp = linalg::get_ortho_basis(ndim, neig, v0.data()); 
                  if(nindp != neig){
                     std::cout << "error: nindp=" << nindp << " does not match neig=" << neig << std::endl;
                     exit(1);
                  } 
               }
               //------------------------------------
               auto t1 = tools::get_time();
               timing.dtb5 = tools::get_duration(t1-t0);
               solver.solve_iter(eopt.data(), vsol.data(), v0.data());
            }else{
               std::cout << "error: no such option for guess=" << schd.ctns.guess << std::endl;
               exit(1);
            }

         }
         nmvp = solver.nmvp;
         timing.dtb6 = solver.t_cal - oper_timer.tcpugpu - oper_timer.tcommgpu; 
         timing.dtb7 = oper_timer.tcpugpu;
         timing.dtb8 = oper_timer.tcommgpu;
         timing.dtb9 = solver.t_comm;
         timing.dtb10 = solver.t_rest;
      }

   // twodot optimization algorithm for energy distribution P(E)
   template <typename Qm, typename Tm>
      void sweep_twodot_enedist(comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            qoper_pool<Qm::ifabelian,Tm>& qops_pool,
            sweep_data& sweeps,
            const int isweep,
            const int ibond,
            const comb<Qm,Tm>& icomb2,
            const comb<Qm,Tm>& licomb2,
            const std::vector<qtensor3<Qm::ifabelian,Tm>>& cpsis2,
            std::vector<qtensor2<Qm::ifabelian,Tm>>& environ){
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const bool ifab = Qm::ifabelian;
         const int alg_hvec = schd.ctns.alg_hvec;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_twodot_enedist"
               << " ifabelian=" << ifab
               << " alg_hvec=" << alg_hvec
               << " alg_renorm=" << alg_renorm
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
            icomb.display_size();
         }
         auto& timing = sweeps.opt_timing[isweep][ibond];
         timing.t0 = tools::get_time();

         // 0. check partition
         const int dots = 2; 
         const auto& dbond = sweeps.seq[ibond];
         icomb.topo.check_partition(dots, dbond, debug, schd.ctns.verbose);

         // 1. load operators
         auto fneed = icomb.topo.get_fqops(dots, dbond, scratch, debug && schd.ctns.verbose>0);
         qops_pool.fetch_to_memory(fneed, alg_hvec>10 || alg_renorm>10);
         const qoper_dictmap<ifab,Tm> qops_dict = {
            {"l" ,qops_pool.at(fneed[0])},
            {"r" ,qops_pool.at(fneed[1])},
            {"c1",qops_pool.at(fneed[2])},
            {"c2",qops_pool.at(fneed[3])}
         };
         size_t opertot = qops_dict.at("l").size()
            + qops_dict.at("r").size()
            + qops_dict.at("c1").size()
            + qops_dict.at("c2").size();
         if(debug && schd.ctns.verbose>0){
            std::cout << "qops info: rank=" << rank << std::endl;
            qops_dict.at("l").print("lqops");
            qops_dict.at("r").print("rqops");
            qops_dict.at("c1").print("c1qops");
            qops_dict.at("c2").print("c2qops");
            std::cout << " qops(tot)=" << opertot 
               << ":" << tools::sizeMB<Tm>(opertot) << "MB"
               << ":" << tools::sizeGB<Tm>(opertot) << "GB"
               << std::endl;
            get_mem_status(rank, 0, "sweep_twodot", alg_hvec>10 or alg_renorm>10);
         }

         // 1.5 look ahead for the next dbond
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
         if(alg_hvec>10 && alg_renorm>10 && !schd.ctns.diagcheck){
            const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
            qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper, qops_pool.frop_prev);
         }
         timing.ta = tools::get_time();

         // 2. twodot wavefunction
         //	 \ /
         //   --*--
         const auto& ql  = qops_dict.at("l").qket;
         const auto& qr  = qops_dict.at("r").qket;
         const auto& qc1 = qops_dict.at("c1").qket;
         const auto& qc2 = qops_dict.at("c2").qket;
         auto sym_state = get_qsym_state(Qm::isym, schd.nelec, 
               (ifab? schd.twom : schd.twos),
               schd.ctns.singlet);
         qtensor4<ifab,Tm> wf(sym_state, ql, qr, qc1, qc2);
         size_t ndim = wf.size();
         int neig = sweeps.nroots;
         if(debug){
            std::cout << "wf4(diml,dimr,dimc1,dimc2)=(" 
               << ql.get_dimAll() << ","
               << qr.get_dimAll() << ","
               << qc1.get_dimAll() << ","
               << qc2.get_dimAll() << ")"
               << " nnz=" << ndim << ":"
               << tools::sizeMB<Tm>(ndim) << "MB"
               << std::endl;
            wf.print("wf4",schd.ctns.verbose-2);
         }
         if(ndim == 0){
            std::cout << "error: symmetry is inconsistent as ndim=0" << std::endl;
            exit(1);
         }
         if(ndim < neig){
            std::cout << "error: ndim<neig! either neig is too large or dcut is too small." << std::endl;
            exit(1);
         }

         // 3. Davidson solver for wf
         // 3.1 diag 
         auto diag_t0 = tools::get_time();
         double* diag = new double[ndim];
         if(alg_hvec <= 10){
            twodot_diag(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1);
#ifdef GPU
         }else{
            twodot_diagGPU(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1, schd.ctns.ifnccl, schd.ctns.diagcheck);
#endif
         }
         auto diag_t1 = tools::get_time();
#ifndef SERIAL
         // reduction of partial diag: no need to broadcast, if only rank=0 
         // executes the preconditioning in Davidson's algorithm
         if(!schd.ctns.ifnccl && size > 1){
            mpi_wrapper::reduce(icomb.world, diag, ndim, 0);
         }
#endif 
         auto diag_t2 = tools::get_time();
         std::transform(diag, diag+ndim, diag,
               [&ecore](const double& x){ return x+ecore; });
         timing.tb = tools::get_time();
         auto diag_t3 = tools::get_time();
         if(rank==0){
            double duration_diagGPU = tools::get_duration(diag_t1-diag_t0);
            double duration_diagreduce = tools::get_duration(diag_t2-diag_t1);
            double duration_diagtransform = tools::get_duration(diag_t3-diag_t2);
            std::cout<<"duration_diagGPU:"<<duration_diagGPU<<std::endl;
            std::cout<<"duration_diagreduce:"<<duration_diagreduce<<std::endl;
            std::cout<<"duration_diagtransform:"<<duration_diagtransform<<std::endl;
         }

         // 3.2 prepare HVec & solve local problem: Hc=cE
         // formulae file
         std::string fname;
         if(schd.ctns.save_formulae){
            fname = scratch+"/hformulae"
               + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         }
         // fmmtask file
         std::string fmmtask;
         if(debug && schd.ctns.save_mmtask && isweep == schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond){
            fmmtask = "hmmtasks_isweep"+std::to_string(isweep) + "_ibond"+std::to_string(ibond);
         }
         HVec_wrapper<Qm,Tm,qinfo4type<ifab,Tm>,qtensor4<ifab,Tm>> HVec;
         HVec.init(dots, icomb.topo.ifmps, qops_dict, int2e, ecore, schd, size, rank, maxthreads, 
               ndim, wf, timing, fname, fmmtask);

         //-------------
         // solve HC=CE         
         //-------------
         linalg::matrix<Tm> vsol(ndim,neig);
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.dot_start();
 
         //====================================================================
         // TODOs:
         // 
         // 1. Linear equation & CG algorithm for solving [(H-E)^2+eta^2]|psi>=eta|psi0>
         //
         // 2. The most tricky part is the preparation of RHS.
         //
         // 3. Also add (H-E)*|psi> in renormalization? (add an option)
         //====================================================================
         /*
         twodot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2,
               ndim, neig, diag, HVec.Hx, eopt, vsol, nmvp, wf, dbond, timing);
         */

         //-------------------------------
         // construct RHS and solver Ax=b 
         //-------------------------------
         // adapted from twodot_guess_v0
         std::vector<Tm> rhs(ndim);
         auto pdx0 = icomb.topo.rindex.at(dbond.p0);
         auto pdx1 = icomb.topo.rindex.at(dbond.p1);
         if(dbond.forward){
            //  /        \
            //  *  |  |  *
            //  \--c--r--/
            const auto& cpsi = cpsis2[dbond.p0.first];
            const auto& lenv = environ[dbond.p0.first-1];
            auto cpsi_new = contract_qt3_qt2("l",cpsi,lenv); 
            const auto& renv = environ[dbond.p1.first+1];
            auto site_new = contract_qt3_qt2("r",icomb2.sites[pdx1],renv);
            // psi[l,a,c1] => cwf[lc1,a]
            auto cwf = cpsi_new.recouple_lc().merge_lc(); 
            // cwf[lc1,a]*r[a,r,c2] => wf3[lc1,r,c2]
            auto wf3 = contract_qt3_qt2("l",site_new,cwf); 
            // wf3[lc1,r,c2] => wf4[l,r,c1,c2]
            auto wf4 = wf3.split_lc1(wf.info.qrow, wf.info.qmid);
            assert(wf4.size() == ndim);
            wf4.to_array(rhs.data());
         }else{
            //  /        \
            //  *  |  |  *
            //  \--l--c--/
            const auto& cpsi = cpsis2[dbond.p1.first];
            const auto& renv = environ[dbond.p1.first+1];
            auto cpsi_new = contract_qt3_qt2("r",cpsi,renv);
            const auto& lenv = environ[dbond.p0.first-1];
            auto site_new = contract_qt3_qt2("l",licomb2.sites[pdx0],lenv); 
            // psi[a,r,c2] => cwf[a,c2r]
            auto cwf = cpsi_new.recouple_cr().merge_cr();
            // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
            auto wf3 = contract_qt3_qt2("r",site_new,cwf.P());
            // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
            auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
            assert(wf4.size() == ndim);
            wf4.to_array(rhs.data());
         } // forward
        
         // solve [(H-E)^2+eta^2]|psi>=-eta|psi2>
         const int icase = 0;
         twodot_local_linear(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2,
               ndim, neig, diag, HVec.Hx, eopt, vsol, nmvp, wf, dbond, timing,
               rhs, icase, schd.ctns.enedist[0], schd.ctns.enedist[1]);
         rhs.clear();
         //-------------------------------

         // free tmp space on CPU & GPU
         delete[] diag;
         HVec.finalize();
         if(debug){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
            get_mem_status(rank, 0, "after twodot_localCI", alg_hvec>10 or alg_renorm>10);
         }
         timing.tc = tools::get_time();

         // 2.3 prefetch files for the next bond
         // join save thread and set frop_prev = empty, such that
         // unused cpu memory can be fully released
         qops_pool.join_save();
         if(schd.ctns.async_fetch){
            // remove used cpu mem of fneed fully, which makes sure
            // that only 2 (rather than 3) qops are in cpumem!
            if(alg_hvec>10 && alg_renorm>10){
               const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
               qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper);
            }
            qops_pool[frop]; // just declare a space for frop
            qops_pool.fetch_to_cpumem(fneed_next, schd.ctns.async_fetch); // just to cpu
         }

         // 3. decimation & renormalize operators
         twodot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_pool, fneed, fneed_next, frop,
               sweeps, isweep, ibond);
         timing.tf = tools::get_time();

         //---------------------
         // update environments
         //---------------------
         if(dbond.forward){
            // /--new--
            // *   |
            // \--l2---
            const auto& cpsi = licomb2.sites[pdx0];
            const auto& lenv = environ[dbond.p0.first-1];
            auto cpsi_new = contract_qt3_qt2("l",cpsi,lenv); 
            environ[dbond.p0.first] = contract_qt3_qt3("lc",icomb.sites[pdx0],cpsi_new);
         }else{
            // --new--\
            //    |   *
            // ---r2--/
            const auto& cpsi = icomb2.sites[pdx1];
            const auto& renv = environ[dbond.p1.first+1];
            auto cpsi_new = contract_qt3_qt2("r",cpsi,renv);
            environ[dbond.p1.first] = contract_qt3_qt3("cr",icomb.sites[pdx1],cpsi_new);
         } 
         //---------------------
         
         // 4. save on disk 
         qops_pool.cleanup_sweep(frop, fdel, schd.ctns.async_save, schd.ctns.async_remove, schd.ctns.keepoper);

         // save for restart
         if(rank == 0 && schd.ctns.timestamp) sweep_save(icomb, schd, scratch, sweeps, isweep, ibond);

         timing.t1 = tools::get_time();
         if(debug){
            get_mem_status(rank, 0, "after sweep_twodot", alg_hvec>10 or alg_renorm>10);
            timing.analysis("local opt", schd.ctns.verbose>0);
         }
      }

} // ctns

#endif
