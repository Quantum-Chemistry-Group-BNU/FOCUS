#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "sweep_util.h"
#include "sweep_onedot_diag.h"
#include "sadmrg/sweep_onedot_diag_su2.h"
#include "sweep_onedot_local.h"
#include "sweep_onedot_renorm.h"
#include "sweep_hvec.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "sweep_onedot_diagGPU.h"
#include "sadmrg/sweep_onedot_diagGPU_su2.h"
#endif

namespace ctns{

   // onedot optimization algorithm
   template <typename Qm, typename Tm>
      void sweep_onedot(comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            qoper_pool<Qm::ifabelian,Tm>& qops_pool,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
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
            std::cout << "ctns::sweep_onedot"
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
         const int dots = 1;
         const auto& dbond = sweeps.seq[ibond];
         auto dims = icomb.topo.check_partition(dots, dbond, debug, schd.ctns.verbose);

         // 1. load operators 
         auto fneed = icomb.topo.get_fqops(dots, dbond, scratch, debug && schd.ctns.verbose>0);
         qops_pool.fetch_to_memory(fneed, alg_hvec>10 || alg_renorm>10);
         const qoper_dictmap<ifab,Tm> qops_dict = {
            {"l",qops_pool.at(fneed[0])},
            {"r",qops_pool.at(fneed[1])},
            {"c",qops_pool.at(fneed[2])}
         };
         size_t opertot = qops_dict.at("l").size()
            + qops_dict.at("r").size()
            + qops_dict.at("c").size();
         if(debug && schd.ctns.verbose>0){
            std::cout << "qops info: rank=" << rank << std::endl;
            qops_dict.at("l").print("lqops");
            qops_dict.at("r").print("rqops");
            qops_dict.at("c").print("cqops");
            std::cout << " qops(tot)=" << opertot
               << ":" << tools::sizeMB<Tm>(opertot) << "MB"
               << ":" << tools::sizeGB<Tm>(opertot) << "GB"
               << std::endl;
            get_mem_status(rank);
         }

         // 1.5 look ahead for the next dbond
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
         // prefetch files for the next bond
         if(alg_hvec>10 && alg_renorm>10){
            const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
            qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper, qops_pool.frop_prev);
            if(debug){
               get_mem_status(rank, 0, "after clear_from_cpumem");
            }
         }
         timing.ta = tools::get_time();

         // 2. onedot wavefunction
         //	    |
         //   --*--
         const auto& ql = qops_dict.at("l").qket;
         const auto& qr = qops_dict.at("r").qket;
         const auto& qc = qops_dict.at("c").qket;
         auto sym_state = get_qsym_state(Qm::isym, schd.nelec, 
               (ifab? schd.twom : schd.twos),
               schd.ctns.singlet);
         qtensor3<ifab,Tm> wf;
         if(ifab){
            // abelian case
            wf.init(sym_state, ql, qr, qc, dir_WF3);
         }else{
            // su2 case
            spincoupling3 couple;
            if(dims[0] <= dims[1]){
               couple = CRcouple; // l|cr
            }else{
               couple = LCcouple; // lc|r
            }
            wf.init(sym_state, ql, qr, qc, dir_WF3, couple);
         }
         size_t ndim = wf.size();
         int neig = sweeps.nroots;
         if(debug){
            std::cout << "wf3(diml,dimr,dimc)=(" 
               << ql.get_dimAll() << ","
               << qr.get_dimAll() << ","
               << qc.get_dimAll() << ")"
               << " nnz=" << ndim  << ":"
               << tools::sizeMB<Tm>(ndim) << "MB"
               << std::endl;
            wf.print("wf3",schd.ctns.verbose-2);
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
            onedot_diag(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1);
#ifdef GPU
         }else{
            onedot_diagGPU(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1, schd.ctns.ifnccl, schd.ctns.diagcheck);
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
         HVec_wrapper<Qm,Tm,qinfo3type<ifab,Tm>,qtensor3<ifab,Tm>> HVec;
         HVec.init(dots, icomb.topo.ifmps, qops_dict, int2e, ecore, schd, size, rank, maxthreads, 
               ndim, wf, timing, fname, fmmtask);

         //-------------
         // solve HC=CE
         //-------------
         linalg::matrix<Tm> vsol(ndim,neig);
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.dot_start();
         onedot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2, 
               ndim, neig, diag, HVec.Hx, eopt, vsol, nmvp, wf, timing);
         // free tmp space on CPU & GPU
         delete[] diag;
         HVec.finalize();
         if(debug){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
            get_mem_status(rank);
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
               if(debug){
                  get_mem_status(rank, 0, "after clear_from_cpumem");
               }
            }
            qops_pool[frop]; // just declare a space for frop
            qops_pool.fetch_to_cpumem(fneed_next, schd.ctns.async_fetch); // just to cpu
            if(debug){
               get_mem_status(rank, 0, "after fetch_to_cpumem");
            }
         }

         // 3. decimation & renormalize operators
         onedot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_pool, fneed, fneed_next, frop,
               sweeps, isweep, ibond);
         timing.tf = tools::get_time();

         // 4. cleanup operators
         qops_pool.cleanup_sweep(frop, fdel, schd.ctns.async_save, schd.ctns.async_remove);

         timing.t1 = tools::get_time();
         if(debug){
            get_mem_status(rank);
            timing.analysis("local opt", schd.ctns.verbose>0);
         }
      }

} // ctns

#endif
