#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "sweep_util.h"
#include "sweep_twodot_diag.h"
#include "sadmrg/sweep_twodot_diag_su2.h"
#include "sweep_twodot_local.h"
#include "sweep_twodot_renorm.h"
#include "sweep_hvec.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "sweep_twodot_diagGPU.h"
#include "sadmrg/sweep_twodot_diagGPU_su2.h"
#endif

namespace ctns{

   // twodot optimization algorithm
   template <typename Qm, typename Tm>
      void sweep_twodot(comb<Qm,Tm>& icomb,
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
            std::cout << "ctns::sweep_twodot"
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
            get_cpumem_status(rank);
         }

         // 1.5 look ahead for the next dbond
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
         // prefetch files for the next bond
         if(schd.ctns.async_fetch){
            if(alg_hvec>10 && alg_renorm>10){
               const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
               qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper);
            }
            qops_pool[frop]; // just declare a space for frop
            qops_pool.fetch_to_cpumem(fneed_next, schd.ctns.async_fetch); // just to cpu
            if(debug) get_cpumem_status(rank);
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
         twodot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2,
               ndim, neig, diag, HVec.Hx, eopt, vsol, nmvp, wf, dbond, timing);
         // free tmp space on CPU & GPU
         delete[] diag;
         HVec.finalize();
         if(debug){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
            get_cpumem_status(rank);
         }
         timing.tc = tools::get_time();

         // 3. decimation & renormalize operators
         twodot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_pool, fneed, fneed_next, frop,
               sweeps, isweep, ibond);
         timing.tf = tools::get_time();

         // 4. save on disk 
         qops_pool.cleanup_sweep(frop, fdel, schd.ctns.async_save, schd.ctns.async_remove);
        
         timing.t1 = tools::get_time();
         if(debug){
            get_cpumem_status(rank);
            timing.analysis("local opt", schd.ctns.verbose>0);
         }
      }

} // ctns

#endif
