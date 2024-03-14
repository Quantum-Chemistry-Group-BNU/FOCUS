#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "../qtensor/qtensor.h"
#include "sweep_util.h"
#include "sweep_onedot_renorm.h"
#include "sweep_onedot_diag.h"
#include "sweep_onedot_local.h"
#include "sweep_onedot_sigma.h"
#include "symbolic_formulae_onedot.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"
#include "preprocess_size.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma_batch.h"
#include "sadmrg/sweep_onedot_diag_su2.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
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
         const int alg_hvec = schd.ctns.alg_hvec;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_onedot"
               << " alg_hvec=" << alg_hvec
               << " alg_renorm=" << alg_renorm
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
         }
         auto& timing = sweeps.opt_timing[isweep][ibond];
         timing.t0 = tools::get_time();

         // check partition 
         const auto& dbond = sweeps.seq[ibond];
         icomb.topo.check_partition(1, dbond, debug, schd.ctns.verbose);

         // 1. load operators 
         auto fneed = icomb.topo.get_fqops(1, dbond, scratch, debug && schd.ctns.verbose>0);
         qops_pool.fetch_to_memory(fneed, alg_hvec>10 || alg_renorm>10);
         const qoper_dictmap<Qm::ifabelian,Tm> qops_dict = {
            {"l",qops_pool.at(fneed[0])},
            {"r",qops_pool.at(fneed[1])},
            {"c",qops_pool.at(fneed[2])}
         };
         if(debug && schd.ctns.verbose>0){
            std::cout << "qops info: rank=" << rank << std::endl;
            qops_dict.at("l").print("lqops");
            qops_dict.at("r").print("rqops");
            qops_dict.at("c").print("cqops");
            size_t tsize = qops_dict.at("l").size()
               + qops_dict.at("r").size()
               + qops_dict.at("c").size();
            std::cout << " qops(tot)=" << tsize 
               << ":" << tools::sizeMB<Tm>(tsize) << "MB"
               << ":" << tools::sizeGB<Tm>(tsize) << "GB"
               << std::endl;
         }
         timing.ta = tools::get_time();

         // 1.5 look ahead for the next dbond
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);

         // 2. onedot wavefunction
         //	    |
         //   --*--
         const auto& ql = qops_dict.at("l").qket;
         const auto& qr = qops_dict.at("r").qket;
         const auto& qc = qops_dict.at("c").qket;
         auto sym_state = get_qsym_state(Qm::isym, schd.nelec, (Qm::ifabelian? schd.twoms : schd.twos));
         qtensor3<Qm::ifabelian,Tm> wf(sym_state, ql, qr, qc, dir_WF3); // su2 case: by default, CRcouple is used.
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
         double* diag = new double[ndim];
         onedot_diag(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1);
#ifndef SERIAL
         // reduction of partial diag: no need to broadcast, if only rank=0 
         // executes the preconditioning in Davidson's algorithm
         if(size > 1){
            mpi_wrapper::reduce(icomb.world, diag, ndim, 0);
         }
#endif 
         std::transform(diag, diag+ndim, diag,
               [&ecore](const double& x){ return x+ecore; });
         timing.tb = tools::get_time();

         // 3.2 Solve local problem: Hc=cE
         // prepare HVec
         std::map<qsym,qinfo3type<Qm::ifabelian,Tm>> info_dict;
         size_t opsize, wfsize, tmpsize, worktot;
         opsize = preprocess_opsize<Qm::ifabelian,Tm>(qops_dict);
         wfsize = preprocess_wfsize<Qm::ifabelian,Tm>(wf.info, info_dict);
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
            + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         HVec_type<Tm> HVec;
         Hx_functors<Tm> Hx_funs; // hvec0
         symbolic_task<Tm> H_formulae; // hvec1,2
         bipart_task<Tm> H_formulae2; // hvec3
         hintermediates<Qm::ifabelian,Tm> hinter; // hvec4,5,6
         Hxlist<Tm> Hxlst; // hvec4
         Hxlist2<Tm> Hxlst2; // hvec5
         HMMtasks<Tm> Hmmtasks; // hvec6
         Tm scale = Qm::ifkr? 0.5*ecore : 1.0*ecore;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* ptrs[5] = {qops_dict.at("l")._data, qops_dict.at("r")._data, qops_dict.at("c")._data, 
            nullptr, nullptr};
         Tm* workspace;
         using std::placeholders::_1;
         using std::placeholders::_2;
         const bool debug_formulae = schd.ctns.verbose>0;

         // consistency check
         if(schd.ctns.ifdistc && !icomb.topo.ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }
         if(alg_hvec >=4){
            std::cout << "error: alg_hvec >=4 does not support onedot yet!" << std::endl;
            exit(1); 
         }
         if(alg_hvec < 10 && schd.ctns.alg_hinter == 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter<2" << std::endl;
            exit(1);
         }
         if(alg_hvec > 10 && schd.ctns.alg_hinter != 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter=2" << std::endl;
            exit(1);
         }

         if(alg_hvec == 0){

            // oldest version
            Hx_funs = onedot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
                  schd.ctns.ifdist1, debug_formulae);
            HVec = bind(&ctns::onedot_Hx<Qm::ifabelian,Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            HVec = bind(&ctns::symbolic_Hx<Qm::ifabelian,Tm,stensor3<Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 2){

            // symbolic formulae + preallocation of workspace 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            HVec = bind(&ctns::symbolic_Hx2<Qm::ifabelian,Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            H_formulae2 = symbolic_formulae_onedot2(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            HVec = bind(&ctns::symbolic_Hx3<Qm::ifabelian,Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else{
            std::cout << "error: no such option for alg_hvec=" << alg_hvec << std::endl;
            exit(1);
         } // alg_hvec

         // solve HC=CE
         linalg::matrix<Tm> vsol(ndim,neig);
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.dot_start();
         onedot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2, 
               ndim, neig, diag, HVec, eopt, vsol, nmvp, wf, timing);
         if(debug){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
         }
         timing.tc = tools::get_time();

         // free temporary space
         delete[] diag;
         if(alg_hvec >=2){
            delete[] workspace;
         }

         // 3. decimation & renormalize operators
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         onedot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_pool, fneed, fneed_next, frop,
               sweeps, isweep, ibond);
         timing.tf = tools::get_time();

         // 4. save on disk
         if(debug){
            get_sys_status();
            icomb.display_size();
         }
         auto t0 = tools::get_time();
         qops_pool.join_and_erase(fneed, fneed_next); 
         auto t1 = tools::get_time();
         qops_pool.save_to_disk(frop, schd.ctns.async_save, schd.ctns.alg_renorm>10 && schd.ctns.async_tocpu, fneed_next);
         auto t2 = tools::get_time();
         qops_pool.remove_from_disk(fdel, schd.ctns.async_remove);
         auto t3 = tools::get_time();
         if(debug){
            std::cout << "TIMING FOR cleanup: " << tools::get_duration(t3-t0)
               << " T(join&erase/save/remove)="
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << ","
               << tools::get_duration(t3-t2)
               << std::endl;
         }

         // save for restart
         if(rank == 0 && schd.ctns.timestamp) sweep_save(icomb, schd, scratch, sweeps, isweep, ibond);

         timing.t1 = tools::get_time();
         if(debug){
            get_sys_status();
            timing.analysis("local", schd.ctns.verbose>0);
         }
      }

} // ctns

#endif
