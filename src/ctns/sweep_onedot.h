#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
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
#include "preprocess_sigma2.h"
#include "preprocess_sigma_batch.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   // onedot optimization algorithm
   template <typename Km>
      void sweep_onedot(comb<Km>& icomb,
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            oper_pool<typename Km::dtype>& qops_pool,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         using Tm = typename Km::dtype;
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_onedot"
               << " alg_hvec=" << schd.ctns.alg_hvec
               << " alg_renorm=" << schd.ctns.alg_renorm
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
         }
         auto& memory = sweeps.opt_memory[isweep][ibond];
         auto& timing = sweeps.opt_timing[isweep][ibond];
         timing.t0 = tools::get_time();

         // check partition 
         const auto& dbond = sweeps.seq[ibond];
         icomb.topo.check_partition(1, dbond, debug, schd.ctns.verbose);

         // 1. load operators 
         auto fneed = icomb.topo.get_fqops(1, dbond, scratch, debug && schd.ctns.verbose>0);
         qops_pool.fetch(fneed);
         const oper_dictmap<Tm> qops_dict = {{"l",qops_pool(fneed[0])},
            {"r",qops_pool(fneed[1])},
            {"c",qops_pool(fneed[2])}};
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
         if(debug){
            memory.comb = sizeof(Tm)*icomb.display_size(); 
            memory.oper = sizeof(Tm)*qops_pool.size(); 
            memory.display();
         }
         timing.ta = tools::get_time();

         // 2. onedot wavefunction
         //	  |
         //   --*--
         const auto& ql = qops_dict.at("l").qket;
         const auto& qr = qops_dict.at("r").qket;
         const auto& qc = qops_dict.at("c").qket;
         auto sym_state = get_qsym_state(Km::isym, schd.nelec, schd.twoms);
         stensor3<Tm> wf(sym_state, ql, qr, qc, dir_WF3);
         size_t ndim = wf.size();
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

         // 3. Davidson solver for wf
         // 3.1 diag 
         std::vector<double> diag(ndim, ecore/size); // constant term
         onedot_diag(qops_dict, wf, diag.data(), size, rank, schd.ctns.ifdist1);
#ifndef SERIAL
         // reduction of partial diag: no need to broadcast, if only rank=0 
         // executes the preconditioning in Davidson's algorithm
         if(size > 1){
            std::vector<double> diag2(ndim);
            mpi_wrapper::reduce(icomb.world, diag.data(), ndim, diag2.data(), std::plus<double>(), 0);
            diag = std::move(diag2);
         }
#endif 
         timing.tb = tools::get_time();

         // 3.2 Solve local problem: Hc=cE
         // prepare HVec
         std::map<qsym,qinfo3<Tm>> info_dict;
         size_t opsize, wfsize, tmpsize, worktot;
         opsize = preprocess_opsize(qops_dict);
         wfsize = preprocess_wfsize(wf.info, info_dict);
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
               + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         HVec_type<Tm> HVec;
         Hx_functors<Tm> Hx_funs; // hvec0
         symbolic_task<Tm> H_formulae; // hvec1,2
         bipart_task<Tm> H_formulae2; // hvec3
         intermediates<Tm> inter; // hvec4,5,6
         Hxlist<Tm> Hxlst; // hvec4
         Hxlist2<Tm> Hxlst2; // hvec5
         MMtasks<Tm> mmtasks; // hvec6
         Tm scale = Km::ifkr? 0.5*ecore : 1.0*ecore;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* ptrs[5] = {qops_dict.at("l")._data, qops_dict.at("r")._data, qops_dict.at("c")._data, 
            nullptr, nullptr};
         Tm* workspace;
         using std::placeholders::_1;
         using std::placeholders::_2;
         const bool debug_formulae = schd.ctns.verbose>0;
         if(schd.ctns.alg_hvec >=4){
            std::cout << "inter does not support onedot yet!" << std::endl;
            exit(1); 
         }
         if(schd.ctns.alg_hvec == 0){

            // oldest version
            Hx_funs = onedot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
                  schd.ctns.ifdist1, debug_formulae);
            HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(schd.ctns.alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                  debug_formulae); 
            HVec = bind(&ctns::symbolic_Hx<Tm,stensor3<Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(schd.ctns.alg_hvec == 2){

            // symbolic formulae + preallocation of workspace 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                  debug_formulae); 
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            memory.hvec = sizeof(Tm)*worktot;
            HVec = bind(&ctns::symbolic_Hx2<Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(schd.ctns.alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            H_formulae2 = symbolic_formulae_onedot2(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                  debug_formulae); 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            memory.hvec = sizeof(Tm)*worktot;
            HVec = bind(&ctns::symbolic_Hx3<Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else{
            std::cout << "error: no such option for alg_hvec=" << schd.ctns.alg_hvec << std::endl;
            exit(1);
         } // alg_hvec

         // solve HC=CE
         int neig = sweeps.nroots;
         linalg::matrix<Tm> vsol(ndim,neig);
         if(debug){
            memory.dvdson += sizeof(Tm)*ndim*(neig + std::min(ndim,size_t(neig+schd.ctns.nbuff))*3);
            memory.display();
         }
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.start();
         onedot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2, 
               ndim, neig, diag, HVec, eopt, vsol, nmvp, wf);
         if(debug && schd.ctns.verbose>0){
            sweeps.print_eopt(isweep, ibond);
            if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
         }
         timing.tc = tools::get_time();

         // free temporary space
         if(schd.ctns.alg_hvec >=2){
            delete[] workspace;
            memory.hvec = 0; 
         }

         // 3. decimation & renormalize operators
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         onedot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_dict, qops_pool(frop), 
               sweeps, isweep, ibond);
         if(debug){
            memory.comb = sizeof(Tm)*icomb.display_size();
            memory.oper = sizeof(Tm)*qops_pool.size();
            memory.display();
         }
         timing.tf = tools::get_time();

         // 4. save on disk 
         qops_pool.save(frop);
         /* NOTE: At the boundary case [ -*=>=*-* and -*=<=*-* ],
                  removing in the later configuration must wait until 
                  the file from the former configuration has been saved!
                  Therefore, oper_remove must come later than save,
                  which contains the synchronization!
         */
         oper_remove(fdel, debug);

         timing.t1 = tools::get_time();
         if(debug) timing.analysis("time_local", schd.ctns.verbose>0);
      }

} // ctns

#endif
