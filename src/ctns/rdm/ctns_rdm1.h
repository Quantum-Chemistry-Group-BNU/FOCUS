#ifndef CTNS_RDM1_H
#define CTNS_RDM1_H

#include "rdm_env.h"
#include "rdm_util.h"
#include "../sweep_init.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <typename Qm, typename Tm>
      void get_rdm1(const bool is_same,
            const comb<Qm,Tm>& combi,
            const comb<Qm,Tm>& combj,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm1){
         const int dots = 1;
         // copy MPS
         auto icomb = combi;
         auto icomb2 = combj;
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const bool ifab = Qm::ifabelian;
         const int alg_rdm = schd.ctns.alg_rdm;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "\nctns::get_rdm1 ifab=" << ifab
               << " alg_rdm=" << alg_rdm
               << " alg_renorm=" << alg_renorm
               << " mpsize=" << size
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         auto t0 = tools::get_time();

         const int order = 1; // 1-RDM

         // prepare environments {C,D} [both are required for icomb != icomb2]
         rdm_env_right(order, is_same, icomb, icomb2, schd, scratch);

         // build operators on the left dot
         rdm_init_dotL(order, icomb, schd, scratch);

         // initialization of single MPS
         sweep_init_single(icomb, schd.ctns.iroot, schd.ctns.singlet);
         sweep_init_single(icomb2, schd.ctns.jroot, schd.ctns.singlet);

         // assemble RDM by sweep
         auto tpatterns = all_type_patterns(order);
         auto fpatterns = all_first_type_patterns(order);
         auto lpatterns = all_last_type_patterns(order);

         // pool for handling operators
         qoper_pool<Qm::ifabelian,Tm> qops_pool(schd.ctns.iomode, debug && schd.ctns.verbose>1);
         // generate sweep sequence
         auto sweep_seq = icomb.topo.get_mps_rdmsweeps(debug);
         std::vector<input::params_sweep> ctrls = {{0,1,icomb.get_dmax(),0.0,0.0}};
         std::vector<input::params_sweep> ctrls2 = {{0,1,icomb2.get_dmax(),0.0,0.0}};
         sweep_data sweeps(sweep_seq, 1, 1, 0, ctrls); 
         sweep_data sweeps2(sweep_seq, 1, 1, 0, ctrls); 
         // loop over sites
         const int isweep = 0;
         for(int ibond=0; ibond<sweep_seq.size(); ibond++){
            auto t0x = tools::get_time();
            const auto& dbond = sweep_seq[ibond];
            assert(dbond.forward);
            auto tp0 = icomb.topo.get_type(dbond.p0);
            auto tp1 = icomb.topo.get_type(dbond.p1);
            std::string superblock;
            if(dbond.forward){
               superblock = dbond.is_cturn()? "lr" : "lc";
            }else{
               superblock = "cr";
            }
            if(debug){
               std::cout << "\nibond=" << ibond << "/seqsize=" << sweep_seq.size()
                  << " dots=" << dots << " dbond=" << dbond
                  << " superblock=" << superblock
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
            }
            auto& timing = sweeps.opt_timing[0][ibond];

            // check partition
            auto dims = icomb.topo.check_partition(dots, dbond, debug, schd.ctns.verbose);

            // load operators
            auto fneed = icomb.topo.get_fqops(dots, dbond, scratch, debug && schd.ctns.verbose>0);
            qops_pool.fetch_to_memory(fneed, alg_rdm>10 || alg_renorm>10);
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
            }

            // 1.5 look ahead for the next dbond
            auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
            auto frop = fbond.first;
            auto fdel = fbond.second;
            auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
            // prefetch files for the next bond
            if(schd.ctns.async_fetch){
               if(alg_rdm>10 && alg_renorm>10){
                  const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
                  qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper);
               }
               qops_pool[frop]; // just declare a space for frop
               qops_pool.fetch_to_cpumem(fneed_next, schd.ctns.async_fetch); // just to cpu
            }
            timing.ta = tools::get_time();
            timing.tb = tools::get_time();

            // load MPS central site (use copy to avoid some problems in guess, where cpsi is changed)
            auto wf3bra = icomb.cpsi[0];
            auto wf3ket = icomb2.cpsi[0];
            size_t ndimbra = wf3bra.size();
            size_t ndimket = wf3ket.size();
            if(debug){
               std::cout << "wf3bra(diml,dimr,dimc)=("
                  << wf3bra.info.qrow.get_dimAll() << ","
                  << wf3bra.info.qcol.get_dimAll() << ","
                  << wf3bra.info.qmid.get_dimAll() << ")"
                  << " nnz=" << ndimbra << ":"
                  << tools::sizeMB<Tm>(ndimbra) << "MB"
                  << std::endl;
               wf3bra.print("wf3bra",schd.ctns.verbose-2);
               std::cout << "wf3ket(diml,dimr,dimc)=("
                  << wf3ket.info.qrow.get_dimAll() << ","
                  << wf3ket.info.qcol.get_dimAll() << ","
                  << wf3ket.info.qmid.get_dimAll() << ")"
                  << " nnz=" << ndimket << ":"
                  << tools::sizeMB<Tm>(ndimket) << "MB"
                  << std::endl;
               wf3ket.print("wf3ket",schd.ctns.verbose-2);
            }

            // assemble rdms
            // spin-recoupling
            // reorder indices to physical
            std::cout << std::endl;
            std::cout << "Assemble RDMs ..." << std::endl;
            std::cout << std::endl;
            timing.tc = tools::get_time();

            // propagtion of MPS via decimation
            qtensor2<Qm::ifabelian,Tm> rotbra, rotket;
            // bra
            linalg::matrix<Tm> vsolbra(ndimbra, 1, wf3bra.data());
            onedot_decimation(icomb, schd, scratch, sweeps, isweep, ibond,
                  superblock, vsolbra, wf3bra, rotbra);
#ifndef SERIAL
            if(size > 1) mpi_wrapper::broadcast(icomb.world, rotbra, 0);
#endif
            onedot_guess_psi(superblock, icomb, dbond, vsolbra, wf3bra, rotbra);
            vsolbra.clear();
            // ket
            if(is_same){
               auto cpsi = icomb.cpsi[0];
               icomb2.cpsi[0] = std::move(cpsi);
               auto rot = rotbra;
               rotket = std::move(rotket);
            }else{
               linalg::matrix<Tm> vsolket(ndimket, 1, wf3ket.data());
               onedot_decimation(icomb2, schd, scratch, sweeps2, isweep, ibond,
                     superblock, vsolket, wf3ket, rotket);
#ifndef SERIAL
               if(size > 1) mpi_wrapper::broadcast(icomb2.world, rotket, 0);
#endif
               onedot_guess_psi(superblock, icomb2, dbond, vsolket, wf3ket, rotket);
               vsolket.clear();
            }
            timing.td = tools::get_time();
            timing.te = tools::get_time();

            // save site and renormalize operators
            auto& qops  = qops_pool[frop];
            const auto& lqops = qops_pool.at(fneed[0]);
            const auto& rqops = qops_pool.at(fneed[1]);
            const auto& cqops = qops_pool.at(fneed[2]);
            const auto p = dbond.get_current();
            const auto& pdx = icomb.topo.rindex.at(p);
            std::string fname;
            std::string fmmtask;
            if(superblock == "lc"){
               icomb.sites[pdx] = rotbra.split_lc(wf3bra.info.qrow, wf3bra.info.qmid);
               icomb2.sites[pdx] = rotket.split_lc(wf3ket.info.qrow, wf3ket.info.qmid);
               qops_pool.clear_from_memory({fneed[1]}, fneed_next);
               rdm_renorm(order, "lc", is_same, icomb, icomb2, p, schd,
                     lqops, cqops, qops, fname, timing, fmmtask);
            }else if(superblock == "cr"){
               icomb.sites[pdx] = rotbra.split_cr(wf3bra.info.qmid, wf3bra.info.qcol);
               icomb2.sites[pdx] = rotket.split_cr(wf3ket.info.qmid, wf3ket.info.qcol);
               qops_pool.clear_from_memory({fneed[0]}, fneed_next);
               rdm_renorm(order, "cr", is_same, icomb, icomb2, p, schd,
                     cqops, rqops, qops, fname, timing, fmmtask);
            }else{
               tools::exit("error: superblock=lr is not supported yet!");
            }
            timing.tf = tools::get_time();

            // 4. cleanup operators
            qops_pool.cleanup_sweep(fneed, fneed_next, frop, fdel, schd.ctns.async_save, schd.ctns.async_remove);

            timing.t1 = tools::get_time();
            if(debug){
               get_sys_status();
               timing.analysis("rdm", schd.ctns.verbose>0);
            }
         } // ibond
         qops_pool.finalize();

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::get_rdm1", t0, t1);
         }
      }

} // ctns

#endif
