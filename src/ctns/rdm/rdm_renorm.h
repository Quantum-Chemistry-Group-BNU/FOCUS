#ifndef RDM_RENORM_H
#define RDM_RENORM_H

#include "../sweep_renorm.h"
#include "../../core/mem_status.h"

namespace ctns{

   // renormalize operators
   template <typename Qm, typename Tm>
      void rdm_renorm(const int order,
            const std::string superblock,
            const bool is_same,
            const comb<Qm,Tm>& icomb,
            const comb<Qm,Tm>& icomb2,
            const comb_coord& p,
            const input::schedule& schd,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const std::string fname,
            dot_timing& timing,
            const std::string fmmtask=""){
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const int sorb = icomb.get_nphysical()*2;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool ifab = Qm::ifabelian;
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::rdm_renorm coord=" << p 
               << " superblock=" << superblock 
               << " is_same=" << is_same
               << " ifab=" << ifab
               << " isym=" << isym 
               << " ifkr=" << ifkr
               << " alg_renorm=" << alg_renorm	
               << " mpisize=" << size
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         timing.tf0 = tools::get_time(); 

         // 0. setup basic information for qops
         qops.sorb = sorb;
         qops.isym = isym;
         qops.ifkr = ifkr;
         qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
         // rest of spatial orbital indices
         const auto& node = icomb.topo.get_node(p);
         const auto& rindex = icomb.topo.rindex;
         const auto& site = icomb.sites[rindex.at(p)];
         const auto& site2 = icomb2.sites[rindex.at(p)];
         if(superblock == "cr"){
            //  ---*--- site
            //     |   \
            //     *    *
            //     |   /
            //  ---*--- site2
            qops.krest = node.lorbs;
            qops.qbra = site.info.qrow;
            qops.qket = site2.info.qrow;
            assert(check_consistency(site.info.qmid, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
            assert(check_consistency(site2.info.qmid, qops1.qket));
            assert(check_consistency(site2.info.qcol, qops2.qket));
         }else if(superblock == "lc"){
            qops.krest = node.rorbs;
            qops.qbra = site.info.qcol;
            qops.qket = site2.info.qcol;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qmid, qops2.qbra));
            assert(check_consistency(site2.info.qrow, qops1.qket));
            assert(check_consistency(site2.info.qmid, qops2.qket));
         }else if(superblock == "lr"){
            tools::exit("error: rdm_renorm does not support superblock=lr yet!");
            qops.krest = node.corbs;
            qops.qbra = site.info.qmid;
            qops.qket = site2.info.qmid;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
            assert(check_consistency(site2.info.qrow, qops1.qket));
            assert(check_consistency(site2.info.qcol, qops2.qket));
         }
         // RDMs:
         qops.ifhermi = is_same;
         if(order == 0){
            qops.oplist = "I";
         }else if(order == 1){
            qops.oplist = is_same? "IC" : "ICD";
         }else if(order == 2){
            qops.oplist = is_same? "ICAB" : "ICABDM";
         }else{
            std::cout << "error: rdm_renorm does not support order=" << order << std::endl;
            exit(1);
         }
         qops.mpisize = size;
         qops.mpirank = rank;
         qops.ifdist2 = true;
         // initialize
         qops.init();
         if(debug){ 
            qops.print("qops", schd.ctns.verbose-1);
            get_cpumem_status(rank);
	    get_gpumem_status(rank);
         }
      
         // 1. kernel for renormalization
         oper_timer.dot_start();
         // declare a fake int2e
         integral::two_body<Tm> int2e;
         Renorm_wrapper<Qm,Tm,qtensor3<ifab,Tm>> Renorm;
         const bool skipId = false;
         Renorm.kernel(superblock, is_same, skipId, icomb.topo.ifmps, site, site2,
               qops1, qops2, qops, int2e, schd, size, rank, maxthreads, timing, fname, fmmtask);
         
         Renorm.finalize();
         timing.tf10 = tools::get_time();
         timing.tf11 = tools::get_time();
         
#ifdef GPU
         // send back to CPU
         if(alg_renorm>10) qops.to_cpu();
#endif
         timing.tf12 = tools::get_time();
         timing.tf13 = tools::get_time();

         timing.tf14 = tools::get_time();
         if(debug){
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
            double t_tot = tools::get_duration(timing.tf14-timing.tf0); 
            std::cout << "----- TIMING FOR rdm_renorm : " << t_tot << " S" 
               << " rank=" << rank << " -----"
               << std::endl;
            get_cpumem_status(rank);
	    get_gpumem_status(rank);
         }
      }

} // ctns

#endif
