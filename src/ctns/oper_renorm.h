#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#ifdef _OPENMP
#include <omp.h>
#endif
#include <type_traits>
#include "ctns_sys.h"
#include "sweep_data.h"
#include "oper_timer.h"
#include "oper_rbasis.h"
#include "oper_reduce.h"
#include "oper_ab2pq.h"
#include "sweep_renorm.h"

namespace ctns{

   const bool debug_oper_renorm = false;
   extern const bool debug_oper_renorm;

   const double thresh_opdiff = 1.e-9;
   extern const double thresh_opdiff;

   // renormalize operators
   // ndots only matter for ifab2pq=true
   template <typename Qm, typename Tm>
      void oper_renorm(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const std::string fname,
            dot_timing& timing,
            const std::string fmmtask,
            const int ndots=2){
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         auto oplist = oper_renorm_oplist(superblock, icomb, pcoord, schd, ndots);
         const bool ifdist1 = schd.ctns.ifdist1;
         const bool ifdistc = schd.ctns.ifdistc;
         const int sorb = icomb.get_nphysical()*2;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool ifab = Qm::ifabelian;
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::oper_renorm coord=" << pcoord 
               << " superblock=" << superblock 
               << " oplist=" << oplist
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
         const auto& node = icomb.topo.get_node(pcoord);
         const auto& rindex = icomb.topo.rindex;
         const auto& site = icomb.sites[rindex.at(pcoord)];
         if(superblock == "cr"){
            qops.krest = node.lorbs;
            qops.qbra = site.info.qrow;
            qops.qket = site.info.qrow;
            assert(check_consistency(site.info.qmid, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
         }else if(superblock == "lc"){
            qops.krest = node.rorbs;
            qops.qbra = site.info.qcol;
            qops.qket = site.info.qcol;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qmid, qops2.qbra));
         }else if(superblock == "lr"){
            qops.krest = node.corbs;
            qops.qbra = site.info.qmid;
            qops.qket = site.info.qmid;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
         }
         qops.oplist = oplist;
         qops.mpisize = size;
         qops.mpirank = rank;
         qops.ifdist2 = true;
         // initialize
         qops.init();
         if(debug){ 
            qops.print("qops", schd.ctns.verbose-1);
            get_sys_status();
         }

         // 1. kernel for renormalization
         oper_timer.dot_start();
         Renorm_wrapper<Qm,Tm,qtensor3<ifab,Tm>> Renorm;
         const bool is_same = true;
         const bool skipId = true;
         Renorm.kernel(superblock, is_same, skipId, icomb.topo.ifmps, site, site,
               qops1, qops2, qops, int2e, schd, size, rank, maxthreads, timing, fname, fmmtask);

         // 2.1 reduction of opS and opH on GPU
#ifndef SERIAL
         if(ifdist1 and size>1 and schd.ctns.ifnccl){
            reduce_opSH_gpu(qops, alg_renorm, ifkr, size, rank);
         }
#endif
         timing.tf10 = tools::get_time();

#ifdef GPU
         // send back to CPU
         if(alg_renorm>10){
            auto t0x = tools::get_time();
            qops.to_cpu();
            auto t1x = tools::get_time();
            double dt = tools::get_duration(t1x-t0x); 
            if(rank == 0){
               std::cout << "qops.to_cpu: size(tot)=" << qops.size()
                  << ":" << tools::sizeMB<Tm>(qops.size()) << "MB" 
                  << ":" << tools::sizeGB<Tm>(qops.size()) << "GB"
                  << " t[to_cpu]=" << dt << "S"
                  << " speed=" << tools::sizeGB<Tm>(qops.size())/dt << "GB/S"
                  << std::endl;
            } 
         }
#endif        
         timing.tf11 = tools::get_time();

         Renorm.finalize();
         timing.tf12 = tools::get_time();

         // debug_oper_renorm 
         if(debug_oper_renorm && rank == 0){
            if(schd.ctns.ifnccl){
               std::cout << "error: debug should not be invoked with ifnccl!" << std::endl;
               exit(1);
            }
            const int target = -1;
            std::cout << "\nqops: rank=" << rank << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank
                     << " key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cnew",2);
               }
            }
            Tm* data0 = new Tm[qops._size];
            linalg::xcopy(qops._size, qops._data, data0);

            // alg_renorm=2: symbolic formulae + preallocation of workspace
            memset(qops._data, 0, qops._size*sizeof(Tm));
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, schd.ctns.sort_formulae, ifdist1, ifdistc, schd.ctns.verbose>0);
            symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, skipId, ifdist1, schd.ctns.verbose);
            std::cout << "\nqops[ref]: rank=" << rank << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank
                     << " key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second[ref]=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cref",2);
               }
            }
            Tm* data1 = new Tm[qops._size];
            linalg::xcopy(qops._size, qops._data, data1);

            linalg::xaxpy(qops._size, -1.0, data0, qops._data);
            auto diff = linalg::xnrm2(qops._size, qops._data);
            std::cout << "\nqops[diff]: rank=" << rank << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank 
                     << " key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second[diff]=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cdiff",2);
               }
            }
            std::cout << "rank=" << rank << " total diff=" << diff << std::endl;
            linalg::xcopy(qops._size, data0, qops._data);
            delete[] data0;
            delete[] data1;
            if(diff > thresh_opdiff) exit(1);
         } // debug

         // 2.2 reduction of opS and opH on CPU and send back to GPU 
#ifndef SERIAL
         if(ifdist1 and size>1 and !schd.ctns.ifnccl){
            reduce_opSH_cpu(qops, icomb, alg_renorm, size, rank);
         } 
#endif 

         // qops is available on CPU, consistency check
         {
            // 3. consistency check for Hamiltonian
            const auto& opH = qops('H').at(0);
            
            //debug:
            //opH.to_matrix().print("lzd opH",10);
            //if(pcoord.first == 15) exit(1);
            
            // NAN check
            for(int i=0; i<opH.size(); i++){
               double Hr = std::real(opH._data[i]);
               double Hi = std::imag(opH._data[i]);
               if(std::isnan(Hr) or std::isnan(Hi)){
                  std::cout << "error: opH contains NAN at rank=" << rank << std::endl;
                  exit(1);
               }
            }
            // Hermicity check
            auto diffH = (opH-opH.H()).normF();
            if(debug){
               std::cout << "check ||H-H.dagger||=" << std::scientific << std::setprecision(3) << diffH 
                  << " coord=" << pcoord << " rank=" << rank << std::defaultfloat << std::endl;
            } 
            if(diffH > thresh_opdiff){
               std::cout <<  "error in oper_renorm: ||H-H.dagger||=" << std::scientific << std::setprecision(3) << diffH 
                  << " is larger than thresh_opdiff=" << thresh_opdiff 
                  << " for rank=" << rank 
                  << std::endl;
               exit(1);
            }
            // check against explicit construction
            if(debug_oper_rbasis){
               for(const auto& key : qops.oplist){
                  if(key == 'C' || key == 'A' || key == 'B'){
                     oper_check_rbasis(icomb, icomb, pcoord, qops, key, size, rank);
                  }else if(key == 'P' || key == 'Q'){
                     oper_check_rbasis(icomb, icomb, pcoord, qops, key, int2e, int1e, size, rank);
                     // check opS and opH only if ifdist1=true [opS and opH collected on rank-0]
                     // or size==1 [serial version, no matter what value ifdist1 is] 
                  }else if((key == 'S' || key == 'H') and (size == 1 || ifdist1)){
                     oper_check_rbasis(icomb, icomb, pcoord, qops, key, int2e, int1e, size, rank, ifdist1);
                  }
               }
            }
         } // end of consistency check
         timing.tf13 = tools::get_time();

         if(schd.ctns.ifab2pq){
            oper_ab2pq(superblock, icomb, pcoord, int2e, schd, qops, ndots);
         }

         timing.tf14 = tools::get_time();
         if(debug){
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
            double t_tot = tools::get_duration(timing.tf13-timing.tf0); 
            std::cout << "----- TIMING FOR oper_renorm : " << t_tot << " S" 
               << " rank=" << rank << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
