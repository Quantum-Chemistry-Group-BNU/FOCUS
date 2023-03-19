#ifndef OPER_RENORM_KERNEL_H
#define OPER_RENORM_KERNEL_H

#include "oper_functors.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{
        
   template <typename Tm>
      void oper_renorm_kernel(const std::string superblock,
            const Hx_functors<Tm>& rfuns,
            const stensor3<Tm>& site,
            oper_dict<Tm>& qops,
            const int verbose){
         if(qops.mpirank==0 and verbose>1){
            std::cout << "ctns::oper_renorm_kernel"
               << " size[rfuns]=" << rfuns.size() 
               << std::endl;
         }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int i=0; i<rfuns.size(); i++){
            char key = rfuns[i].label[0];
            int index = rfuns[i].index; 
            if(verbose>2){
               std::cout << "rank=" << qops.mpirank 
                  << " idx=" << i 
                  << " key=" << key 
                  << " index=" << index 
                  << std::endl;
            }
            auto opxwf = rfuns[i]();
            auto op = contract_qt3_qt3(superblock, site, opxwf);
            linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
         } // i
      }

   template <typename Tm>
      Hx_functors<Tm> oper_renorm_functors(const std::string superblock,
            const stensor3<Tm>& site,
            const integral::two_body<Tm>& int2e,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const oper_dict<Tm>& qops,
            const bool ifdist1){
         Hx_functors<Tm> rfuns;
         // opC
         if(qops.oplist.find('C') != std::string::npos){
            auto info = oper_combine_opC(qops1.cindex, qops2.cindex);
            for(const auto& pr : info){
               int index = pr.first, iformula = pr.second;
               Hx_functor<Tm> Hx("C", index, iformula);
               Hx.opxwf = bind(&oper_normxwf_opC<Tm>, 
                     std::cref(superblock), std::cref(site), 
                     std::cref(qops1), std::cref(qops2),
                     index, iformula, false);
               rfuns.push_back(Hx);
            }
         }
         // opA
         if(qops.oplist.find('A') != std::string::npos){
            auto ainfo = oper_combine_opA(qops1.cindex, qops2.cindex, qops.ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first, iformula = pr.second;
               int iproc = distribute2(qops.ifkr, qops.mpisize, index);
               if(iproc == qops.mpirank){
                  Hx_functor<Tm> Hx("A", index, iformula);
                  Hx.opxwf = bind(&oper_normxwf_opA<Tm>, 
                        std::cref(superblock), std::cref(site), 
                        std::cref(qops1), std::cref(qops2),
                        index, iformula, false);
                  rfuns.push_back(Hx);
               }
            }
         }
         // opB
         if(qops.oplist.find('B') != std::string::npos){
            auto binfo = oper_combine_opB(qops1.cindex, qops2.cindex, qops.ifkr);
            for(const auto& pr : binfo){
               int index = pr.first, iformula = pr.second;
               int iproc = distribute2(qops.ifkr, qops.mpisize, index);
               if(iproc == qops.mpirank){
                  Hx_functor<Tm> Hx("B", index, iformula);
                  Hx.opxwf = bind(&oper_normxwf_opB<Tm>, 
                        std::cref(superblock), std::cref(site), 
                        std::cref(qops1), std::cref(qops2),
                        index, iformula, false);
                  rfuns.push_back(Hx);
               }
            }
         }
         // opP
         if(qops.oplist.find('P') != std::string::npos){
            for(const auto& pr : qops('P')){
               int index = pr.first;
               Hx_functor<Tm> Hx("P", index);
               Hx.opxwf = bind(&oper_compxwf_opP<Tm>,
                     std::cref(superblock), std::cref(site),
                     std::cref(qops1), std::cref(qops2), std::cref(int2e), 
                     index, false);
               rfuns.push_back(Hx);
            }
         }
         // opQ
         if(qops.oplist.find('Q') != std::string::npos){
            for(const auto& pr : qops('Q')){
               int index = pr.first;
               Hx_functor<Tm> Hx("Q", index);
               Hx.opxwf = bind(&oper_compxwf_opQ<Tm>,
                     std::cref(superblock), std::cref(site),
                     std::cref(qops1), std::cref(qops2), std::cref(int2e), 
                     index, false);
               rfuns.push_back(Hx);
            }
         }
         // opS
         if(qops.oplist.find('S') != std::string::npos){
            for(const auto& pr : qops('S')){
               int index = pr.first;
               Hx_functor<Tm> Hx("S", index);
               Hx.opxwf = bind(&oper_compxwf_opS<Tm>,
                     std::cref(superblock), std::cref(site),
                     std::cref(qops1), std::cref(qops2), std::cref(int2e),
                     index, qops.mpisize, qops.mpirank, ifdist1, false);
               rfuns.push_back(Hx);
            }
         }
         // opH
         if(qops.oplist.find('H') != std::string::npos){
            Hx_functor<Tm> Hx("H");
            Hx.opxwf = bind(&oper_compxwf_opH<Tm>, 
                  std::cref(superblock), std::cref(site),
                  std::cref(qops1), std::cref(qops2),
                  qops.mpisize, qops.mpirank, ifdist1);
            rfuns.push_back(Hx);
         }
         return rfuns;
      }

} // ctns

#endif
