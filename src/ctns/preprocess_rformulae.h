#ifndef PREPROCESS_RFORMULAE_H
#define PREPROCESS_RFORMULAE_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"

namespace ctns{

   template <bool ifab, typename Tm, typename QTm>
      void preprocess_formulae_Rlist(const bool ifDirect,
            const int alg_rcoper,
            const std::string superblock,
            const qoper_dict<ifab,Tm>& qops,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const renorm_tasks<Tm>& rtasks,
            const QTm& site,
            const rintermediates<ifab,Tm>& rinter,
            Rlist<Tm>& Rlst,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         Rlist2<Tm> Rlst2;
         preprocess_formulae_Rlist2(ifDirect, alg_rcoper, superblock, qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
               Rlst2, blksize, blksize0, cost, debug);
         size_t size = 0;
         for(int i=0; i<Rlst2.size(); i++){
            size += Rlst2[i].size();
         }
         Rlst.reserve(size);
         for(int i=0; i<Rlst2.size(); i++){
            std::move(Rlst2[i].begin(), Rlst2[i].end(), std::back_inserter(Rlst));
         }

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR preprocess_formulae_Rlist : "
               << tools::get_duration(t1-t0) << " S"
               << " size(rtasks)=" << rtasks.size()
               << " size(Rlst)=" << Rlst.size() << " -----"
               << std::endl;
         }
      }

   template <bool ifab, typename Tm, typename QTm>
      void preprocess_formulae_Rlist2(const bool ifDirect,
            const int alg_rcoper,
            const std::string superblock,
            const qoper_dict<ifab,Tm>& qops,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const renorm_tasks<Tm>& rtasks,
            const QTm& site,
            const rintermediates<ifab,Tm>& rinter,
            Rlist2<Tm>& Rlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Rmu
         int rsize = rtasks.size();
         std::vector<std::vector<Rmu_ptr<ifab,Tm>>> Rmu(rsize);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            Rmu[k].resize(formula.size());
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].rinfo = const_cast<qinfo2type<ifab,Tm>*>(&qops(key).at(index).info);
               Rmu[k][it].offrop = qops._offset.at(std::make_pair(key,index));
               Rmu[k][it].init(ifDirect, k, it, formula, qops_dict, rinter, oploc);
            }
         } // it
         auto ta = tools::get_time();

         // 2. from Rmu to expanded block forms
         blksize = 0;
         blksize0 = 0;
         cost = 0.0;
         int nnzblk = qops.qbra.size(); // partitioned according to rows
         Rlst2.resize(nnzblk);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].gen_Rlist2(alg_rcoper, opaddr, superblock, site.info, Rlst2, blksize, blksize0, cost, false);
               if(key == 'H'){
                  Rmu[k][it].gen_Rlist2(alg_rcoper, opaddr, superblock, site.info, Rlst2, blksize, blksize0, cost, true);
               }
            }
         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR preprocess_formulae_Rlist2 : "
               << tools::get_duration(t1-t0) << " S"
               << " size(rtasks)=" << rsize
               << " size(Rlst2)=" << nnzblk
               << " T(Rmu/Rlist)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
