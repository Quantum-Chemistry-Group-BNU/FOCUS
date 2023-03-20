#ifndef PREPROCESS_RFORMULAE_H
#define PREPROCESS_RFORMULAE_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"

namespace ctns{

   template <typename Tm, typename QTm>
      void preprocess_formulae_Rlist(const std::string superblock,
            const oper_dict<Tm>& qops,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            const renorm_tasks<Tm>& rtasks,
            const QTm& site,
            const rintermediates<Tm>& rinter,
            Rlist<Tm>& Rlst,
            size_t& blksize,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Rmu
         int rsize = rtasks.size();
         std::vector<std::vector<Rmu_ptr<Tm>>> Rmu(rsize);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            Rmu[k].resize(formula.size());
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].rinfo = const_cast<qinfo2<Tm>*>(&qops(key).at(index).info);
               Rmu[k][it].offrop = qops._offset.at(std::make_pair(key,index));
               Rmu[k][it].init(k, it, formula, qops_dict, rinter, oploc);
            }
         } // it
         auto ta = tools::get_time();

         // 2. from Rmu to Rxlst in expanded block forms
         blksize = 0;
         cost = 0.0;
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);

            //if(key != 'C' or index != 9) continue;
            //if(key != 'C') continue;

            /*
            if(key == 'B' and index == 10007){
               std::cout << "B10007" << " k=" << k << " key=" << key << " index=" << index << std::endl;
               site.info.print("site");
               for(auto pr :  qops_dict.at("r")('B')){
                  if(pr.first == 10007){
                      std::cout << "Br10007" << std::endl;
                      qops_dict.at("r")('B').at(10007).print("B");
                  }
               }
            }
            */
 
            for(int it=0; it<formula.size(); it++){
               /*
               Rlist<Tm> Rlst0;
               Rmu[k][it].gen_Rlist(superblock, site.info, Rlst0, blksize, cost, false);
               if(key == 'H'){
                  Rmu[k][it].gen_Rlist(superblock, site.info, Rlst0, blksize, cost, true);
               }
               std::copy(Rlst0.begin(), Rlst0.end(), std::back_inserter(Rlst));
               */
               Rmu[k][it].gen_Rlist(superblock, site.info, Rlst, blksize, cost, false);
               if(key == 'H'){
                  Rmu[k][it].gen_Rlist(superblock, site.info, Rlst, blksize, cost, true);
               }
               /*
               if(key == 'B' and index == 10007){
                  std::cout << "B10007" << " k=" << k << " key=" << key << " index=" << index
                            << " it=" << it 
                            << " size=" << Rlst0.size()
                            << std::endl;
               }
               */
            }
         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "T(Rmu/Rxlist/tot)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << ","
               << tools::get_duration(t1-t0) 
               << std::endl;
            tools::timing("preprocess_formulae_Rlist", t0, t1);
         }
      }

   template <typename Tm, typename QTm>
      void preprocess_formulae_Rlist2(const std::string superblock,
            const oper_dict<Tm>& qops,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            const renorm_tasks<Tm>& rtasks,
            const QTm& site,
            const rintermediates<Tm>& rinter,
            Rlist2<Tm>& Rlst2,
            size_t& blksize,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Hmu
         int rsize = rtasks.size();
         std::vector<std::vector<Rmu_ptr<Tm>>> Rmu(rsize);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            Rmu[k].resize(formula.size());
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].rinfo = const_cast<qinfo2<Tm>*>(&qops(key).at(index).info);
               Rmu[k][it].offrop = qops._offset.at(std::make_pair(key,index));
               Rmu[k][it].init(k, it, formula, qops_dict, rinter, oploc);
            }
         } // it
         auto ta = tools::get_time();

         // 2. from Hmu to expanded block forms
         blksize = 0;
         cost = 0.0;
         int nnzblk = qops.qbra.size(); // partitioned according to rows 
         Rlst2.resize(nnzblk);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);

            if(key != 'C') continue;

            for(int it=0; it<formula.size(); it++){

               std::cout << "it=" << it << std::endl;

               Rmu[k][it].gen_Rlist2(superblock, site.info, Rlst2, blksize, cost, false);
               if(key == 'H'){
                  Rmu[k][it].gen_Rlist2(superblock, site.info, Rlst2, blksize, cost, true);
               }
            }

         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "T(Hmu/Hxlist/tot)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << ","
               << tools::get_duration(t1-t0) 
               << std::endl;
            tools::timing("preprocess_formulae_Rlist2", t0, t1);
         }
      }

} // ctns

#endif
