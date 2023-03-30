#ifndef PREPROCESS_HFORMULAE_H
#define PREPROCESS_HFORMULAE_H

#include "preprocess_hinter.h"
#include "preprocess_hmu.h"

namespace ctns{

   template <typename Tm, typename QTm>
      void preprocess_formulae_Hxlist(const bool ifDirect,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            const symbolic_task<Tm>& H_formulae,
            const QTm& wf,
            const hintermediates<Tm>& hinter,
            Hxlist<Tm>& Hxlst,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Hmu
         int hsize = H_formulae.size();
         std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
         for(int it=0; it<hsize; it++){
            Hmu_vec[it].init(ifDirect, it, H_formulae, qops_dict, hinter, oploc);
         } // it
         auto ta = tools::get_time();

         // 2. from Hmu to Hxlst in expanded block forms
         blksize = 0;
         blksize0 = 0;
         cost = 0.0;
         for(int it=0; it<hsize; it++){
            Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, blksize0, cost, false);
            Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, blksize0, cost, true);
         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "T(Hmu/Hxlist/tot)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << ","
               << tools::get_duration(t1-t0) 
               << std::endl;
            tools::timing("preprocess_formulae_Hxlist", t0, t1);
         }
      }

   template <typename Tm, typename QTm>
      void preprocess_formulae_Hxlist2(const bool ifDirect,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            const symbolic_task<Tm>& H_formulae,
            const QTm& wf,
            const hintermediates<Tm>& hinter,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Hmu
         int hsize = H_formulae.size();
         std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
         for(int it=0; it<hsize; it++){
            Hmu_vec[it].init(ifDirect, it, H_formulae, qops_dict, hinter, oploc); 
         } // it
         auto ta = tools::get_time();

         // 2. from Hmu to expanded block forms
         blksize = 0;
         blksize0 = 0;
         cost = 0.0;
         int nnzblk = wf.info._nnzaddr.size();
         Hxlst2.resize(nnzblk);
         for(int it=0; it<hsize; it++){
            Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, blksize0, cost, false);
            Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, blksize0, cost, true);
         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "T(Hmu/Hxlist/tot)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << ","
               << tools::get_duration(t1-t0) 
               << std::endl;
            tools::timing("preprocess_formulae_Hxlist2", t0, t1);
         }
      }

} // ctns

#endif
