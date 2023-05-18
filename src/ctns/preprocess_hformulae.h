#ifndef PREPROCESS_HFORMULAE_H
#define PREPROCESS_HFORMULAE_H

#include "preprocess_hinter.h"
#include "preprocess_hmu.h"

namespace ctns{

   template <typename Tm, typename QTm>
      void preprocess_formulae_Hxlist(const bool ifDirect,
            const int alg_coper,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const symbolic_task<Tm>& H_formulae,
            const QTm& wf,
            const hintermediates<Tm>& hinter,
            Hxlist<Tm>& Hxlst,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         t0 = tools::get_time();
         Hxlist2<Tm> Hxlst2;
         preprocess_formulae_Hxlist2(ifDirect, alg_coper, qops_dict, oploc, opaddr, H_formulae, wf, hinter, 
               Hxlst2, blksize, blksize0, cost, debug);
         size_t size = 0;
         for(int i=0; i<Hxlst2.size(); i++){
            size += Hxlst2[i].size();
         }
         Hxlst.reserve(size);
         for(int i=0; i<Hxlst2.size(); i++){
            std::move(Hxlst2[i].begin(), Hxlst2[i].end(), std::back_inserter(Hxlst));
         }

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR preprocess_formulae_Hxlist : "
               << tools::get_duration(t1-t0) << " S"
               << " size(H_formulae)=" << H_formulae.size()
               << " size(Hxlst)=" << Hxlst.size() << " -----"
               << std::endl;
         }
      }

   template <typename Tm, typename QTm>
      void preprocess_formulae_Hxlist2(const bool ifDirect,
            const int alg_coper,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
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
         //for(int it=0; it<hsize; it++){
         for(int it=1; it<2; it++){
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
            Hmu_vec[it].gen_Hxlist2(alg_coper, opaddr, wf.info, Hxlst2, blksize, blksize0, cost, false);
            //Hmu_vec[it].gen_Hxlist2(alg_coper, opaddr, wf.info, Hxlst2, blksize, blksize0, cost, true);
         }
         auto tb = tools::get_time();

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR preprocess_formulae_Hxlist2 : "
               << tools::get_duration(t1-t0) << " S"
               << " size(H_formulae)=" << hsize  
               << " size(Hxlst2)=" << nnzblk 
               << " T(Hmu/Hxlist)="
               << tools::get_duration(ta-t0) << ","
               << tools::get_duration(tb-ta) << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
