#ifndef SYMBOLIC_KERNEL_SIGMA3_H
#define SYMBOLIC_KERNEL_SIGMA3_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_kernel_sum.h"

namespace ctns{

   //
   // symbolic formulae (factorized) + preallocation of workspace 
   //

   template <typename Tm, typename QTm, typename QInfo> 
      void symbolic_HxTerm3(const oper_dictmap<Tm>& qops_dict,
            const int it,
            const bipart_oper<Tm>& bipart_op,
            const QTm& wf,
            QTm& Hwf,
            const std::map<qsym,QInfo>& info_dict, 
            const size_t& opsize,
            const size_t& wfsize,
            Tm* workspace,
            const bool ifdagger){
         const bool debug = false;
         if(debug){
            std::cout << "it=" << it;
            bipart_op.display();
         }
         const auto& lformulae = bipart_op.lop;
         const auto& rformulae = bipart_op.rop;
         if(lformulae.size() == 0 and rformulae.size() == 0) return;
         // temporary
         QInfo *opxwf0_info, *opxwf_info;
         Tm *opxwf0_data, *opxwf_data;
         // rop*|wf>
         QInfo *wf0_info = const_cast<QInfo*>(&wf.info);
         Tm *wf0_data = wf.data();
         for(int it=0; it<rformulae.size(); it++){
            const auto& HTerm = rformulae.tasks[it];
            // term[it]*wf
            opxwf0_info = wf0_info;
            opxwf0_data = wf0_data;
            qsym sym = wf0_info->sym;
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               const auto& sop0 = sop.sums[0].second;
               const auto& parity = sop0.parity;
               const auto& dagger = sop0.dagger;
               const auto& block = sop0.block;
               // form operator
               auto optmp = symbolic_sum_oper(qops_dict, sop, workspace);
               const bool op_dagger = ifdagger^dagger; // (w op^d1)^d2 = (w^d1 op)^d1d2 
               if(op_dagger) linalg::xconj(optmp.size(), optmp.data());
               // op(dagger)*|wf>
               sym += op_dagger? -optmp.info.sym : optmp.info.sym;
               opxwf_info = const_cast<QInfo*>(&info_dict.at(sym));
               opxwf_data = workspace+opsize+(1+idx%2)*wfsize;
               contract_opxwf_info(block, *opxwf0_info, opxwf0_data,
                     optmp.info, optmp.data(),
                     *opxwf_info, opxwf_data, op_dagger);
               // impose antisymmetry here
               if(parity) cntr_signed(block, *opxwf_info, opxwf_data);
               opxwf0_info = opxwf_info;
               opxwf0_data = opxwf_data;
            } // idx
            double fac = ifdagger? HTerm.Hsign() : 1.0;
            if(lformulae.size() == 0){ // case: Hl 
               assert(opxwf_info->_size == Hwf.size());
               linalg::xaxpy(opxwf_info->_size, fac, opxwf_data, Hwf.data());
            }else{
               if(it == 0){
                  linalg::xscal(opxwf_info->_size, fac, opxwf_data);
                  linalg::xcopy(opxwf_info->_size, opxwf_data, &workspace[opsize]);
               }else{
                  linalg::xaxpy(opxwf_info->_size, fac, opxwf_data, &workspace[opsize]);
               }
            }
         } // it
           // lop*|wf>
         double fac0 = ifdagger? bipart_op.Hsign() : 1.0;
         if(rformulae.size() != 0){
            wf0_info = opxwf_info;
            wf0_data = workspace+opsize;
         }
         for(int it=0; it<lformulae.size(); it++){
            const auto& HTerm = lformulae.tasks[it];
            // term[it]*wf
            opxwf0_info = wf0_info;
            opxwf0_data = wf0_data;
            qsym sym = wf0_info->sym;
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               const auto& sop0 = sop.sums[0].second;
               const auto& parity = sop0.parity;
               const auto& dagger = sop0.dagger;
               const auto& block = sop0.block;
               // form operator
               auto optmp = symbolic_sum_oper(qops_dict, sop, workspace);
               const bool op_dagger = ifdagger^dagger; // (w op^d1)^d2 = (w^d1 op)^d1d2 
               if(op_dagger) linalg::xconj(optmp.size(), optmp.data());
               // op(dagger)*|wf>
               sym += op_dagger? -optmp.info.sym : optmp.info.sym;
               opxwf_info = const_cast<QInfo*>(&info_dict.at(sym));
               opxwf_data = workspace+opsize+(1+idx%2)*wfsize;
               contract_opxwf_info(block, *opxwf0_info, opxwf0_data,
                     optmp.info, optmp.data(),
                     *opxwf_info, opxwf_data, op_dagger);
               // impose antisymmetry here
               if(parity) cntr_signed(block, *opxwf_info, opxwf_data);
               opxwf0_info = opxwf_info;
               opxwf0_data = opxwf_data;
            } // idx
            double fac = fac0*(ifdagger? HTerm.Hsign() : 1.0);
            assert(opxwf_info->_size == Hwf.size());
            linalg::xaxpy(opxwf_info->_size, fac, opxwf_data, Hwf.data());
         } // it
      }

   template <bool ifab, typename Tm, typename QTm, typename QInfo>
      void symbolic_Hx3(Tm* y,
            const Tm* x,
            const bipart_task<Tm>& H_formulae,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const double& ecore,
            QTm& wf,
            const int& size,
            const int& rank,
            const std::map<qsym,QInfo>& info_dict, 
            const size_t& opsize,
            const size_t& wfsize,
            const size_t& tmpsize,
            Tm* workspace){
         std::cout << "error: no implementation of symbolic_Hx3 for su2! ifab=" << ifab << std::endl;
         exit(1);
      } 
   template <bool ifab, typename Tm, typename QTm, typename QInfo, std::enable_if_t<ifab,int> = 0>
      void symbolic_Hx3(Tm* y,
            const Tm* x,
            const bipart_task<Tm>& H_formulae,
            const oper_dictmap<Tm>& qops_dict,
            const double& ecore,
            QTm& wf,
            const int& size,
            const int& rank,
            const std::map<qsym,QInfo>& info_dict, 
            const size_t& opsize,
            const size_t& wfsize,
            const size_t& tmpsize,
            Tm* workspace){
         const bool debug = false;
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug){
            std::cout << "ctns::symbolic_Hx3"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         //=======================
         // Parallel evaluation
         //=======================
         wf.from_array(x);

         // initialization
         std::vector<QTm> Hwfs(maxthreads);
         for(int i=0; i<maxthreads; i++){
            Hwfs[i].init(wf.info, false);
            Hwfs[i].setup_data(&workspace[i*tmpsize]);
            Hwfs[i].set_zero();
         }
         auto t1 = tools::get_time();

         // compute
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int it=0; it<H_formulae.size(); it++){
#ifdef _OPENMP
            int omprank = omp_get_thread_num();
#else
            int omprank = 0;
#endif
            const auto& HTerm = H_formulae[it];
            // opN*|wf>
            symbolic_HxTerm3(qops_dict,it,HTerm,wf,Hwfs[omprank],
                  info_dict,opsize,wfsize,
                  &workspace[omprank*tmpsize+wfsize],
                  false);
            // opH*|wf>
            symbolic_HxTerm3(qops_dict,it,HTerm,wf,Hwfs[omprank],
                  info_dict,opsize,wfsize,
                  &workspace[omprank*tmpsize+wfsize],
                  true);
         } // it
           // reduction & save
         for(int i=1; i<maxthreads; i++){
            Hwfs[0] += Hwfs[i];
         }
         Hwfs[0].to_array(y);
         auto t2 = tools::get_time();

         // add const term
         if(rank == 0){
            const Tm scale = qops_dict.at("l").ifkr? 0.5 : 1.0;
            linalg::xaxpy(wf.size(), scale*ecore, x, y);
         }
         auto t3 = tools::get_time();
         oper_timer.tHxInit += tools::get_duration(t1-t0);
         oper_timer.tHxCalc += tools::get_duration(t2-t1);
         oper_timer.tHxFinl += tools::get_duration(t3-t2);
      }

} // ctns

#endif
