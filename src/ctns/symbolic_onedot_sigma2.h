#ifndef SYMBOLIC_ONEDOT_SIGMA2_H
#define SYMBOLIC_ONEDOT_SIGMA2_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_dict.h"
#include "oper_timer.h"
#include "symbolic_oper.h"

namespace ctns{

template <typename Tm>
stensor2<Tm> symbolic_sum_oper(const oper_dict<Tm>& qops,
			       const symbolic_sum<Tm>& sop,
	          	       const char& label,
			       const bool& dagger,
			       Tm* workspace){
   int len = sop.size();
   // we assume the rest of terms have the same label/dagger
   auto wt0 = sop.sums[0].first;
   const auto& sop0 = sop.sums[0].second;
   int index0 = sop0.index;
   int nbar0  = sop0.nbar;
   // form opsum = wt0*op0 + wt1*op1 + ...
   const auto& op0 = qops(label).at(index0);
   if(dagger) wt0 = tools::conjugate(wt0);
   stensor2<Tm> optmp;
   optmp.init(op0.info,false);
   optmp.setup_data(workspace);
   optmp.clear();
   if(nbar0 == 0){
      linalg::xaxpy(op0.size(), wt0, op0.data(), optmp.data());
   }else{
      auto op0k = op0.K(nbar0);
      linalg::xaxpy(op0.size(), wt0, op0k.data(), optmp.data());
   }
   for(int k=1; k<len; k++){
      auto wtk = sop.sums[k].first;
      const auto& sopk = sop.sums[k].second;
      int indexk = sopk.index;
      int nbark  = sopk.nbar;
      const auto& opk = qops(label).at(indexk);
      if(dagger) wtk = tools::conjugate(wtk);
      if(nbark == 0){
         linalg::xaxpy(opk.size(), wtk, opk.data(), optmp.data());
      }else{
         auto opkk = opk.K(nbark);
         linalg::xaxpy(opk.size(), wtk, opkk.data(), optmp.data());
      }
   } // k
   return optmp;
}

template <typename Tm> 
void symbolic_onedot_HxTerm2(const oper_dictmap<Tm>& qops_dict,
			     const int it,
		             const symbolic_term<Tm>& HTerm,
			     const stensor3<Tm>& wf,
		             stensor3<Tm>& Hwf,
		             const size_t& opsize,
			     const size_t& wfsize,
			     const std::map<qsym,qinfo3<Tm>>& info_dict, 
			     Tm* workspace){
   const bool debug = false;
   if(debug) std::cout << "\niterm=" << it << " HTerm=" << HTerm << std::endl;
   // compute (HTerm+HTerm.H)*|wf>
   int isym = wf.info.sym.isym();
   qsym sym;
   stensor3<Tm> opxwf0, opxwf;
   // 1. opN*|wf>
   sym = wf.info.sym;
   opxwf0.init(wf.info,false);
   opxwf0.setup_data(wf.data());
   for(int idx=HTerm.size()-1; idx>=0; idx--){
      const auto& sop = HTerm.terms[idx];
      const auto& sop0 = sop.sums[0].second;
      const auto& index0 = sop0.index;
      const auto& parity = sop0.parity;
      const auto& label  = sop0.label;
      const auto& dagger = sop0.dagger;
      const auto& block = sop0.block;
      const auto& qops = qops_dict.at(block);
      auto optmp = symbolic_sum_oper(qops, sop, label, dagger, workspace);
      qsym sym_op = get_qsym_op(label, isym, index0); 
      sym = dagger? sym-sym_op : sym+sym_op;
      // opN*|wf>
      const auto& info = info_dict.at(sym);
      Tm* wptr = workspace+opsize+(idx%2)*wfsize;
      opxwf.init(info,false);
      opxwf.setup_data(wptr);
      contract_qt3_qt2_info(block,opxwf0,optmp,opxwf,dagger);
      if(parity) opxwf.cntr_signed(block); // impose antisymmetry here
      if(idx != 0){
	 opxwf0.info = opxwf.info;
         opxwf0.setup_data(wptr);	 
      }
   } // idx
   linalg::xaxpy(Hwf.size(), 1.0, opxwf.data(), Hwf.data()); 
   // 2. opH*|wf> 
   sym = wf.info.sym;
   opxwf0.init(wf.info,false);
   opxwf0.setup_data(wf.data());
   for(int idx=HTerm.size()-1; idx>=0; idx--){
      const auto& sop = HTerm.terms[idx];
      const auto& sop0 = sop.sums[0].second;
      const auto& index0 = sop0.index;
      const auto& parity = sop0.parity;
      const auto& label  = sop0.label;
      const auto& dagger = sop0.dagger;
      const auto& block = sop0.block;
      const auto& qops = qops_dict.at(block);
      auto optmp = symbolic_sum_oper(qops, sop, label, dagger, workspace);
      qsym sym_op = get_qsym_op(label, isym, index0); 
      sym = !dagger? sym-sym_op : sym+sym_op;
      // opH*|wf>
      const auto& info = info_dict.at(sym);
      Tm* wptr = workspace+opsize+(idx%2)*wfsize;
      opxwf.init(info,false);
      opxwf.setup_data(wptr);
      contract_qt3_qt2_info(block,opxwf0,optmp,opxwf,!dagger);
      if(parity) opxwf.cntr_signed(block); // impose antisymmetry here
      if(idx != 0){
	 opxwf0.info = opxwf.info;
         opxwf0.setup_data(wptr);	 
      }
   } // idx
   double fac = HTerm.Hsign(); // (opN)^H = sgn*opH
   linalg::xaxpy(Hwf.size(), fac, opxwf.data(), Hwf.data()); 
}

template <typename Tm> 
void symbolic_onedot_Hx2(Tm* y,
	                 const Tm* x,
	                 const symbolic_task<Tm>& H_formulae,
	           	 const oper_dictmap<Tm>& qops_dict,
			 const double& ecore,
	                 stensor3<Tm>& wf,
			 const int& size,
	                 const int& rank,
	                 const size_t& opsize,
			 const size_t& wfsize,
			 const size_t& tmpsize,
			 const std::map<qsym,qinfo3<Tm>>& info_dict, 
			 Tm* workspace){
   const bool debug = false;
   auto t0 = tools::get_time();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::symbolic_onedot_Hx2"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }
   //=======================
   // Parallel evaluation
   //=======================
   wf.from_array(x);
   // initialization
   std::vector<stensor3<Tm>> Hwfs(maxthreads);
   for(int i=0; i<maxthreads; i++){
      Hwfs[i].init(wf.info, false);
      Hwfs[i].setup_data(&workspace[i*tmpsize]);
      Hwfs[i].clear();
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
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_onedot_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		              opsize,wfsize,info_dict,
			      &workspace[omprank*tmpsize+wfsize]);
   } // it
   auto t2 = tools::get_time();
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   Hwfs[0].to_array(y);
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
