#ifndef SYMBOLIC_SIGMA2_H
#define SYMBOLIC_SIGMA2_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_sum_kernel.h"

namespace ctns{

template <typename Tm, typename QTm, typename QInfo> 
void symbolic_HxTerm2(const oper_dictmap<Tm>& qops_dict,
		      const int it,
		      const symbolic_prod<Tm>& HTerm,
		      const QTm& wf,
		      QTm& Hwf,
		      const std::map<qsym,QInfo>& info_dict, 
		      const size_t& opsize,
		      const size_t& wfsize,
		      Tm* workspace,
		      const bool ifdagger){
   const bool debug = false;
   if(debug){ 
      std::cout << "iterm=" << it 
		<< " ifdagger=" << ifdagger
	        << " HTerm=" << HTerm 
		<< std::endl;
   }
   // compute (HTerm+HTerm.H)*|wf>
   qsym sym;
   QInfo *opxwf0_info, *opxwf_info;
   Tm *opxwf0_data, *opxwf_data;
   // op(dagger)*|wf>
   sym = wf.info.sym;
   opxwf0_info = const_cast<QInfo*>(&wf.info);
   opxwf0_data = wf.data();
   for(int idx=HTerm.size()-1; idx>=0; idx--){
      const auto& sop = HTerm.terms[idx];
      const auto& sop0 = sop.sums[0].second;
      const auto& index0 = sop0.index;
      const auto& parity = sop0.parity;
      const auto& label  = sop0.label;
      const auto& dagger = sop0.dagger;
      const auto& block = sop0.block;
      const auto& qops = qops_dict.at(block);
      // form operator
      auto optmp = symbolic_sum_oper(qops, sop, label, dagger, workspace);
      const bool op_dagger = ifdagger^dagger; // (w op^d1)^d2 = (w^d1 op)^d1d2 
      if(op_dagger) linalg::xconj(optmp.size(), optmp.data());
      // op(dagger)*|wf>
      sym += op_dagger? -optmp.info.sym : optmp.info.sym;
      opxwf_info = const_cast<QInfo*>(&info_dict.at(sym));
      opxwf_data = workspace+opsize+(idx%2)*wfsize;
      contract_opxwf_info(block, *opxwf0_info, opxwf0_data,
			  optmp.info, optmp.data(),
             	          *opxwf_info, opxwf_data, op_dagger);
      // impose antisymmetry here
      if(parity) cntr_signed(block, *opxwf_info, opxwf_data);
      opxwf0_info = opxwf_info;
      opxwf0_data = opxwf_data;
   }
   double fac = ifdagger? HTerm.Hsign() : 1.0;
   linalg::xaxpy(Hwf.size(), fac, opxwf_data, Hwf.data());
}
		      
template <typename Tm, typename QTm, typename QInfo> 
void symbolic_Hx2(Tm* y,
	          const Tm* x,
	          const symbolic_task<Tm>& H_formulae,
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
      std::cout << "ctns::symbolic_Hx2"
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
      // opN*|wf>
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		       info_dict,opsize,wfsize,
		       &workspace[omprank*tmpsize+wfsize],
		       false);
      // opH*|wf>
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		       info_dict,opsize,wfsize,
		       &workspace[omprank*tmpsize+wfsize],
		       true);
   } // it
   auto t2 = tools::get_time();
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   Hwfs[0].to_array(y);

/*
   memset(y, 0, wf.size()*sizeof(Tm));
   auto t1 = tools::get_time();
   #pragma omp parallel
   {
      int omprank = omp_get_thread_num();

   // initialization
   QTm Hwfs(wf.info,  false);
   Hwfs.setup_data(&workspace[omprank*tmpsize]);
   Hwfs.clear();

   #pragma omp for schedule(dynamic,1)
   for(int it=0; it<H_formulae.size(); it++){
      int rk = omprank;
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs,
		       info_dict,opsize,wfsize,
		       &workspace[rk*tmpsize+wfsize],
		       false);
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs,
		       info_dict,opsize,wfsize,
		       &workspace[rk*tmpsize+wfsize],
		       true);
   } // it

   #pragma omp critical
   {
      linalg::xaxpy(Hwfs.size(), 1.0, Hwfs.data(), y);
   }

   }
   auto t2 = tools::get_time();
*/
/*
   memset(y, 0, wf.size()*sizeof(Tm));
   auto t1 = tools::get_time();
   #pragma omp parallel
   {
      int omprank = omp_get_thread_num();
   
   Tm* worklocal = new Tm[tmpsize];
   //Tm* worklocal = workspace+omprank*tmpsize;
   // initialization
   QTm Hwfs(wf.info,  false);
   Hwfs.setup_data(worklocal);
   Hwfs.clear();
   //#pragma omp for schedule(static,1) nowait
   #pragma omp for schedule(dynamic,100)
   for(int it=0; it<H_formulae.size(); it++){
      int rk = omprank;
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs,
		       info_dict,opsize,wfsize,
		       &worklocal[wfsize],
		       false);
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs,
		       info_dict,opsize,wfsize,
		       &worklocal[wfsize],
		       true);
   }

   #pragma omp critical
   {
      linalg::xaxpy(Hwfs.size(), 1.0, Hwfs.data(), y);
   }

   delete[] worklocal;

   }
   auto t2 = tools::get_time();
*/

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
