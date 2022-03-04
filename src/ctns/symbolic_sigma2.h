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
		      const symbolic_term<Tm>& HTerm,
		      const QTm& wf,
		      QTm& Hwf,
		      const std::map<qsym,QInfo>& info_dict, 
		      const size_t& opsize,
		      const size_t& wfsize,
		      Tm* workspace){
/*
   const bool debug = true;
   if(debug) std::cout << "\niterm=" << it << " HTerm=" << HTerm << std::endl;
   auto t0 = tools::get_time();
*/
   // compute (HTerm+HTerm.H)*|wf>
   int isym = wf.info.sym.isym();
   qsym sym;
   QTm opxwf0, opxwf;
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
      contract_opxwf_info(block,opxwf0,optmp,opxwf,dagger);
      // impose antisymmetry here
      if(parity) opxwf.cntr_signed(block); 
      if(idx != 0){
	 opxwf0.info = opxwf.info;
         opxwf0.setup_data(wptr);	 
      }
   } // idx

//   linalg::xaxpy(Hwf.size(), 1.0, opxwf.data(), Hwf.data()); 

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
      contract_opxwf_info(block,opxwf0,optmp,opxwf,!dagger);
      // impose antisymmetry here
      if(parity) opxwf.cntr_signed(block); 
      if(idx != 0){
	 opxwf0.info = opxwf.info;
         opxwf0.setup_data(wptr);	 
      }
   } // idx
   double fac = HTerm.Hsign(); // (opN)^H = sgn*opH

//   linalg::xaxpy(Hwf.size(), fac, opxwf.data(), Hwf.data());
 
/*   
   auto t1 = tools::get_time();
   std::cout << "dt=" << std::scientific << std::setprecision(4)
	     << tools::get_duration(t1-t0) << std::endl;
*/
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
	      	<< " QTm=" << get_name<QTm>() 
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
   #pragma omp parallel for schedule(dynamic,1)
#endif
   for(int it=0; it<H_formulae.size(); it++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		       info_dict,opsize,wfsize,
		       &workspace[omprank*tmpsize+wfsize]);
   } // it
   auto t2 = tools::get_time();
/*
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   Hwfs[0].to_array(y);
*/

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
      //std::cout << "it=" << it << " rk=" << rk << std::endl;
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs,
		       info_dict,opsize,wfsize,
		       &workspace[rk*tmpsize+wfsize]);
   } // it
   #pragma omp critical
   {
   linalg::xaxpy(Hwfs.size(), 1.0, Hwfs.data(), y);
   }
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
