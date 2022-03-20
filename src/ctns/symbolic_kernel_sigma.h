#ifndef SYMBOLIC_SIGMA_H
#define SYMBOLIC_SIGMA_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_dict.h"
#include "oper_timer.h"
#include "symbolic_oper.h"

namespace ctns{
      
template <typename Tm, typename QTm> 
void symbolic_HxTerm(const oper_dictmap<Tm>& qops_dict,
		     const int it,
		     const symbolic_prod<Tm>& HTerm,
		     const QTm& wf,
		     QTm& Hwf,
		     const bool ifdagger){
   const bool debug = false;
   if(debug){ 
      std::cout << "iterm=" << it 
		<< " ifdagger=" << ifdagger
	        << " HTerm=" << HTerm 
		<< std::endl;
   }
   // compute (HTerm+HTerm.H)*|wf>
   QTm opxwf;
   for(int idx=HTerm.size()-1; idx>=0; idx--){
      const auto& sop = HTerm.terms[idx];
      int len = sop.size();
      auto wt0 = sop.sums[0].first;
      auto sop0 = sop.sums[0].second;
      // we assume the rest of terms have the same label/dagger/parity
      auto block  = sop0.block;
      char label  = sop0.label;
      bool dagger = sop0.dagger;
      bool parity = sop0.parity;
      int  index0 = sop0.index;
      int  nbar0  = sop0.nbar;
      if(debug){
         std::cout << " idx=" << idx
		   << " len=" << len
		   << " block=" << block
		   << " label=" << label
		   << " dagger=" << dagger
		   << " parity=" << parity
		   << " index0=" << index0 
		   << std::endl;
      }
      const auto& qops = qops_dict.at(block);
      // form opsum = wt0*op0 + wt1*op1 + ...
      const auto& op0 = qops(label).at(index0);
      if(dagger) wt0 = tools::conjugate(wt0);
      auto optmp = wt0*((nbar0==0)? op0 : op0.K(nbar0));      
      for(int k=1; k<len; k++){
         auto wtk = sop.sums[k].first;
	 auto sopk = sop.sums[k].second;
	 int indexk = sopk.index;
	 int nbark  = sopk.nbar;
	 const auto& opk = qops(label).at(indexk);
         if(dagger) wtk = tools::conjugate(wtk);
	 optmp += wtk*((nbark==0)? opk : opk.K(nbark));
      } // k
      const bool op_dagger = ifdagger^dagger; 
      if(op_dagger) linalg::xconj(optmp.size(), optmp.data());
      // (opN+opH)*|wf>
      if(idx == HTerm.size()-1){
         opxwf = contract_opxwf(block,wf,optmp,op_dagger);
      }else{
         opxwf = contract_opxwf(block,opxwf,optmp,op_dagger);
      }
      // impose antisymmetry here
      if(parity) opxwf.cntr_signed(block);
   } // idx
   double fac = ifdagger? HTerm.Hsign() : 1.0; // (opN)^H = sgn*opH
   linalg::xaxpy(Hwf.size(), fac, opxwf.data(), Hwf.data()); 
}

template <typename Tm, typename QTm> 
void symbolic_Hx(Tm* y,
	         const Tm* x,
	         const symbolic_task<Tm>& H_formulae,
	         const oper_dictmap<Tm>& qops_dict,
		 const double& ecore,
	         QTm& wf,
	         const int size,
	         const int rank){
   const bool debug = false;
   auto t0 = tools::get_time();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){ 
      std::cout << "ctns::symbolic_Hx"
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
      Hwfs[i].init(wf.info);
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
      symbolic_HxTerm(qops_dict,it,HTerm,wf,Hwfs[omprank],false);
      symbolic_HxTerm(qops_dict,it,HTerm,wf,Hwfs[omprank],true);
   }
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
