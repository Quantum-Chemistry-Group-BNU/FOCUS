#ifndef SYMBOLIC_ONEDOT_SIGMA_H
#define SYMBOLIC_ONEDOT_SIGMA_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_dict.h"
#include "oper_timer.h"
#include "symbolic_oper.h"

namespace ctns{
      
template <typename Tm> 
void symbolic_onedot_HxTerm(const oper_dict<Tm>& lqops,
	          	    const oper_dict<Tm>& rqops,
	          	    const oper_dict<Tm>& cqops,
			    const int it,
		            const symbolic_term<Tm> HTerm,
			    const stensor3<Tm>& wf,
		            stensor3<Tm>& Hwf){
   const bool debug = false;
   if(debug){ 
      std::cout << "\niterm=" << it << " HTerm=" << HTerm << std::endl;
   }
   const std::map<std::string,const oper_dict<Tm>&> qops_dict = {{"l",lqops},
	   		 	                                 {"r",rqops},
	   			 	                 	 {"c",cqops}};
   // compute (HTerm+HTerm.H)*|wf>
   stensor3<Tm> opNxwf = wf, opHxwf = wf;
   for(int idx=HTerm.size()-1; i>=0; i--){
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
      // opsum = wt0*op0 + wt1*op1 + ...
      const auto& op0 = qops(label).at(index0);
      stensor2<Tm> optmp = wt0*(dagger? op0.H() : op0); 
      for(int k=1; k<len; k++){
         auto wtk = sop.sums[k].first;
	 auto sopk = sop.sums[k].second;
	 int indexk = sopk.index;
	 const auto& opk = qops(label).at(indexk);
         optmp += wtk*(dagger? opk.H() : opk);
      }
      // impose antisymmetry here
      if(parity){ 
         opNxwf.cntr_signed(block);
         opHxwf.cntr_signed(block);
      }
      // (opN+opH)*|wf> 
      opNxwf = contract_qt3_qt2(block,opNxwf,optmp);
      opHxwf = contract_qt3_qt2(block,opHxwf,optmp,true);
   }
   int N = Hwf.size();
   double fac = HTerm.Hsign(); // (opN)^H = sgn*opH
   linalg::xaxpy(N, 1.0, opNxwf.data(), Hwf.data()); 
   linalg::xaxpy(N, fac, opHxwf.data(), Hwf.data());  
}

template <typename Tm> 
void symbolic_onedot_Hx(Tm* y,
	                const Tm* x,
	                const symbolic_task<Tm>& H_formulae,
	          	const oper_dict<Tm>& lqops,
	          	const oper_dict<Tm>& rqops,
	          	const oper_dict<Tm>& cqops,
			const double& ecore,
	                stensor3<Tm>& wf,
	                const int size,
	                const int rank){
   const bool debug = true;
   auto t0 = tools::get_time();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){ 
      std::cout << "ctns::symbolic_onedot_Hx size=" << size 
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
      symbolic_onedot_HxTerm(lqops,rqops,cqops,it,HTerm,wf,Hwfs[omprank]);
   }
   auto t2 = tools::get_time();
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   Hwfs[0].to_array(y);
   // add const term
   if(rank == 0){
      const Tm scale = lqops.ifkr? 0.5 : 1.0;
      linalg::xaxpy(wf.size(), scale*ecore, x, y);
   }
   auto t3 = tools::get_time();
   oper_timer.tHxInit += tools::get_duration(t1-t0);
   oper_timer.tHxCalc += tools::get_duration(t2-t1);
   oper_timer.tHxFinl += tools::get_duration(t3-t2);
}

} // ctns

#endif
