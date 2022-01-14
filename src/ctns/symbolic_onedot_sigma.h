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
		            const symbolic_term<Tm> HTerm,
			    const stensor3<Tm>& wf,
		            stensor3<Tm>& Hwf){

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
   for(int i=0; i<H_formulae.tasks.size(); i++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      const auto& HTerm = H_formulae.tasks[i];
      symbolic_onedot_HxTerm(lqops,cqops,rqops,HTerm,wf,Hwfs[omprank]);
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
