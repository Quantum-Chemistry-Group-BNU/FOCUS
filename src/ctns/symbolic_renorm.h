#ifndef SYMBOLIC_RENORM_H
#define SYMBOLIC_RENORM_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ctns{

template <typename Tm>
void symbolic_renorm_kernel(const std::string superblock,
		            const stensor3<Tm>& site,
		            const integral::two_body<Tm>& int2e,
		            const oper_dict<Tm>& qops1,
		            const oper_dict<Tm>& qops2,
		            oper_dict<Tm>& qops,
			    const int rank,
			    const bool debug){
   std::cout << "XXX" << std::endl;
   exit(1);
/*
   oper_timer.clear();
   auto Hx_funs = oper_renorm_functors(superblock, site, int2e, qops1, qops2, qops);
   if(debug){
      std::cout << "rank=" << rank << " size[Hx_funs]=" << Hx_funs.size() << std::endl;
      for(int i=0; i<Hx_funs.size(); i++){
         std::cout << "rank=" << rank << " i=" << i << Hx_funs[i] << std::endl;
      }
   }
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<Hx_funs.size(); i++){
      char key = Hx_funs[i].label[0];
      int index = Hx_funs[i].index; 
      if(debug) std::cout << "cal: rank=" << rank 
	                  << " i=" << i 
	                  << " key=" << key 
			  << " index=" << index 
			  << std::endl;
      auto opxwf = Hx_funs[i]();
      auto op = contract_qt3_qt3(superblock, site, opxwf);
      linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
   }
*/
}

} // ctns

#endif
