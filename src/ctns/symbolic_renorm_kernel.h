#ifndef SYMBOLIC_RENORM_KERNEL_H
#define SYMBOLIC_RENORM_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_renorm_formulae.h"

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
   const std::string block1 = superblock.substr(0,1);
   const std::string block2 = superblock.substr(1,2);
   const oper_dictmap<Tm> qops_dict = {{block1,qops1},
	   		 	       {block2,qops2}};

   // generate formulae for renormalization first
   auto tasks = symbolic_renorm_formulae(superblock, int2e, qops1, qops2, qops);
   if(!debug){
      std::cout << "rank=" << rank << " size[tasks]=" << tasks.size() << std::endl;
      for(int i=0; i<tasks.size(); i++){
         const auto& task = tasks[i];
	 auto key = std::get<0>(task);
	 auto index = std::get<1>(task);
	 auto formula = std::get<2>(task);
         std::cout << "rank=" << rank 
		   << " i=" << i 
		   << " key=" << key
		   << " index=" << index
		   << std::endl;
	 formula.display("formula", 1);
      }
   }
   exit(1);
/*
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
