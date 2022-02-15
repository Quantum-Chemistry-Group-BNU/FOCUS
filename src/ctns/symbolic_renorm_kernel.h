#ifndef SYMBOLIC_RENORM_KERNEL_H
#define SYMBOLIC_RENORM_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_renorm_formulae.h"

namespace ctns{

template <typename Tm> 
stensor3<Tm> symbolic_renorm_single(const std::string& block1,
				    const std::string& block2,
				    const oper_dictmap<Tm>& qops_dict,
			            const char key,
	  	                    const symbolic_task<Tm>& formulae,
			            const stensor3<Tm>& wf){
   const bool debug = false;
   stensor3<Tm> opxwf;
   for(int it=0; it<formulae.size(); it++){
      const auto& HTerm = formulae.tasks[it];
      stensor3<Tm> opNxwf, opHxwf;
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
         // (opN+opH)*|wf>
         if(idx == HTerm.size()-1){
            opNxwf = contract_qt3_qt2(block,wf,optmp,dagger);
	    if(key == 'H') opHxwf = contract_qt3_qt2(block,wf,optmp,!dagger);
         }else{
            opNxwf = contract_qt3_qt2(block,opNxwf,optmp,dagger);
	    if(key == 'H') opHxwf = contract_qt3_qt2(block,opHxwf,optmp,!dagger);
         }
         // impose antisymmetry here
         if(block == block2 and parity){ 
            if(block1 == "l"){ // lc or lr
	       opNxwf.row_signed();
	       if(key == 'H') opHxwf.row_signed();
	    }else if(block1 == "c"){
	       opNxwf.mid_signed();
	       if(key == 'H') opHxwf.mid_signed();
	    }
         }
      } // idx
      if(it == 0) opxwf.init(opNxwf.info);
      linalg::xaxpy(opxwf.size(), 1.0, opNxwf.data(), opxwf.data());
      if(key == 'H') linalg::xaxpy(opxwf.size(), 1.0, opHxwf.data(), opxwf.data());
   } // it
   return opxwf;
}

template <typename Tm>
void symbolic_renorm_kernel(const std::string superblock,
		            const stensor3<Tm>& site,
		            const integral::two_body<Tm>& int2e,
		            const oper_dict<Tm>& qops1,
		            const oper_dict<Tm>& qops2,
		            oper_dict<Tm>& qops,
			    const int rank,
			    const bool debug){
   // generate formulae for renormalization first
   auto tasks = symbolic_renorm_formulae(superblock, int2e, qops1, qops2, qops);
   if(debug) std::cout << "rank=" << rank << " size[tasks]=" << tasks.size() << std::endl;
   const std::string block1 = superblock.substr(0,1);
   const std::string block2 = superblock.substr(1,2);
   const oper_dictmap<Tm> qops_dict = {{block1,qops1},
	   		 	       {block2,qops2}};
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<tasks.size(); i++){
      const auto& task = tasks[i];
      auto key = std::get<0>(task);
      auto index = std::get<1>(task);
      auto formula = std::get<2>(task);
      if(debug){
         std::cout << "rank=" << rank 
		   << " i=" << i 
		   << " key=" << key
		   << " index=" << index
		   << std::endl;
	 formula.display("formula", 1);
      }
      auto opxwf = symbolic_renorm_single(block1,block2,qops_dict,key,formula,site);
      auto op = contract_qt3_qt3(superblock, site, opxwf); 
      linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
   } // i
}

} // ctns

#endif
