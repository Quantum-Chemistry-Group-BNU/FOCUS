#ifndef SYMBOLIC_KERNEL_RENORM2_H
#define SYMBOLIC_KERNEL_RENORM2_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_formulae_renorm.h"
#include "symbolic_kernel_sum.h"

namespace ctns{

template <typename Tm> 
void symbolic_renorm_single2(const std::string& block1,
			     const std::string& block2,
			     const oper_dictmap<Tm>& qops_dict,
			     const char key,
	  	             const symbolic_task<Tm>& formulae,
			     const stensor3<Tm>& wf,
		             qinfo3<Tm>& Hwf_info,
			     Tm* Hwf_data,
			     const size_t& opsize,
			     const size_t& wfsize,
			     const std::map<qsym,qinfo3<Tm>>& info_dict,
			     Tm* workspace){
   const bool debug = false;
   if(debug) formulae.display("formulae");
   if(formulae.size()==0){
      memset(Hwf_data, 0, Hwf_info._size*sizeof(Tm));
      return;
   }
   qsym sym;
   qinfo3<Tm> *opxwf0_info, *opxwf_info;
   Tm *opxwf0_data, *opxwf_data;
   // opN*|wf>
   for(int it=0; it<formulae.size(); it++){
      const auto& HTerm = formulae.tasks[it];
      // term[it]*wf
      sym = wf.info.sym;
      opxwf0_info = const_cast<qinfo3<Tm>*>(&wf.info);
      opxwf0_data = wf.data();
      for(int idx=HTerm.size()-1; idx>=0; idx--){
         const auto& sop = HTerm.terms[idx];
         const auto& sop0 = sop.sums[0].second;
         const auto& parity = sop0.parity;
         const auto& dagger = sop0.dagger;
         const auto& block = sop0.block;
         // form operator
         auto optmp = symbolic_sum_oper(qops_dict, sop, workspace);
         if(dagger) linalg::xconj(optmp.size(), optmp.data());
	 // opN*|wf> 
         sym += dagger? -optmp.info.sym : optmp.info.sym;
         opxwf_info = const_cast<qinfo3<Tm>*>(&info_dict.at(sym));
	 opxwf_data = workspace+opsize+(idx%2)*wfsize; 
	 contract_opxwf_info(block, *opxwf0_info, opxwf0_data,
			     optmp.info, optmp.data(),
			     *opxwf_info, opxwf_data, dagger);
	 // impose antisymmetry here
         if(block == block2 and parity){ 
            if(block1 == "l"){ // lc or lr
	       row_signed(*opxwf_info, opxwf_data);
	    }else if(block1 == "c"){
	       mid_signed(*opxwf_info, opxwf_data);
	    }
	 }
         opxwf0_info = opxwf_info;
         opxwf0_data = opxwf_data;
      } // idx
      if(it == 0){
	 linalg::xcopy(Hwf_info._size, opxwf_data, Hwf_data);
      }else{
	 linalg::xaxpy(Hwf_info._size, 1.0, opxwf_data, Hwf_data);
      }
   } // it
}

template <typename Tm>
size_t symbolic_kernel_renorm2(const std::string superblock,
			     const renorm_tasks<Tm>& rtasks,
		             const stensor3<Tm>& site,
		             const oper_dict<Tm>& qops1,
		             const oper_dict<Tm>& qops2,
		             oper_dict<Tm>& qops,
			     const int verbose){
   if(qops.mpirank==0 and verbose>1){
      std::cout << "ctns::symbolic_kernel_renorm2"
	        << " size(formulae)=" << rtasks.size() 
		<< std::endl;
   }
   const std::string block1 = superblock.substr(0,1);
   const std::string block2 = superblock.substr(1,2);
   const oper_dictmap<Tm> qops_dict = {{block1,qops1},
	   		 	       {block2,qops2}};
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   std::map<qsym,qinfo3<Tm>> info_dict;
   size_t opsize = preprocess_opsize(qops_dict);
   size_t wfsize = preprocess_wfsize(site.info, info_dict);
   size_t tmpsize = opsize + 3*wfsize;
   size_t worktot = maxthreads*tmpsize;
   if(qops.mpirank == 0 and verbose>0){
      std::cout << "preprocess for renorm:"
                << " opsize=" << opsize
                << " wfsize=" << wfsize
                << " worktot=" << worktot
                << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                << ":" << tools::sizeGB<Tm>(worktot) << "GB"
                << std::endl; 
   }
   Tm* workspace = new Tm[worktot];
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<rtasks.size(); i++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      const auto& task = rtasks.op_tasks[i];
      auto key = std::get<0>(task);
      auto index = std::get<1>(task);
      auto formula = std::get<2>(task);
      auto size = formula.size();
      if(verbose>2){
         std::cout << "rank=" << qops.mpirank 
		   << " idx=" << i 
		   << " op=" << key
		   << " index=" << index
		   << " size=" << size
		   << std::endl;
	 formula.display("formula", 1);
      }
      if(size == 0) continue;
      // op|ket>
      auto sym_op = qops.get_qsym_op(key, index);
      auto sym = sym_op + site.info.sym;
      qinfo3<Tm> *opxwf_info = const_cast<qinfo3<Tm>*>(&info_dict.at(sym));
      Tm* opxwf_data = workspace + omprank*tmpsize;
      symbolic_renorm_single2(block1,block2,qops_dict,
		              key,formula,site,*opxwf_info,opxwf_data,
			      opsize,wfsize,info_dict,
			      &workspace[omprank*tmpsize+wfsize]);
      // <bra|op|ket>
      auto& op = qops(key)[index];
      contract_qt3_qt3_info(superblock, site.info, site.data(),
		            *opxwf_info, opxwf_data,
			    op.info, op.data());
      if(key == 'H') op += op.H();
      if(key == 'H' && qops.ifkr) op += op.K();
   } // i
   delete[] workspace;
   return worktot;
}

} // ctns

#endif
