#ifndef SYMBOLIC_RENORM_KERNEL2_H
#define SYMBOLIC_RENORM_KERNEL2_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_renorm_formulae.h"
#include "symbolic_sum_kernel.h"

namespace ctns{

template <typename Tm> 
void symbolic_renorm_single2(const std::string& block1,
			     const std::string& block2,
			     const oper_dictmap<Tm>& qops_dict,
			     const char key,
	  	             const symbolic_task<Tm>& formulae,
			     const stensor3<Tm>& wf,
		             stensor3<Tm>& Hwf,
			     const size_t& opsize,
			     const size_t& wfsize,
			     const std::map<qsym,qinfo3<Tm>>& info_dict,
			     Tm* workspace){
   const bool debug = true;
   formulae.display("formulae");
   // initialization 
   int isym = wf.info.sym.isym();
   qsym sym;
   stensor3<Tm> opxwf0, opxwf;
   // opN*|wf>
   for(int it=0; it<formulae.size(); it++){
      const auto& HTerm = formulae.tasks[it];
      // term[it]*wf
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
         // impose antisymmetry here
         if(block == block2 and parity){ 
            if(block1 == "l"){ // lc or lr
	       opxwf.row_signed();
	    }else if(block1 == "c"){
	       opxwf.mid_signed();
	    }
	 }
         if(idx != 0){
            opxwf0.info = opxwf.info;
            opxwf0.setup_data(wptr);	 
         }
      } // idx
      linalg::xaxpy(Hwf.size(), 1.0, opxwf.data(), Hwf.data()); 
   } // it
   // opH*|wf>
   if(key != 'H') return;
   for(int it=0; it<formulae.size(); it++){
      const auto& HTerm = formulae.tasks[it];
      // term[it]*wf
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
	 // opN*|wf> 
         const auto& info = info_dict.at(sym);
	 Tm* wptr = workspace+opsize+(idx%2)*wfsize; 
	 opxwf.init(info,false);
	 opxwf.setup_data(wptr);
	 contract_qt3_qt2_info(block,opxwf0,optmp,opxwf,!dagger);
       	 // impose antisymmetry here
         if(block == block2 and parity){ 
            if(block1 == "l"){ // lc or lr
	       opxwf.row_signed();
	    }else if(block1 == "c"){
	       opxwf.mid_signed();
	    }
	 }
         if(idx != 0){
            opxwf0.info = opxwf.info;
            opxwf0.setup_data(wptr);	 
         }
      } // idx
      double fac = HTerm.Hsign(); // (opN)^H = sgn*opH
      linalg::xaxpy(Hwf.size(), fac, opxwf.data(), Hwf.data()); 
   } // it
}

template <typename Tm>
void symbolic_renorm_kernel2(const std::string superblock,
		             const stensor3<Tm>& site,
		             const integral::two_body<Tm>& int2e,
		             const oper_dict<Tm>& qops1,
		             const oper_dict<Tm>& qops2,
		             oper_dict<Tm>& qops,
			     const std::string fname, 
			     const bool debug){
   // generate formulae for renormalization first
   auto tasks = symbolic_renorm_formulae(superblock, int2e, qops1, qops2, qops, fname);
   if(debug) std::cout << "rank=" << qops.mpirank 
	               << " size[tasks]=" << tasks.size() 
		       << std::endl;
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
   size_t wfsize = preprocess_wf3size(site.info, info_dict);
   size_t tmpsize = opsize + 3*wfsize;
   size_t worktot = maxthreads*tmpsize;
   if(qops.mpirank == 0){
      std::cout << "maxthreads=" << maxthreads
                << " opsize=" << opsize
                << " wfsize=" << wfsize
                << " tmpsize=" << tmpsize
                << " worktot=" << worktot
                << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                << std::endl; 
   }
   Tm* workspace = new Tm[worktot];
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<tasks.size(); i++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      const auto& task = tasks[i];
      auto key = std::get<0>(task);
      auto index = std::get<1>(task);
      auto formula = std::get<2>(task);
      if(debug){
         std::cout << "rank=" << qops.mpirank 
		   << " i=" << i 
		   << " key=" << key
		   << " index=" << index
		   << std::endl;
	 formula.display("formula", 1);
      }
      // op|ket>
      stensor3<Tm> opxwf; 
      auto sym_op = qops.get_qsym_op(key, index);
      auto sym = sym_op + site.info.sym;
      opxwf.init(info_dict.at(sym), false);
      opxwf.setup_data(&workspace[omprank*tmpsize]);
      opxwf.clear();
      symbolic_renorm_single2(block1,block2,qops_dict,
		              key,formula,site,opxwf,
			      opsize,wfsize,info_dict,
			      &workspace[omprank*tmpsize+wfsize]);
      // <bra|op|ket>
      stensor2<Tm> op;
      op.init(qops(key)[index].info, false);
      op.setup_data(&workspace[omprank*tmpsize+wfsize]);
      op.clear();
      contract_qt3_qt3_info(superblock, site, opxwf, op);
      if(key == 'H' && qops.ifkr) op += op.K();
      linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
   } // i
   delete[] workspace;
}

} // ctns

#endif
