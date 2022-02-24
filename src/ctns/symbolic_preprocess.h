#ifndef SYMBOLIC_PREPROCESS_H
#define SYMBOLIC_PREPROCESS_H

#include "oper_dict.h"
#include "symbolic_oper.h"

namespace ctns{

inline std::set<qsym> get_qsym_allops(const int isym,
				      const int nbody){
   std::set<qsym> sym_ops;
   if(isym == 0){
      sym_ops.insert( qsym(0,0) );
      sym_ops.insert( qsym(0,1) );
   }else if(isym == 1){
      for(int n=-nbody; n<=nbody; n++){
         sym_ops.insert( qsym(1,n) );
      }
   }else if(isym == 2){
      //    : (0,0)
      // ap+: (1,1),(1,-1)
      // aq : (-1,1),(-1,-1)
      // ap^+aq^+: (2,2),(2,0),(2,-2)
      // ap^+aq  : (0,2),(0,0),(0,-2)
      // ap aq   : (-2,2),(-2,0),(-2,-2)
      // ...
      for(int nop=0; nop<=nbody; nop++){
         for(int n=-nop; n<=nop; n+=2){
            for(int m=-nop; m<=nop; m+=2){
	       sym_ops.insert( qsym(2,n,m) );
            }
	 }
      }
   }
   return sym_ops;
}

template <typename Tm>
std::pair<size_t,size_t> symbolic_preprocess(const oper_dictmap<Tm>& qops_dict,
				             const qsym& sym_state,
			 	             const int nbody=2){
   const bool debug = true;
   int dims = qops_dict.size();
   if(debug){
      std::cout << "ctns::symbolic_preprocess"
	        << " nbody=" << nbody
		<< " dims=" << dims 
		<< std::endl;
   }
   int isym = (qops_dict.begin()->second).isym;
   // op2
   size_t opsize = 0;
   for(const auto& pr : qops_dict){
      assert(isym == pr.second.isym);
      opsize = std::max(opsize, pr.second.opsize());
   }
   // wf3 / wf4
   auto sym_ops = get_qsym_allops(isym,nbody);
   size_t wfsize = 0;
   if(dims == 3){
      auto qrow = qops_dict.at("l").qket;
      auto qcol = qops_dict.at("r").qket;
      auto qmid = qops_dict.at("c").qket;
      int idx = 0;
      for(const auto& sym_op : sym_ops){
	 auto sym = sym_op+sym_state;
         qinfo3<Tm> info3;
	 info3.init(sym, qrow, qcol, qmid, {1,1,1});
	 wfsize = std::max(wfsize, info3._size);
	 if(debug){
            std::cout << " idx=" << idx 
	              << " sym_op=" << sym_op
	              << " sym_state=" << sym_state
	              << " sym=" << sym
	              << " size=" << info3._size
	              << std::endl;
	 }
	 idx++;
      }
   }else if(dims == 4){
      auto qrow = qops_dict.at("l").qket;
      auto qcol = qops_dict.at("r").qket;
      auto qmid = qops_dict.at("c1").qket;
      auto qver = qops_dict.at("c2").qket;
      int idx = 0;
      for(const auto& sym_op : sym_ops){
	 auto sym = sym_op+sym_state;
         qinfo4<Tm> info4;
	 info4.init(sym, qrow, qcol, qmid, qver);
	 wfsize = std::max(wfsize, info4._size);
	 if(debug){
            std::cout << " idx=" << idx 
	              << " sym_op=" << sym_op
	              << " sym_state=" << sym_state
	              << " sym=" << sym
	              << " size=" << info4._size
	              << std::endl;
	 }
	 idx++;
      }
   }
   if(debug){ 
      std::cout << " summary: isym=" << isym 
	        << " opsize=" << opsize 
	        << " wfsize=" << wfsize 
		<< std::endl;
   }
   return std::make_pair(opsize,wfsize);
}

} // ctns

#endif
