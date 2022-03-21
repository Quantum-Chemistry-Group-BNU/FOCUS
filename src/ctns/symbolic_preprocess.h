#ifndef SYMBOLIC_PREPROCESS_H
#define SYMBOLIC_PREPROCESS_H

#include "oper_dict.h"

namespace ctns{

const bool debug_preprocess = false;
extern const bool debug_preprocess;

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
size_t preprocess_opsize(const oper_dictmap<Tm>& qops_dict){
   size_t opsize = 0;
   for(const auto& pr : qops_dict){
      opsize = std::max(opsize, pr.second.opsize());
   }
   return opsize;
}

template <typename Tm>
size_t preprocess_wfsize(const qinfo3<Tm>& wf3info, 
    	 	         std::map<qsym,qinfo3<Tm>>& info_dict,
		         const int nbody=2){
   const auto& sym_state = wf3info.sym; 
   const auto& qrow = wf3info.qrow;
   const auto& qcol = wf3info.qcol;
   const auto& qmid = wf3info.qmid;
   const auto& dir = wf3info.dir;
   int isym = sym_state.isym();
   if(debug_preprocess){ 
      std::cout << "ctns::preprocess_wfsize"
	        << " isym=" << isym
	        << " sym_state=" << sym_state	
	        << " nbody=" << nbody
		<< std::endl;
   }
   // generate all operators
   auto sym_ops = get_qsym_allops(isym,nbody);
   size_t wfsize = 0;
   int idx = 0;
   for(const auto& sym_op : sym_ops){
      auto sym = sym_op+sym_state;
      qinfo3<Tm> info;
      info.init(sym, qrow, qcol, qmid, dir);
      info_dict[sym] = info;
      wfsize = std::max(wfsize, info._size);
      if(debug_preprocess){
         std::cout << " idx=" << idx 
                   << " sym_op=" << sym_op
                   << " sym_state=" << sym_state
                   << " sym=" << sym
                   << " size=" << info._size
                   << std::endl;
      }
      idx++;
   }
   return wfsize;
}

template <typename Tm>
size_t preprocess_wfsize(const qinfo4<Tm>& wf4info, 
    	 	         std::map<qsym,qinfo4<Tm>>& info_dict,
		         const int nbody=2){
   const auto& sym_state = wf4info.sym; 
   const auto& qrow = wf4info.qrow;
   const auto& qcol = wf4info.qcol;
   const auto& qmid = wf4info.qmid;
   const auto& qver = wf4info.qver;
   int isym = sym_state.isym();
   if(debug_preprocess){ 
      std::cout << "ctns::preprocess_wfsize"
	        << " isym=" << isym 
	        << " sym_state=" << sym_state	
		<< " nbody=" << nbody 
		<< std::endl;
   }
   // generate all operators
   auto sym_ops = get_qsym_allops(isym,nbody);
   size_t wfsize = 0;
   int idx = 0;
   for(const auto& sym_op : sym_ops){
      auto sym = sym_op+sym_state;
      qinfo4<Tm> info;
      info.init(sym, qrow, qcol, qmid, qver);
      info_dict[sym] = info;
      wfsize = std::max(wfsize, info._size);
      if(debug_preprocess){
         std::cout << " idx=" << idx 
                   << " sym_op=" << sym_op
                   << " sym_state=" << sym_state
                   << " sym=" << sym
                   << " size=" << info._size
                   << std::endl;
      }
      idx++;
   }
   return wfsize;
}

} // ctns

#endif
