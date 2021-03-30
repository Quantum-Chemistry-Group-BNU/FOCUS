#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include "ctns_topo.h"
#include "ctns_kind.h"

//#include "qtensor.h"

namespace ctns{

template <typename Tm>
class comb{
   public:
      comb(const topology& topo1): topo(topo1) 
      {
         if(!kind::is_available<Tm>()){
            std::cout << "error: no such kind for CTNS!" << std::endl;
	    exit(1);
	 }
      }
   public:
      topology topo;
//      std::map<comb_coord,qtensor3<Tm>> rsites; // right canonical form 
//      qtensor2<Tm> rwfuns; // wavefunction at the left boundary -*-
};

/*
// comb tensor network states 
template <typename Tm>	
class comb{
   public:
      comb(const topology& topo1): topo(topo1) {}
      // helpers
      int get_nphysical() const{ return topo.nphysical; }
      int get_nstate() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
	 return rwfuns.qrow.get_dim(0);
      }
      qsym get_sym_state() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
         return rwfuns.qrow.get_sym(0);
      }
      // return rwfun for istate, extracted from rwfuns
      qtensor2<Tm> get_state(const int istate) const{
         assert(rwfuns.rows() == 1);
	 qsym_space qrow({{rwfuns.qrow.get_sym(0),1}});
	 qtensor2<Tm> rwfun(rwfuns.sym, qrow, rwfuns.qcol, rwfuns.dir);
	 const auto& blk0 = rwfuns(0,0);
	 auto& blk = rwfun(0,0);
	 for(int ic=0; ic<rwfuns.qcol.get_dim(0); ic++){
	    blk(0,ic) = blk0(istate,ic);
	 }
	 return rwfun;
      }
      // symmetry information used in opt_sweep
      qsym_space get_qc(const comb_coord& p) const{
	 //
	 //				  |
	 // MPS-like:	    Additional: --pc
	 //  \|/			 \|/
	 // --p--			--p--
	 //
         auto pc = topo.get_node(p).center;
	 bool physical = (pc == coord_phys);
         return physical? get_qsym_space_phys<Tm>() : rsites.at(pc).qrow; 
      }
      qsym_space get_ql(const comb_coord& p) const{
	 //
	 // 			           |
	 // MPS-like:       Additional:  --p 
	 //   |      | 			  /|\
	 // --pl-->--p-- 		 --pl--
	 //
         auto pl = topo.get_node(p).left;
         bool cturn = (topo.node_type(pl) == 3 and p.second == 1);
	 return cturn? lsites.at(pl).qmid : lsites.at(pl).qcol;
      }
      qsym_space get_qr(const comb_coord& p) const{
	 //
	 // MPS-like:
	 //    |     |
	 //  --p--<--pr-- : qrow of rsites[pr]
	 //
         auto pr = topo.get_node(p).right;
         return rsites.at(pr).qrow;
      }
   public:
      topology topo;
      std::map<comb_coord,renorm_basis<Tm>> rbases; // renormalized basis from SCI
      std::map<comb_coord,qtensor3<Tm>> rsites; // right canonical form 
      qtensor2<Tm> rwfuns; // wavefunction at the left boundary -*-
      std::map<comb_coord,qtensor3<Tm>> lsites; // left canonical form 
      //std::vector<qtensor3<Tm>> psi; // propagation of initial guess 
};
*/

} // ctns

#endif
