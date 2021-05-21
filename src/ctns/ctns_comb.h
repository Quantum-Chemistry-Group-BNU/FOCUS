#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include "../core/serialization.h"
#include "ctns_phys.h"
#include "ctns_kind.h"
#include "ctns_topo.h"
#include "ctns_rbasis.h"
#include "qtensor.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace ctns{

template <typename Tm>
using rbases_type = std::map<comb_coord,renorm_basis<Tm>>;
template <typename Tm>
using sites_type = std::map<comb_coord,qtensor3<Tm>>;

template <typename Km>
class comb{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & topo & rsites & rwfuns;
      }
   public:
      // constructors
      comb(){
         if(!kind::is_available<Km>()) tools::exit("error: no such kind for CTNS!");
      }
      // helpers
      int get_nphysical() const{ return topo.nphysical; }
      int get_nstates() const{
	 assert(rwfuns.rows() == 1); // currently, only allow one symmetry sector
	 return rwfuns.qrow.get_dim(0);
      }
      qsym get_sym_state() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
         return rwfuns.qrow.get_sym(0);
      }
      // return rwfun for istate, extracted from rwfuns
      qtensor2<typename Km::dtype> get_istate(const int istate) const;
      // get_qc/ql/qr used in setting up optimization
      qbond get_qc(const comb_coord& p) const;
      qbond get_ql(const comb_coord& p) const;
      qbond get_qr(const comb_coord& p) const;
      std::vector<int> get_suppc(const comb_coord& p, const bool ifprt=true) const;
      std::vector<int> get_suppl(const comb_coord& p, const bool ifprt=true) const;
      std::vector<int> get_suppr(const comb_coord& p, const bool ifprt=true) const;
   public:
      // -- CTNS ---
      topology topo;
      sites_type<typename Km::dtype> rsites; // right canonical form 
      qtensor2<typename Km::dtype> rwfuns; // wavefunction at the left boundary -*-
      // --- auxilliary data ---
      rbases_type<typename Km::dtype> rbases; // used in initialization & debug operators 
      sites_type<typename Km::dtype> lsites; // left canonical form 
      std::vector<qtensor3<typename Km::dtype>> psi; // propagation of initial guess 
      // --- MPI ---
#ifndef SERIAL
      boost::mpi::communicator world;
#endif
};

// return rwfun for istate, extracted from rwfuns
template <typename Km>
qtensor2<typename Km::dtype> comb<Km>::get_istate(const int istate) const{
   assert(rwfuns.rows() == 1);
   qbond qrow({{rwfuns.qrow.get_sym(0),1}});
   qtensor2<typename Km::dtype> rwfun(rwfuns.sym, qrow, rwfuns.qcol, rwfuns.dir);
   const auto& blk0 = rwfuns(0,0);
   auto& blk = rwfun(0,0);
   for(int ic=0; ic<rwfuns.qcol.get_dim(0); ic++){
      blk(0,ic) = blk0(istate,ic);
   }
   return rwfun;
}

// symmetry information used in opt_sweep
template <typename Km>
qbond comb<Km>::get_qc(const comb_coord& p) const{
   //
   //				  |
   // MPS-like:	    Additional: --pc
   //  \|/			 \|/
   // --p--			--p--
   //
   auto pc = topo.get_node(p).center;
   bool physical = (pc == coord_phys);
   return physical? get_qbond_phys(Km::isym) : rsites.at(pc).qrow; 
}

template <typename Km>
std::vector<int> comb<Km>::get_suppc(const comb_coord& p, const bool ifprt) const{
   auto pc = topo.get_node(p).center;
   bool physical = (pc == coord_phys);
   auto suppc = physical? std::vector<int>({topo.get_node(p).pindex}) : topo.get_node(pc).rsupport;
   if(ifprt){
      std::cout << " suppc :";
      for(const auto& k : suppc) std::cout << " " << k;
      std::cout << std::endl; 
   }
   return suppc;
}

template <typename Km>
qbond comb<Km>::get_ql(const comb_coord& p) const{
   //
   // 			          |
   // MPS-like:     Additional: --p 
   //   |      |                 /|\
   // --pl-->--p--     	        --pl--
   //
   auto pl = topo.get_node(p).left;
   bool cturn = topo.is_cturn(pl,p);
   return cturn? lsites.at(pl).qmid : lsites.at(pl).qcol;
}

template <typename Km>
std::vector<int> comb<Km>::get_suppl(const comb_coord& p, const bool ifprt) const{
   auto suppl = topo.get_node(p).lsupport;
   if(ifprt){
      std::cout << " suppl :";
      for(const auto& k : suppl) std::cout << " " << k;
      std::cout << std::endl; 
   }
   return suppl;
}

template <typename Km>
qbond comb<Km>::get_qr(const comb_coord& p) const{
   //
   // MPS-like:
   //    |     |
   //  --p--<--pr-- : qrow of rsites[pr]
   //
   auto pr = topo.get_node(p).right;
   return rsites.at(pr).qrow;
}

template <typename Km>
std::vector<int> comb<Km>::get_suppr(const comb_coord& p, const bool ifprt) const{
   auto pr = topo.get_node(p).right;
   auto suppr = topo.get_node(pr).rsupport;
   if(ifprt){
      std::cout << " suppr :";
      for(const auto& k : suppr) std::cout << " " << k;
      std::cout << std::endl; 
   }
   return suppr;
}

} // ctns

#endif
