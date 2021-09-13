#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include "../core/serialization.h"
#include "qtensor/qtensor.h"
#include "ctns_topo.h"
#include "init_rbasis.h"
#include "init_phys.h" // get_qbond_phys

namespace ctns{

template <typename Km>
class comb{
   private:
      // serialize [for MPI]
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
//	 ar & topo & rsites & rwfuns;
      }
   public:
      // constructors
      comb(){
	 std::cout << "\ncomb: qkind=" << qkind::get_name<Km>() << std::endl;
         if(!qkind::is_available<Km>()) tools::exit("error: no such qkind for CTNS!");
      }
/*
      // helpers
      int get_nphysical() const{ return topo.nphysical; }
      int get_nroots() const{
	 assert(rwfuns.rows() == 1); // currently, only allow one symmetry sector
	 return rwfuns.qrow.get_dim(0);
      }
      qsym get_sym_state() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
         return rwfuns.qrow.get_sym(0);
      }
      // return rwfun for iroot, extracted from rwfuns
      qtensor2<typename Km::dtype> get_iroot(const int iroot) const;
      //
      //				  |
      //    MPS-like:	    Additional: --pc
      //     \|/			 \|/
      //    --p--			--p--
      //
      qbond get_qc(const comb_coord& p) const{
         auto pc = topo.get_node(p).center;
         bool physical = (pc == coord_phys);
         return physical? get_qbond_phys(Km::isym) : rsites.at(pc).qrow; 
      }
      //
      // 			          |
      //    MPS-like:     Additional: --p 
      //      |      |                 /|\
      //    --pl-->--p--     	        --pl--
      //
      qbond get_ql(const comb_coord& p) const{
         auto pl = topo.get_node(p).left;
         bool cturn = topo.is_cturn(pl,p);
         return cturn? lsites.at(pl).qmid : lsites.at(pl).qcol;
      }
      //
      // MPS-like:
      //    |     |
      //  --p--<--pr-- : qrow of rsites[pr]
      //
      template <typename Km>
      qbond get_qr(const comb_coord& p) const{
         auto pr = topo.get_node(p).right;
         return rsites.at(pr).qrow;
      }
*/
   public:
      using Tm = typename Km::dtype;
      // -- CTNS ---
      topology topo;
      // used in initialization & debug operators 
      std::vector<renorm_basis<Tm>> rbases;
      // right canonical form 
      std::vector<stensor3<Tm>> rsites;
      // wavefunction at the left boundary -*-
      stensor2<Tm> rwfuns; 
      // left canonical form 
      std::vector<stensor3<Tm>> lsites;
      // propagation of initial guess 
      std::vector<stensor3<Tm>> psi; 
      // --- MPI ---
#ifndef SERIAL
      boost::mpi::communicator world;
#endif
};

/*
// return rwfun for iroot, extracted from rwfuns
template <typename Km>
qtensor2<typename Km::dtype> comb<Km>::get_iroot(const int iroot) const{
   assert(rwfuns.rows() == 1);
   qbond qrow({{rwfuns.qrow.get_sym(0),1}});
   qtensor2<typename Km::dtype> rwfun(rwfuns.sym, qrow, rwfuns.qcol, rwfuns.dir);
   const auto& blk0 = rwfuns(0,0);
   auto& blk = rwfun(0,0);
   for(int ic=0; ic<rwfuns.qcol.get_dim(0); ic++){
      blk(0,ic) = blk0(iroot,ic);
   }
   return rwfun;
}
*/

} // ctns

#endif
