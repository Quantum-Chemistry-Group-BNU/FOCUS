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
      // serialize [for MPI] in src/drivers/ctns.cpp
      friend class boost::serialization::access;	   
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & topo & rwfuns;
	 for(int idx=0; idx<topo.ntotal; idx++){
	    ar & rsites[idx];
	 }
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & topo & rwfuns;
	 rsites.resize(topo.ntotal);
	 for(int idx=0; idx<topo.ntotal; idx++){
	    ar & rsites[idx];
	 }
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
   public:
      // constructors
      comb(){
	 std::cout << "\ncomb: qkind=" << qkind::get_name<Km>() << std::endl;
         if(!qkind::is_available<Km>()) tools::exit("error: no such qkind for CTNS!");
      }
      // helpers
      int get_nphysical() const{ return topo.nphysical; }
      int get_nroots() const{
	 assert(rwfuns.rows() == 1); // currently, only allow one symmetry sector
	 return rwfuns.info.qrow.get_dim(0);
      }
      qsym get_sym_state() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
         return rwfuns.info.qrow.get_sym(0);
      }
      // return rwfun for iroot, extracted from rwfuns
      stensor2<typename Km::dtype> get_iroot(const int iroot) const{
         assert(rwfuns.rows() == 1);
         qbond qrow({{rwfuns.info.qrow.get_sym(0),1}});
         stensor2<typename Km::dtype> rwfun(rwfuns.info.sym, qrow, rwfuns.info.qcol, rwfuns.info.dir);
	 // copy data from blk0 to blk
         const auto blk0 = rwfuns(0,0);
         auto blk = rwfun(0,0);
         for(int ic=0; ic<rwfuns.info.qcol.get_dim(0); ic++){
            blk(0,ic) = blk0(iroot,ic);
         }
         return rwfun;
      }
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

} // ctns

#endif
