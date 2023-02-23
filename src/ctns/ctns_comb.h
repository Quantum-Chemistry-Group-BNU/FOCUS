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

   template <typename Tm, typename Qm>
      size_t display_vec_size(const Qm& vec, std::string name){
         std::cout << " " << name << ": len=" << vec.size() << " mem(GB)=";
         size_t sz = 0;
         for(const auto& item : vec){
            sz += item.size();
         }
         std::cout << tools::sizeGB<Tm>(sz) << std::endl;
         return sz;
      }

   template <typename Km>
      class comb{
         private:
            // serialize [for MPI] in src/drivers/ctns.cpp
            friend class boost::serialization::access;	   
            template <class Archive>
               void save(Archive & ar, const unsigned int version) const{
                  ar & topo
                     & rbases // ZL@20220606: for usage in debug oper_rbasis
                     & rsites
                     & rwfuns;
                  /*
                     for(int idx=0; idx<topo.ntotal; idx++){
                     ar & rsites[idx];
                     }
                     */
               }
            template <class Archive>
               void load(Archive & ar, const unsigned int version){
                  ar & topo 
                     & rbases
                     & rsites
                     & rwfuns;
                  /*
                     rsites.resize(topo.ntotal);
                     for(int idx=0; idx<topo.ntotal; idx++){
                     ar & rsites[idx];
                     }
                     */
               }
            BOOST_SERIALIZATION_SPLIT_MEMBER()
         public:
               // constructors
               comb(){
                  //std::cout << "\ncomb: qkind=" << qkind::get_name<Km>() << std::endl;
                  if(!qkind::is_available<Km>()) tools::exit("error: no such qkind for CTNS!");
               }
               // helpers
               int get_nphysical() const{ return topo.nphysical; }
               qsym get_sym_state() const{
                  assert(rwfuns[0].rows() == 1); // only one symmetry sector
                  return rwfuns[0].info.qrow.get_sym(0);
               }
               int get_nroots() const{ return rwfuns.size(); }
               // wf2(iroot,icol)
               stensor2<typename Km::dtype> get_wf2() const{
                  int nroots = rwfuns.size();
                  qbond qrow({{this->get_sym_state(),nroots}});
                  const auto& qcol = rwfuns[0].info.qcol;
                  const auto& dir = rwfuns[0].info.dir;
                  stensor2<typename Km::dtype> wf2(rwfuns[0].info.sym, qrow, qcol, dir);
                  for(int iroot=0; iroot<nroots; iroot++){
                     for(int ic=0; ic<rwfuns[0].info.qcol.get_dim(0); ic++){
                        wf2(0,0)(iroot,ic) = rwfuns[iroot](0,0)(0,ic);
                     }
                  }
                  return wf2;
               }
               // print size 
               size_t display_size() const{
                  std::cout << "comb::display_size" << std::endl;
                  size_t sz = 0;
                  using Tm = typename Km::dtype;
                  sz += display_vec_size<Tm>(rbases, "rbases");
                  sz += display_vec_size<Tm>(rsites, "rsites");
                  sz += display_vec_size<Tm>(rwfuns, "rwfuns");
                  sz += display_vec_size<Tm>(lsites, "lsites");
                  sz += display_vec_size<Tm>(psi, "psi");
                  std::cout << "tot size of comb (GB)=" << tools::sizeGB<Tm>(sz) << std::endl;
                  return sz;
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
               std::vector<stensor2<Tm>> rwfuns; 
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
