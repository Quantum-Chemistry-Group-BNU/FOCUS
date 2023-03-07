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
         std::cout << " " << name << ": len=" << vec.size() << " mem=";
         size_t sz = 0;
         for(int i=0; i<vec.size(); i++){
            sz += vec[i].size();
         }
         std::cout << sz << ":" 
            << tools::sizeMB<Tm>(sz) << "MB:"
            << tools::sizeGB<Tm>(sz) << "GB"
            << std::endl;
         return sz;
      }

   template <typename Km>
      class comb{
         private:
            // serialize [for MPI] in src/drivers/ctns.cpp
            friend class boost::serialization::access;	   
            template <class Archive>
               void serialize(Archive & ar, const unsigned int version){
                  ar & topo
                     & rbases // ZL@20220606: for usage in debug oper_rbasis
                     & sites;
                  & cpsi;
               }
         public:
            // constructors
            comb(){
               if(!qkind::is_available<Km>()) tools::exit("error: no such qkind for CTNS!");
            }
            // print size 
            size_t display_size() const{
               std::cout << "comb::display_size" << std::endl;
               size_t sz = 0;
               using Tm = typename Km::dtype;
               sz += display_vec_size<Tm>(rbases, "rbases");
               sz += display_vec_size<Tm>(sites, "sites");
               sz += display_vec_size<Tm>(cpsi, "cpsi");
               std::cout << "total mem of comb=" << sz << ":" 
                  << tools::sizeMB<Tm>(sz) << "MB:"
                  << tools::sizeGB<Tm>(sz) << "GB"
                  << std::endl;
               return sz;
            }
            // helpers
            int get_nphysical() const{ return topo.nphysical; }
            qsym get_rcanon_sym() const{
               const auto& site0 = sites[topo.rindex.at(std::make_pair(0,0))];
               assert(site0.rows() == 1);
               return site0.info.qrow.get_sym(0);
            }
            int get_rcanon_nroots() const{
               const auto& site0 = sites[topo.rindex.at(std::make_pair(0,0))];
               assert(site0.rows() == 1);
               return site0.info.qrow.get_dim(0);
            }
            // stack_cpsi at the first site in right canonical form
            //          |                      |       
            // state ->-|->- [RCF]  <=  vac -<-|->- [cpsi] 
            //         /|\                    /|\
            //          vac                  state  
            void cpsi0_to_site0(){
               int nroots = cpsi.size();
               qsym sym_state = cpsi[0].info.sym;
               qbond qrow({{sym_state, nroots}});
               qbond qcol = cpsi[0].info.qcol;
               qbond qmid = cpsi[0].info.qmid;
               stensor3<Tm> site0(qsym(Km::isym), qrow, qcol, qmid); // dir_RCF 
               for(int bc=0; bc<qcol.size(); bc++){
                  for(int bm=0; bm<qmid.size(); bm++){
                     auto blk = site0(0,bc,bm);
                     if(blk.empty()) continue;
                     int cdim = qcol.get_dim(bc);
                     int mdim = qmid.get_dim(bm);
                     for(int im=0; im<mdim; im++){
                        for(int ic=0; ic<cdim; ic++){
                           for(int iroot=0; iroot<nroots; iroot++){
                              const auto blk0 = cpsi[iroot](0,bc,bm);
                              blk(iroot,ic,im) = blk0(0,ic,im); 
                           }  // iroot
                        } // ic
                     } // im
                  } // bm
               } // bc
               sites[topo.rindex.at(std::make_pair(0,0))] = std::move(site0);
            }
            // used in initialization:  generate initial guess for 
            // the initial sweep optimization at p=(1,0): CRRR => LCRR (L=Id)
            void site0_to_cpsi1(const int nroots){
               if(this->get_rcanon_nroots() < nroots){
                  std::cout << "dim(psi0)=" << this->get_rcanon_nroots() 
                            << " nroots=" << nroots 
                            << std::endl;
                  tools::exit("error in initiate_psi0: requested nroots exceed!");
               }
               // site0_to_cpsi0 & cpsi0_to_cpsi1
               auto& site0 = sites[topo.rindex.at(std::make_pair(0,0))];
               const auto& site1 = sites[topo.rindex.at(std::make_pair(1,0))];
               qsym sym_state = this->get_rcanon_sym();
               qbond qcol = site0.info.qcol;
               qbond qmid = site0.info.qmid;
               cpsi.resize(nroots);
               for(int iroot=0; iroot<nroots; iroot++){
                  // construct cpsi0
                  stensor3<Tm> cpsi0(sym_state, get_qbond_vac(Km::isym), site0.info.qcol, site0.info.qmid, dir_WF3);
                  for(int bc=0; bc<qcol.size(); bc++){
                     for(int bm=0; bm<qmid.size(); bm++){                        
                        auto blk = cpsi0(0,bc,bm);
                        if(blk.empty()) continue;
                        int cdim = qcol.get_dim(bc);
                        int mdim = qmid.get_dim(bm);
                        for(int im=0; im<mdim; im++){
                           for(int ic=0; ic<cdim; ic++){
                              const auto blk0 = site0(0,bc,bm);
                              blk(0,ic,im) = blk0(iroot,ic,im); 
                           } // ic
                        } // im
                     } // bm
                  } // bc
                  // construct cpsi1
                  auto wf2 = cpsi0.merge_lc(); // (1,n,r)->(n,r)
                  cpsi[iroot] = contract_qt3_qt2("l", site1, wf2);
               }  // iroot
               site0 = get_left_bsite<Tm>(Km::isym); // C[0]R[1] => L[0]C[1] (L[0]=Id) 
            }
            // rank-2 wavefunction at the first site: ->-*->- in ctns_alg.h
            stensor2<typename Km::dtype> get_rwfun(const int iroot) const{
               int nroots = this->get_rcanon_nroots();
               qsym sym_state = this->get_rcanon_sym();
               qbond qrow({{sym_state, 1}});
               qbond qcol({{sym_state, nroots}});
               stensor2<Tm> rwfun(qsym(Km::isym), qrow, qcol, dir_RWFUN);
               rwfun(0,0)(0,iroot) = 1.0; 
               return rwfun;
            }
         public:
            using Tm = typename Km::dtype;
            // -- CTNS ---
            topology topo;
            // used in initialization & debug operators 
            std::vector<renorm_basis<Tm>> rbases;
            // mixed canonical form
            std::vector<stensor3<Tm>> sites;
            // central wavefunction
            std::vector<stensor3<Tm>> cpsi;
#ifndef SERIAL
            boost::mpi::communicator world; // for MPI
#endif
      };

} // ctns

#endif
