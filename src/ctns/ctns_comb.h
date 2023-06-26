#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include "../qtensor/qtensor.h"
#include "ctns_topo.h"
#include "init_rbasis.h"
#include "init_phys.h" // get_qbond_vac

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <typename Tm, typename Vm>
      size_t display_vec_size(const Vm& vec, std::string name){
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

   template <typename Qm, typename Tm>
      struct comb{
         public:
            // constructors
            comb(){
               if(!qkind::is_available<Qm>()) tools::exit("error: no such qkind for CTNS!");
            }
            // print size 
            size_t display_size() const{
               std::cout << "comb::display_size" << std::endl;
               size_t sz = 0;
               sz += display_vec_size<Tm>(rbases, "rbases");
               sz += display_vec_size<Tm>(sites, "sites");
               sz += display_vec_size<Tm>(rwfuns, "rwfuns");
               sz += display_vec_size<Tm>(cpsi, "cpsi");
               std::cout << "total mem of comb=" << sz << ":" 
                  << tools::sizeMB<Tm>(sz) << "MB:"
                  << tools::sizeGB<Tm>(sz) << "GB"
                  << std::endl;
               return sz;
            }
            // helpers
            int get_nphysical() const{ return topo.nphysical; }
            qsym get_sym_state() const{
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns[0].info.qrow.get_sym(0);
            }
            int get_nroots() const{ 
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns.size(); 
            }
            // wf2(iroot,icol): ->-*->-
            stensor2<Tm> get_wf2() const{
               int nroots = this->get_nroots();
               qbond qrow({{this->get_sym_state(),nroots}});
               const auto& qcol = rwfuns[0].info.qcol;
               const auto& dir = rwfuns[0].info.dir;
               assert(dir == dir_RWF);
               stensor2<Tm> wf2(rwfuns[0].info.sym, qrow, qcol, dir);
               for(int iroot=0; iroot<nroots; iroot++){
                  for(int ic=0; ic<rwfuns[0].info.qcol.get_dim(0); ic++){
                     wf2(0,0)(iroot,ic) = rwfuns[iroot](0,0)(0,ic);
                  }
               }
               return wf2;
            }
         public:
            // -- CTNS ---
            topology topo;
            // used in initialization & debug operators 
            std::vector<renorm_basis<Tm>> rbases;
            // mixed canonical form
            std::vector<stensor3<Tm>> sites;
            // wavefunction at the left boundary -*-
            std::vector<stensor2<Tm>> rwfuns;
            // central wavefunction
            std::vector<stensor3<Tm>> cpsi;
#ifndef SERIAL
            boost::mpi::communicator world; // for MPI
#endif
      };

} // ctns

#ifndef SERIAL

namespace mpi_wrapper{

   // icomb: assuming the individual size of sites is small
   template <typename Qm, typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::comb<Qm,Tm>& icomb, int root){
         boost::mpi::broadcast(comm, icomb.topo, root);
         boost::mpi::broadcast(comm, icomb.rbases, root);
         int rank = comm.rank();
         if(rank != root) icomb.sites.resize(icomb.topo.ntotal); // reserve space
         // sites could be packed in future following: 
         // https://gist.github.com/hsidky/2f0e075095026d2ebda1
         for(int i=0; i<icomb.topo.ntotal; i++){
            boost::mpi::broadcast(comm, icomb.sites[i], root);
         }
         boost::mpi::broadcast(comm, icomb.rwfuns, root);
      }

} // mpi_wrapper

#endif

#endif
