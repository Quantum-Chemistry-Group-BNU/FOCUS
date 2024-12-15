#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include "../qtensor/qtensor.h"
#include "ctns_topo.h"
#include "init_rbasis.h"
#include "init_phys.h" // get_qbond_vac
#include "../core/mem_status.h"
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
            // print shape
            void display_shape() const{
               std::cout << "\ncomb::display_shape qkind=" << qkind::get_name<Qm>() << std::endl;
               for(int i=0; i<sites.size(); i++){
                  auto shape = sites[i].get_shape();
                  std::cout << " idx=" << i
                     << " node=" << topo.rcoord[i]
                     << " shape(l,r,c)=(" << std::get<0>(shape) << ","
                     << std::get<1>(shape) << ","
                     << std::get<2>(shape) << ")";
                  if(!Qm::ifabelian){
                     auto shapeU1 = sites[i].get_shapeU1();
                     std::cout << " shapeU1(l,r,c)=(" << std::get<0>(shapeU1) << ","
                        << std::get<1>(shapeU1) << ","
                        << std::get<2>(shapeU1) << ")";
                  }
                  std::cout << std::endl;
               } // i
               auto wf2 = this->get_wf2();
               wf2.print("wf2");
            }
            // maximal bond dimension
            int get_dmax() const{
               int dmax = -1;
               for(int i=0; i<sites.size(); i++){
                  dmax = std::max(dmax,std::get<1>(sites[i].get_shape()));
               } // i
               return dmax;
            }
            // print size 
            size_t display_size() const{
               std::cout << "comb::display_size qkind=" << qkind::get_name<Qm>() << std::endl;
               size_t sz = 0;
               sz += display_vec_size<Tm>(rbases, "rbases");
               sz += display_vec_size<Tm>(sites, "sites");
               sz += display_vec_size<Tm>(rwfuns, "rwfuns");
               sz += display_vec_size<Tm>(cpsi, "cpsi");
               std::cout << "total mem of comb=" << sz << ":" 
                  << tools::sizeMB<Tm>(sz) << "MB:"
                  << tools::sizeGB<Tm>(sz) << "GB"
                  << std::endl;
               get_mem_status(0);
               return sz;
            }
            // helpers
            int get_nphysical() const{ return topo.nphysical; }
            qsym get_qsym_state() const{
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns[0].info.qrow.get_sym(0);
            }
            int get_nroots() const{ 
               assert(rwfuns[0].rows() == 1); // only one symmetry sector
               return rwfuns.size(); 
            }
            // wf2(iroot,icol): ->-*->-
            qtensor2<Qm::ifabelian,Tm> get_wf2() const{
               int nroots = this->get_nroots();
               auto sym_state = this->get_qsym_state();
               qbond qrow({{sym_state,nroots}});
               const auto& qcol = rwfuns[0].info.qcol;
               const auto& dir = rwfuns[0].info.dir;
               assert(dir == dir_RWF);
               // matching state symmetry
               int jdx = -1;
               for(int j=0; j<qcol.size(); j++){
                  if(qcol.get_sym(j) == sym_state){
                     jdx = j;
                     break;
                  }
               }
               if(jdx == -1){
                  std::cout << "error: no matching symmetry for sym_state=" << sym_state << std::endl;
                  exit(1);
               }
               // copy data
               qtensor2<Qm::ifabelian,Tm> wf2(rwfuns[0].info.sym, qrow, qcol, dir);
               for(int iroot=0; iroot<nroots; iroot++){
                  for(int ic=0; ic<rwfuns[0].info.qcol.get_dim(jdx); ic++){
                     wf2(0,jdx)(iroot,ic) = rwfuns[iroot](0,jdx)(0,ic);
                  }
               }
               return wf2;
            }
            // reorthogonalize {cpsi}
            void orthonormalize_cpsi(){
               assert(cpsi.size() > 0);
               size_t ndim = cpsi[0].size();
               int nroots = cpsi.size();
               std::vector<Tm> v0(ndim*nroots);
               for(int i=0; i<nroots; i++){
                  cpsi[i].to_array(&v0[ndim*i]);
               }
               int nindp = linalg::get_ortho_basis(ndim, nroots, v0.data()); // reorthogonalization
               assert(nindp == nroots);
               for(int i=0; i<nroots; i++){
                  cpsi[i].from_array(&v0[ndim*i]);
               }
               v0.clear();
            }
         public:
            // -- CTNS ---
            topology topo;
            // used in initialization & debug operators 
            std::vector<renorm_basis<Tm>> rbases;
            // mixed canonical form:
            // note that the sites are count from the right as in RCF [!]
            // that is the first one is the last site in MPS!
            std::vector<qtensor3<Qm::ifabelian,Tm>> sites;
            // wavefunction at the left boundary -*-
            std::vector<qtensor2<Qm::ifabelian,Tm>> rwfuns;
            // central wavefunction
            std::vector<qtensor3<Qm::ifabelian,Tm>> cpsi;
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
            mpi_wrapper::broadcast(comm, icomb.sites[i], root);
         }
         boost::mpi::broadcast(comm, icomb.rwfuns, root);
      }

} // mpi_wrapper

#endif

#endif
