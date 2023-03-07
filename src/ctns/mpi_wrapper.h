#ifndef SERIAL

#ifndef MPI_WRAPPER_H
#define MPI_WRAPPER_H

#include <boost/mpi.hpp>
#include "../core/tools.h"
#include "../core/integral.h"
#include "qtensor/qtensor.h"
#include "ctns_comb.h"
#include "ctns_comb0.h"

namespace mpi_wrapper{

   template <typename Tm>
      size_t get_chunksize(){
         return tools::is_complex<Tm>()? (1ULL<<26) : (1ULL<<30); // based on some tests; int 2^31-1
      }

   //--- broadcast ---
   // raw data
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, Tm* ptr, size_t size, int root){
         size_t chunksize = get_chunksize<Tm>();
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::broadcast(comm, ptr+offset, len, root); 
         }
      }

   // int2e: assuming int2e.data is large
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, integral::two_body<Tm>& int2e, int root){
         int rank = comm.rank();
         boost::mpi::broadcast(comm, int2e.sorb, root);
         if(rank != root) int2e.init_mem();
         boost::mpi::broadcast(comm, int2e.Q, root);
         broadcast(comm, int2e.data.data(), int2e.data.size(), root);
      }

   // stensor2
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::stensor2<Tm>& qt2, int root){
         boost::mpi::broadcast(comm, qt2.own, root);
         boost::mpi::broadcast(comm, qt2.info, root);
         int rank = comm.rank();
         if(rank != root && !qt2.own) qt2._data = new Tm[qt2.info._size];
         size_t chunksize = get_chunksize<Tm>();
         size_t size = qt2.info._size; 
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::broadcast(comm, qt2._data+offset, len, root); 
         }
      }

   // icomb: assuming the individual size of sites is small
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::comb0<Tm>& icomb, int root){
         /*
         boost::mpi::broadcast(comm, icomb.topo, root);
         boost::mpi::broadcast(comm, icomb.rbases, root);
         // sites could be packed in future: 
         // https://gist.github.com/hsidky/2f0e075095026d2ebda1
         for(int i=0; i<icomb.topo.ntotal; i++){
            boost::mpi::broadcast(comm, icomb.sites[i], root);
         }
         */
      }

   // icomb: assuming the individual size of sites is small
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::comb<Tm>& xxx, int root){
         /*
         boost::mpi::broadcast(comm, icomb.topo, root);
         boost::mpi::broadcast(comm, icomb.rbases, root);
         // sites could be packed in future: 
         // https://gist.github.com/hsidky/2f0e075095026d2ebda1
         for(int i=0; i<icomb.topo.ntotal; i++){
            boost::mpi::broadcast(comm, icomb.sites[i], root);
         }
         */
      }

   //--- reduce ---
   // raw data
   template <typename Tm, typename Op>
      void reduce(const boost::mpi::communicator & comm, const Tm* ptr_in, size_t size, 
            Tm* ptr_out, Op op, int root){
         size_t chunksize = get_chunksize<Tm>();
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::reduce(comm, ptr_in+offset, len, ptr_out+offset, op, root); 
         }
      }


} // ctns

#endif

#endif
