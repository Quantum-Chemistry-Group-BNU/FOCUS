#ifndef SERIAL

#ifndef MPI_WRAPPER_H
#define MPI_WRAPPER_H

#include <boost/mpi.hpp>
#include "tools.h"

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
