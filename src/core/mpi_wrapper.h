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
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, Tm* ptr, 
            const size_t size, const int root){
         size_t chunksize = get_chunksize<Tm>();
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::broadcast(comm, ptr+offset, len, root); 
         }
      }

   //--- in-place reduce ---
   template <typename Tm>
      void reduce(const boost::mpi::communicator & comm, Tm* ptr_inout, 
            const size_t size, const int root){ 
         size_t chunksize = get_chunksize<Tm>();
         if(!tools::is_complex<Tm>()){
            for(size_t offset=0; offset<size; offset+=chunksize){
               size_t len = std::min(chunksize, size-offset);
               if(comm.rank() == root){
                  // send_data, recv_data, count, datatype, op, root, communicator
                  MPI_Reduce(MPI_IN_PLACE, ptr_inout+offset, len, MPI_DOUBLE, MPI_SUM, root, comm);
               }else{
                  MPI_Reduce(ptr_inout+offset, nullptr, len, MPI_DOUBLE, MPI_SUM, root, comm);
               } 
            }
         }else{
            for(size_t offset=0; offset<size; offset+=chunksize){
               size_t len = std::min(chunksize, size-offset);
               if(comm.rank() == root){
                  MPI_Reduce(MPI_IN_PLACE, ptr_inout+offset, len, MPI_DOUBLE_COMPLEX, MPI_SUM, root, comm);
               }else{
                  MPI_Reduce(ptr_inout+offset, nullptr, len, MPI_DOUBLE_COMPLEX, MPI_SUM, root, comm);
               } 
            }
         }
      }

} // ctns

#endif

#endif
