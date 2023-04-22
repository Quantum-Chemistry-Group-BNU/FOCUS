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
      void broadcast(const boost::mpi::communicator & comm, Tm* ptr, 
            const size_t size, const int root,
            const int alg_comm=0){
         size_t chunksize = get_chunksize<Tm>();
         if(alg_comm == 0){
            for(size_t offset=0; offset<size; offset+=chunksize){
               size_t len = std::min(chunksize, size-offset);
               boost::mpi::broadcast(comm, ptr+offset, len, root); 
            }
         }else if(alg_comm == 1){
            int nchunk = (size+chunksize-1)/chunksize;
            std::vector<boost::mpi::request> requests;
            if(comm.rank() == root){
               requests.resize((comm.size()-1)*nchunk);
               int idx = 0;
               for(int r=0; r<comm.size(); r++){
                  if(r == root) continue;
                  for(size_t offset=0; offset<size; offset+=chunksize){
                     size_t len = std::min(chunksize, size-offset);
                     // send from root to rank
                     int tag = r*1000 + int(offset/chunksize);
                     //requests[idx] = comm.isend(root, tag, ptr+offset, len);
                     comm.send(r, tag, ptr+offset, len);
                     idx += 1;
                  }
               }
            }else{
               requests.resize(nchunk);
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  int tag = comm.rank()*1000 + int(offset/chunksize);
                  int idx = offset/chunksize;
                  // receiv from rank-r to root
                  //requests[idx] = comm.irecv(root, tag, ptr+offset, len);
                  comm.recv(root, tag, ptr+offset, len);
               }
            }
            //boost::mpi::wait_all(requests.begin(), requests.end());

         }else if(alg_comm == 2){            
            int nchunk = (size+chunksize-1)/chunksize;
            std::vector<boost::mpi::request> requests;
            if(comm.rank() == root){
               requests.resize((comm.size()-1)*nchunk);
               int idx = 0;
               for(int r=0; r<comm.size(); r++){
                  if(r == root) continue;
                  for(size_t offset=0; offset<size; offset+=chunksize){
                     size_t len = std::min(chunksize, size-offset);
                     // send from root to rank
                     int tag = r*1000 + int(offset/chunksize);
                     requests[idx] = comm.isend(r, tag, ptr+offset, len);
                     idx += 1;
                  }
               }
            }else{
               requests.resize(nchunk);
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  int tag = comm.rank()*1000 + int(offset/chunksize);
                  int idx = offset/chunksize;
                  // receiv from rank-r to root
                  requests[idx] = comm.irecv(root, tag, ptr+offset, len);
               }
            }
            boost::mpi::wait_all(requests.begin(), requests.end());

         }else if(alg_comm == 3){
            int nchunk = (size+chunksize-1)/chunksize;
            std::vector<boost::mpi::request> requests;
            if(comm.rank() == root){
               requests.resize((comm.size()-1)*nchunk);
               int idx = 0;
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  for(int r=0; r<comm.size(); r++){
                     if(r == root) continue;
                     // send from root to rank
                     int tag = r*1000 + int(offset/chunksize);
                     //requests[idx] = comm.isend(root, tag, ptr+offset, len);
                     comm.send(r, tag, ptr+offset, len);
                     idx += 1;
                  }
               }
            }else{
               requests.resize(nchunk);
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  int tag = comm.rank()*1000 + int(offset/chunksize);
                  int idx = offset/chunksize;
                  // receiv from rank-r to root
                  //requests[idx] = comm.irecv(root, tag, ptr+offset, len);
                  comm.recv(root, tag, ptr+offset, len);
               }
            }
            //boost::mpi::wait_all(requests.begin(), requests.end());

         }else if(alg_comm == 4){            
            int nchunk = (size+chunksize-1)/chunksize;
            std::vector<boost::mpi::request> requests;
            if(comm.rank() == root){
               requests.resize((comm.size()-1)*nchunk);
               int idx = 0;
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  for(int r=0; r<comm.size(); r++){
                     if(r == root) continue;
                     // send from root to rank
                     int tag = r*1000 + int(offset/chunksize);
                     requests[idx] = comm.isend(r, tag, ptr+offset, len);
                     idx += 1;
                  }
               }
            }else{
               requests.resize(nchunk);
               for(size_t offset=0; offset<size; offset+=chunksize){
                  size_t len = std::min(chunksize, size-offset);
                  int tag = comm.rank()*1000 + int(offset/chunksize);
                  int idx = offset/chunksize;
                  // receiv from rank-r to root
                  requests[idx] = comm.irecv(root, tag, ptr+offset, len);
               }
            }
            boost::mpi::wait_all(requests.begin(), requests.end());

         } // alg_comm
      }

   //--- reduce ---
   // raw data
   template <typename Tm, typename Op>
      void reduce(const boost::mpi::communicator & comm, const Tm* ptr_in, const size_t size, 
            Tm* ptr_out, Op op, const int root, 
            const int alg_comm=0){
         size_t chunksize = get_chunksize<Tm>();
         //if(alg_comm == 0){
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::reduce(comm, ptr_in+offset, len, ptr_out+offset, op, root); 
         }
         //}else{

         //}
      }

} // ctns

#endif

#endif
