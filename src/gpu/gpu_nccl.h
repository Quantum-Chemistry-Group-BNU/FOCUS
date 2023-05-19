#ifdef GPU

#ifdef NCCL

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#define MPICHECK(cmd) \
   do {                          \
      int e = cmd;                                      \
      if( e != MPI_SUCCESS ) {                          \
         printf("Failed: MPI error %s:%d '%d'\n",        \
               __FILE__,__LINE__, e);   \
         exit(EXIT_FAILURE);                             \
      }                                                 \
   } while(0)

#define CUDACHECK(cmd) \
   do {                         \
      cudaError_t e = cmd;                              \
      if( e != cudaSuccess ) {                          \
         printf("Failed: Cuda error %s:%d '%s'\n",             \
               __FILE__,__LINE__,cudaGetErrorString(e));   \
         exit(EXIT_FAILURE);                             \
      }                                                 \
   } while(0)

#define NCCLCHECK(cmd) \
do {                         \
ncclResult_t r = cmd;                             \
if (r!= ncclSuccess) {                            \
printf("Failed, NCCL error %s:%d '%s'\n",             \
__FILE__,__LINE__,ncclGetErrorString(r));   \
exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

// adapted from
// https://docs.nvidia.com/deeplearning/nccl/archives/nccl_21210/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread
struct nccl_communicator{
   public:
      void init(){
         int myRank, nRanks;
         MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
         MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
         if(myRank == 0) ncclGetUniqueId(&id);
         MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
         CUDACHECK(cudaStreamCreate(&s));
         //initializing NCCL
         NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
      }
      template <typename Tm>
         void reduce(Tm* ptr, const size_t size, const int root){
            // regarded as double?
            NCCLCHECK(ncclReduce((const void*)ptr, (void*)ptr, 
                     size*sizeof(Tm)/sizeof(ncclDouble),
                     ncclDouble, ncclSum, root, comm, s));
            //completing NCCL operation by synchronizing on the CUDA stream
            CUDACHECK(cudaStreamSynchronize(s));
         }
      template <typename Tm>
         void broadcast(Tm* ptr, const size_t size, const int root){
            NCCLCHECK(ncclBcast((void*)ptr, size*sizeof(Tm)/sizeof(ncclDouble), 
                     ncclDouble, root, comm, s));
            //completing NCCL operation by synchronizing on the CUDA stream
            CUDACHECK(cudaStreamSynchronize(s));
         }
      void finalize(){
         ncclCommDestroy(comm);
         cudaStreamDestroy(s);
      }
   public:
      ncclUniqueId id;
      ncclComm_t comm;
      cudaStream_t s;
};

#endif

#endif
