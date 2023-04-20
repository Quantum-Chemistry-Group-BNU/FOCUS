#ifndef PREPROCESS_HEADER_H
#define PREPROCESS_HEADER_H

namespace ctns{

   const int locInter = 4; // intermediates
   const int locIn    = 5; // x
   const int locOut   = 6; // y
   extern const int locInter;
   extern const int locIn;
   extern const int locOut;

   //
   // Determine batchsize dynamically:
   // total = dvdson + sizeof(Tm)*N*(blksize*2+blksize0) + math [GEMM_BATCH & GEMV_BATCH]
   // GEMM_BATCH = [6*(N+1)*sizeof(int) + 3*N*sizeof(double*/complex*)] (gpu_blas_batch.h)
   //            = 6*8*(N+1) + 3*N*8 = 72*N+48 [size of pointer is 8]
   // GEMV_BATCH = [5*(N+1)*sizeof(int) + 3*N*sizeof(double*/complex*)]
   //            = 5*8*(N+1) + 3*N*8 = 64*N+40
   // Coefficient in reduction: N*sizeof(double/complex) => N*sizeof(Tm)
   // Thus, total = dvdson + sizeof(Tm)*N*BLKSIZE+ 136*N+88
   // 
   // => N = (total-reserved)/(sizeof(Tm)*BLKSIZE+136) [BLKSIZE=2*blksize+blksize+1]
   //
#ifdef GPU
   template <typename Tm>
      void preprocess_gpu_batchsize(const double batchmem,
            const size_t blocksize,
            const size_t gpumem_other,
            const int rank,
            size_t& batchsize,
            size_t& gpumem_batch){
         size_t gpumem_avail = std::min(GPUmem.available(rank),size_t(batchmem*std::pow(1024,3)));
         size_t gpumem_reserved = gpumem_other + 88; 
         if(gpumem_avail > gpumem_reserved){
            batchsize = std::floor(double(gpumem_avail - gpumem_reserved)/(sizeof(Tm)*blocksize + 136));
            if(batchsize == 0){
               std::cout << "error: in sufficient GPU batch memory: batchsize=0!" << std::endl;
               exit(1);
            }
         }else{
            std::cout << "error: in sufficient GPU batch memory: avail<=reserved!" << std::endl;
            exit(1);
         }
         gpumem_batch = sizeof(Tm)*batchsize*blocksize;
      }
#endif

   template <typename Tm>
      void preprocess_cpu_batchsize(const double batchmem,
            const size_t blocksize,
            size_t& batchsize,
            size_t& cpumem_batch){
         assert(blocksize>0);
         batchsize = size_t(batchmem*std::pow(1024,3)/(sizeof(Tm)*blocksize));
         if(batchsize == 0){
            std::cout << "error: in sufficient CPU batch memory: batchsize=0!" << std::endl;
            exit(1);
         }
         cpumem_batch = batchsize*blocksize; // no sizeof(Tm) here for compatability
      }

} // ctns

#endif
