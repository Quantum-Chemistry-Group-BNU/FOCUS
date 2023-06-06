#ifndef PREPROCESS_MMBATCH_H
#define PREPROCESS_MMBATCH_H

#include "../core/blas_batch.h"
#ifdef GPU
#include "../gpu/gpu_blas_batch.h"
#endif

namespace ctns{

   // Matrix-matrix operations: interface to XGEMM
   template <typename Tm>
      struct MMinfo{
         public:
            double cost() const{ return 2*double(M)*N*K; }
            // ordered by cost and then lexicographic ordering of {M,N,K}
            std::tuple<int,int,int,int,int> get_dims() const{ return std::make_tuple(M,N,K,LDA,LDB); }
            bool operator >(const MMinfo<Tm>& mm) const{
               size_t mnk = size_t(M)*size_t(N)*size_t(K);
               size_t mnk2 = size_t(mm.M)*size_t(mm.N)*size_t(mm.K);
               return (mnk>mnk2) || (mnk==mnk2 && this->get_dims()>mm.get_dims());
            }
         public:
            char transA, transB;
            int M, N, K, LDA, LDB;
            int locA, locB, locC;
            size_t offA, offB, offC;
      };
   template <typename Tm>
      using MMlist = std::vector<MMinfo<Tm>>;  
   template <typename Tm>
      using MMlist2 = std::vector<std::vector<MMinfo<Tm>>>;

   // Matrix-matrix operations: interface to XGEMM_BATCH
   template <typename Tm>
      struct MMbatch{
         public:
            void init(const MMlist<Tm>& MMlst);
            // save dimension for optimization
            void save(const std::string fname) const{
               std::ofstream fout(fname);
               if(size > 0){
                  // total batch size
                  fout << size << " " << transA[0] << " " << transB[0] <<std::endl;
                  // group information
                  fout << gsta.size();
                  for(int i=0; i<gsta.size(); i++){
                     fout << " " << gsta[i]; 
                  }
                  fout << std::endl;
                  // (M,N,K,LDA,LDB)
                  for(int i=0; i<size; i++){
                     fout << M[i] << " " << N[i] << " " << K[i] 
                        << " " << LDA[i] << " " << LDB[i] 
                        << std::endl;
                  }
               }else{
                  fout << "empty" << std::endl;
               }
               fout.close();
            }
            void kernel(const int batchgemm, Tm** ptrs){
               if(batchgemm == 0){
                  this->xgemm_omp(ptrs);   
               }else if(batchgemm == 1){
                  this->xgemm_batch_cpu(ptrs);   
#ifdef GPU 
               }else if(batchgemm == 2){
                  this->xgemm_batch_gpu_magma(ptrs);
#ifndef USE_HIP
               }else if(batchgemm == 3){
                  this->xgemm_batch_gpu_grouped(ptrs);
               }else if(batchgemm == 4){
                  this->xgemm_batch_gpu_stream(ptrs);
#endif
#endif 
               }else{
                  std::cout << "error: no such option in MMbatch::kernel batchgemm=" << batchgemm << std::endl;
                  exit(1);
               }
            }
            void xgemm_omp(Tm** ptrs);
            void xgemm_batch_cpu(Tm** ptrs);
#ifdef GPU
            void xgemm_batch_gpu_magma(Tm** ptrs);
#ifndef USE_HIP
            void xgemm_batch_gpu_grouped(Tm** ptrs);
            void xgemm_batch_gpu_stream(Tm** ptrs);
#endif
#endif
         public:
            size_t size = 0;
            double cost = 0.0;
            std::vector<char> transA, transB;
            std::vector<MKL_INT> M, N, K, LDA, LDB;
            std::vector<MKL_INT> locA, locB, locC;
            std::vector<size_t> offA, offB, offC;
            std::vector<const Tm*> Aptr, Bptr;
            std::vector<Tm*> Cptr;
            std::vector<Tm> alpha_vec, beta_vec;
            std::vector<MKL_INT> size_per_group_vec;
            std::vector<int> gsta; // groups
      };

   template <typename Tm>
      void MMbatch<Tm>::init(const MMlist<Tm>& MMlst){
         // remove zero dimensions
         size = 0;
         for(size_t i=0; i<MMlst.size(); i++){
            const auto& mm = MMlst[i];
            if(mm.M*mm.N*mm.K == 0) continue;
            size += 1; 
         } 
         transA.resize(size); transB.resize(size);
         M.resize(size); N.resize(size); K.resize(size);
         LDA.resize(size); LDB.resize(size);
         locA.resize(size); locB.resize(size); locC.resize(size);
         offA.resize(size); offB.resize(size); offC.resize(size);
         cost = 0.0;
         size_t i = 0;
         std::tuple<int,int,int,int,int> dims;
         for(size_t j=0; j<MMlst.size(); j++){
            const auto& mm = MMlst[i];
            if(mm.M*mm.N*mm.K == 0) continue;
            // grouping
            if(i == 0){
               dims = mm.get_dims();
               gsta.push_back(0);
            }else{
               auto dimsnew = mm.get_dims();
               if(dimsnew != dims){
                  dims = dimsnew;
                  gsta.push_back(i);
               }
            }
            cost += mm.cost();
            transA[i] = mm.transA; transB[i] = mm.transB;
            M[i] = mm.M; N[i] = mm.N; K[i] = mm.K;
            LDA[i] = mm.LDA; LDB[i] = mm.LDB;
            locA[i] = mm.locA; locB[i] = mm.locB; locC[i] = mm.locC;
            offA[i] = mm.offA; offB[i] = mm.offB; offC[i] = mm.offC;
            i += 1; 
         }
         gsta.push_back(size);
         Aptr.resize(size); Bptr.resize(size); Cptr.resize(size);
         alpha_vec.resize(size,1.0);
         beta_vec.resize(size,0.0);
         size_per_group_vec.resize(size,1);
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_omp(Tm** ptrs){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(size_t i=0; i<size; i++){
            Tm* aptr = ptrs[locA[i]] + offA[i];
            Tm* bptr = ptrs[locB[i]] + offB[i];
            Tm* cptr = ptrs[locC[i]] + offC[i];
            linalg::xgemm(&transA[i], &transB[i], M[i], N[i], K[i], alpha_vec[i],
                  aptr, LDA[i], bptr, LDB[i], beta_vec[i],
                  cptr, M[i]);
         } // i
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_cpu(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         if(size > 0){ 
            MKL_INT group_count = size;
            linalg::xgemm_batch(transA.data(), transB.data(), M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), &group_count, size_per_group_vec.data());
         }
      }

#ifdef GPU
   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_gpu_magma(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu_magma(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size);
         }
      }

#ifndef USE_HIP
   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_gpu_grouped(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu_grouped(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size, gsta);
         }
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_gpu_stream(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu_stream(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size, gsta);
         }
      }

   template <typename Tm>
      void xgemm_batch_gpu_merged(MMbatch<Tm>& batch1, MMbatch<Tm>& batch2, Tm** ptrs){
         std::cout << "error: xgemm_batch_gpu_merged is not implemented yet!" << std::endl;
         exit(1);
      }

   template <>
      inline void xgemm_batch_gpu_merged(MMbatch<double>& batch1, MMbatch<double>& batch2, double** ptrs){
         // initialization
         size_t size1 = batch1.size;
         size_t size2 = batch2.size;
         if(size1 == 0 && size2 == 0) return;
         for(size_t i=0; i<size1; i++){
            batch1.Aptr[i] = ptrs[batch1.locA[i]] + batch1.offA[i];
            batch1.Bptr[i] = ptrs[batch1.locB[i]] + batch1.offB[i];
            batch1.Cptr[i] = ptrs[batch1.locC[i]] + batch1.offC[i];
         }
         // initialization 
         for(size_t i=0; i<size2; i++){
            batch2.Aptr[i] = ptrs[batch2.locA[i]] + batch2.offA[i];
            batch2.Bptr[i] = ptrs[batch2.locB[i]] + batch2.offB[i];
            batch2.Cptr[i] = ptrs[batch2.locC[i]] + batch2.offC[i];
         }
         size_t total_dsize = 3*(size1+size2)*sizeof(double*);
         void* dev_dtotal = GPUmem.allocate(total_dsize);
         double** dev_a_array1 = (double**)dev_dtotal;
         double** dev_b_array1 = dev_a_array1 + size1;
         double** dev_c_array1 = dev_b_array1 + size1;
         double** dev_a_array2 = dev_c_array1 + size1;
         double** dev_b_array2 = dev_a_array2 + size2;
         double** dev_c_array2 = dev_b_array2 + size2;
         GPUmem.to_gpu(dev_a_array1, batch1.Aptr.data(), size1*sizeof(double*));
         GPUmem.to_gpu(dev_b_array1, batch1.Bptr.data(), size1*sizeof(double*));
         GPUmem.to_gpu(dev_c_array1, batch1.Cptr.data(), size1*sizeof(double*));
         GPUmem.to_gpu(dev_a_array2, batch2.Aptr.data(), size2*sizeof(double*));
         GPUmem.to_gpu(dev_b_array2, batch2.Bptr.data(), size2*sizeof(double*));
         GPUmem.to_gpu(dev_c_array2, batch2.Cptr.data(), size2*sizeof(double*));
         
         size_t gsize1 = batch1.gsta.size()-1;
         size_t gsize2 = batch2.gsta.size()-1;
         size_t gsize = gsize1+gsize2; 
         int ntimes = (gsize+NSTREAMS-1)/NSTREAMS; 
         for(int k=0; k<ntimes; k++){
            size_t off = k*NSTREAMS;
            size_t jlen = std::min(gsize-off, size_t(NSTREAMS));
         
            for(int j=0; j<jlen; j++){
               size_t jdx = off+j;
               CUBLAS_CHECK(cublasSetStream(handle_cublas, stream[j])); 
               
               const auto& batch = (jdx<gsize1)? batch1 : batch2;
               double** dev_a_array = (jdx<gsize1)? dev_a_array1 : dev_a_array2;
               double** dev_b_array = (jdx<gsize1)? dev_b_array1 : dev_b_array2;
               double** dev_c_array = (jdx<gsize1)? dev_c_array1 : dev_c_array2;
               int i = (jdx<gsize1)? jdx : jdx-gsize1;
               
               int ista = batch.gsta[i];
               int nbatch = batch.gsta[i+1]-ista;
               // convert from magma_int_t to int 
               int m = batch.M[ista], n = batch.N[ista], k = batch.K[ista];
               int lda = batch.LDA[ista], ldb = batch.LDB[ista], ldc = batch.M[ista]; 
               const char transa = batch.transA[0];
               const char transb = batch.transB[0];
               cublasOperation_t transA = CUBLAS_OP_N ;
               if(transa=='T' || transa=='C'){
                  transA = CUBLAS_OP_T;
               }
               cublasOperation_t transB = CUBLAS_OP_N ;
               if(transb=='T' || transb=='C'){
                  transB = CUBLAS_OP_T;
               }
               const double* alpha = batch.alpha_vec.data();
               const double* beta = batch.beta_vec.data();
               // https://docs.nvidia.com/cuda/cublas/index.html
               CUBLAS_CHECK(cublasDgemmBatched(handle_cublas,
                                  transA, transB,
                                  m, n, k,
                                  alpha,
                                  &dev_a_array[ista], lda, // pointer should be on device
                                  &dev_b_array[ista], ldb,
                                  beta,
                                  &dev_c_array[ista], ldc,
                                  nbatch));
            } // j

            for(int j=0; j<jlen; j++){
               CUDA_CHECK(cudaStreamSynchronize(stream[j]));
            }
         }

         GPUmem.deallocate(dev_dtotal, total_dsize);
      }

#endif
#endif

} // ctns

#endif
