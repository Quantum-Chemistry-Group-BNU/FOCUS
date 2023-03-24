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
            void kernel(const int batchblas, Tm** ptrs){
               if(batchblas == 0){
                  this->xgemm_omp(ptrs);   
               }else if(batchblas == 1){
                  this->xgemm_batch_cpu(ptrs);   
#ifdef GPU 
               }else if(batchblas == 2){
                  this->xgemm_batch_gpu(ptrs);
#endif 
               }else{
                  std::cout << "error: no such option in MMbatch::kernel batchblas=" << batchblas << std::endl;
                  exit(1);
               }
            }
            void xgemm_omp(Tm** ptrs);
            void xgemm_batch_cpu(Tm** ptrs);
#ifdef GPU
            void xgemm_batch_gpu(Tm** ptrs);
#endif
            // save dimension for optimization
            void save(const std::string fname) const{
               std::ofstream fout(fname);
               fout << size << " " << transA[0] << " " << transB[0] << " " << std::endl;
               for(int i=0; i<size; i++){
                  fout << M[i] << " " << N[i] << " " << K[i] << std::endl;
               }
               fout.close();
            }
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
         for(size_t j=0; j<MMlst.size(); j++){
            const auto& mm = MMlst[i];
            if(mm.M*mm.N*mm.K == 0) continue;
            cost += mm.cost();
            transA[i] = mm.transA; transB[i] = mm.transB;
            M[i] = mm.M; N[i] = mm.N; K[i] = mm.K;
            LDA[i] = mm.LDA; LDB[i] = mm.LDB;
            locA[i] = mm.locA; locB[i] = mm.locB; locC[i] = mm.locC;
            offA[i] = mm.offA; offB[i] = mm.offB; offC[i] = mm.offC;
            i += 1; 
         }
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
      void MMbatch<Tm>::xgemm_batch_gpu(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size);
         }
      }
#endif

} // ctns

#endif
