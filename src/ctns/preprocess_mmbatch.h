#ifndef PREPROCESS_MMBATCH_H
#define PREPROCESS_MMBATCH_H

#include <time.h>
#include <sys/time.h>
#include "blas_batch.h"
#include "blas_batch_gpu.h"

namespace ctns{

   // Matrix-matrix operations: interface to XGEMM_BATCH
   template <typename Tm>
      struct MMbatch{
         public:
            void init(const MMlist<Tm>& MMlst);
            void kernel(const int batchgemm, Tm** ptrs){
               if(batchgemm == 0){
                  this->xgemm_omp(ptrs);   
               }else if(batchgemm == 1){
                  this->xgemm_batch_cpu(ptrs);   
#ifdef GPU 
               }else if(batchgemm == 2){
                  this->xgemm_batch_gpu(ptrs);    
               }else if(batchgemm == 3){
                  this->xgemm_batch_gpu_precopy(ptrs);
#endif 
               }else{
                  std::cout << "error: no such option in MMbatch::kernel batchgemm=" << batchgemm << std::endl;
                  exit(1);
               }
            }
            void xgemm_omp(Tm** ptrs);
            void xgemm_batch_cpu(Tm** ptrs);
#ifdef GPU
            void xgemm_batch_gpu(Tm** ptrs);
            void xgemm_batch_gpu_precopy(Tm** ptrs);
#endif
            void save(const std::string fname){
               std::ofstream fout(fname);
               fout << size << " " << transA[0] << " " << transB[0] << " " << std::endl;
               for(int i=0; i<size; i++){
                  fout << M[i] << " " 
                       << N[i] << " "
                       << K[i] 
                       << std::endl;
               }
               fout.close();
            }
         public:
            size_t size = 0;
            double cost = 0.0;
            std::vector<char> transA, transB;
            std::vector<int> M, N, K, LDA, LDB;
            std::vector<int> locA, locB, locC;
            std::vector<size_t> offA, offB, offC;
            std::vector<const Tm*> Aptr, Bptr;
            std::vector<Tm*> Cptr;
            std::vector<Tm> alpha_vec, beta_vec;
            std::vector<int> size_per_group_vec;
      };

   template <typename Tm>
      void MMbatch<Tm>::init(const MMlist<Tm>& MMlst){
         size = MMlst.size();
         transA.resize(size); transB.resize(size);
         M.resize(size); N.resize(size); K.resize(size);
         LDA.resize(size); LDB.resize(size);
         locA.resize(size); locB.resize(size); locC.resize(size);
         offA.resize(size); offB.resize(size); offC.resize(size);
         cost = 0.0;
         for(int i=0; i<size; i++){
            const auto& mm = MMlst[i];
            cost += mm.cost();
            transA[i] = mm.transA; transB[i] = mm.transB;
            M[i] = mm.M; N[i] = mm.N; K[i] = mm.K;
            LDA[i] = mm.LDA; LDB[i] = mm.LDB;
            locA[i] = mm.locA; locB[i] = mm.locB; locC[i] = mm.locC;
            offA[i] = mm.offA; offB[i] = mm.offB; offC[i] = mm.offC; 
         }
         Aptr.resize(size); Bptr.resize(size); Cptr.resize(size);
         alpha_vec.resize(size,1.0);
         beta_vec.resize(size,0.0);
         size_per_group_vec.resize(size,1);
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_omp(Tm** ptrs){
         const Tm alpha = 1.0, beta = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int i=0; i<size; i++){
            Tm* aptr = ptrs[locA[i]] + offA[i];
            Tm* bptr = ptrs[locB[i]] + offB[i];
            Tm* cptr = ptrs[locC[i]] + offC[i];
            linalg::xgemm(&transA[i], &transB[i], &M[i], &N[i], &K[i], &alpha,
                  aptr, &LDA[i], bptr, &LDB[i], &beta,
                  cptr, &M[i]);
         } // i
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_cpu(Tm** ptrs){
         // initialization 
         for(int i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
         }
         int group_count = size;
         std::cout << "lzd group_count = " << group_count << std::endl;
         if(size > 0){ 
            linalg::xgemm_batch(transA.data(), transB.data(), M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), &group_count, size_per_group_vec.data());
         }
      }

#ifdef GPU
   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_gpu(Tm** ptrs){
         //   std::cout<<"xgemm_batch_gpu"<<std::endl;
         int a_total=0;
         int b_total=0;
         int c_total=0;
         // initialization 
         for(int i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
            a_total +=M[i]*K[i];
            b_total +=K[i]*N[i];
            c_total +=M[i]*N[i];
         }
         if(size >0){
            linalg::xgemm_batch_gpu(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size, a_total, b_total, c_total);
         }
      }

   template <typename Tm>
      void MMbatch<Tm>::xgemm_batch_gpu_precopy(Tm** ptrs){
         int a_total=0;
         int b_total=0;
         int c_total=0;
         // initialization 
         for(int i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
            a_total +=M[i]*K[i];
            b_total +=K[i]*N[i];
            c_total +=M[i]*N[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu_precopy(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size, a_total, b_total, c_total);
         }
      }
#endif

} // ctns

#endif
