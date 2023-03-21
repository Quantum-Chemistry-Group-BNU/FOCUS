#ifndef PREPROCESS_MVBATCH_H
#define PREPROCESS_MVBATCH_H

#include "../core/blas_batch.h"
#ifdef GPU
#include "../gpu/gpu_blas_batch.h"
#endif

namespace ctns{

   // Matrix-vector operations: interface to XGEMV
   template <typename Tm>
      struct MVinfo{
         public:
            double cost() const{ return 2*double(M)*N; }
         public:
            char transA;
            int M, N, LDA;
            int locA, locx, locy;
            size_t offA, offx, offy;
      };
   template <typename Tm>
      using MVlist = std::vector<MVinfo<Tm>>;  

   // Matrix-vector operations: interface to XGEMV_BATCH
   template <typename Tm>
      struct MVbatch{
         public:
            void init(const MVlist<Tm>& MVlst, const Tm beta=0.0);
            void kernel(const int batchblas, Tm** ptrs){
               if(batchblas == 0){
                  this->xgemv_omp(ptrs);   
               }else if(batchblas == 1){
                  this->xgemv_batch_cpu(ptrs);   
#ifdef GPU 
               }else if(batchblas == 2){
                  this->xgemv_batch_gpu(ptrs);
#endif 
               }else{
                  std::cout << "error: no such option in MVbatch::kernel batchblas=" << batchblas << std::endl;
                  exit(1);
               }
            }
            void xgemv_omp(Tm** ptrs);
            void xgemv_batch_cpu(Tm** ptrs);
#ifdef GPU
            void xgemv_batch_gpu(Tm** ptrs);
#endif
            // save dimension for optimization
            void save(const std::string fname) const{
               std::ofstream fout(fname);
               fout << size << " " << transA[0] << " " << std::endl;
               for(int i=0; i<size; i++){
                  fout << M[i] << " " << N[i] << std::endl;
               }
               fout.close();
            }
         public:
            size_t size = 0;
            double cost = 0.0;
            std::vector<char> transA;
            std::vector<MKL_INT> M, N, LDA, INCX, INCY;
            std::vector<MKL_INT> locA, locx, locy;
            std::vector<size_t> offA, offx, offy;
            std::vector<const Tm*> Aptr, xptr;
            std::vector<Tm*> yptr;
            std::vector<Tm> alpha_vec, beta_vec;
            std::vector<MKL_INT> size_per_group_vec;
      };

   template <typename Tm>
      void MVbatch<Tm>::init(const MVlist<Tm>& MVlst, const Tm beta){
         size = MVlst.size();
         transA.resize(size);
         M.resize(size); N.resize(size);
         LDA.resize(size); INCX.resize(size,1), INCY.resize(size,1); // by default = 1
         locA.resize(size); locx.resize(size); locy.resize(size);
         offA.resize(size); offx.resize(size); offy.resize(size);
         cost = 0.0;
         for(size_t i=0; i<size; i++){
            const auto& mv = MVlst[i];
            cost += mv.cost();
            transA[i] = mv.transA; 
            M[i] = mv.M; N[i] = mv.N;
            LDA[i] = mv.LDA;
            locA[i] = mv.locA; locx[i] = mv.locx; locy[i] = mv.locy;
            offA[i] = mv.offA; offx[i] = mv.offx; offy[i] = mv.offy;
         }
         Aptr.resize(size); xptr.resize(size); yptr.resize(size);
         alpha_vec.resize(size,1.0);
         beta_vec.resize(size,beta); // beta is allowed
         size_per_group_vec.resize(size,1);
      }

   template <typename Tm>
      void MVbatch<Tm>::xgemv_omp(Tm** ptrs){
         const Tm alpha = 1.0, beta = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(size_t i=0; i<size; i++){
            Tm* aptr = ptrs[locA[i]] + offA[i];
            Tm* xptr = ptrs[locx[i]] + offx[i];
            Tm* yptr = ptrs[locy[i]] + offy[i];
            linalg::xgemv(&transA[i], M[i], N[i], alpha,
                  aptr, LDA[i], xptr, INCX[i], beta,
                  yptr, INCY[i]);
         } // i
      }

   template <typename Tm>
      void MVbatch<Tm>::xgemv_batch_cpu(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            xptr[i] = ptrs[locx[i]] + offx[i];
            yptr[i] = ptrs[locy[i]] + offy[i];
         }
         if(size > 0){ 
            MKL_INT group_count = size;
            linalg::xgemv_batch(transA.data(), M.data(), N.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), xptr.data(), INCX.data(), beta_vec.data(),
                  yptr.data(), INCY.data(), &group_count, size_per_group_vec.data());
         }
      }

#ifdef GPU
   template <typename Tm>
      void MVbatch<Tm>::xgemv_batch_gpu(Tm** ptrs){
         std::cout << "xgemv_batch_gpu is NOT IMPLEMENTED YET!" << std::endl;
         exit(1);
/*
         int a_total=0;
         int b_total=0;
         int c_total=0;
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            Bptr[i] = ptrs[locB[i]] + offB[i];
            Cptr[i] = ptrs[locC[i]] + offC[i];
            a_total +=M[i]*K[i];
            b_total +=K[i]*N[i];
            c_total +=M[i]*N[i];
         }
         if(size > 0){
            linalg::xgemm_batch_gpu(transA[0], transB[0], M.data(), N.data(), K.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
                  Cptr.data(), M.data(), size, a_total, b_total, c_total);
         }
*/
      }
#endif

} // ctns

#endif
