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
         // remove zero dimensions
         size = 0;
         for(size_t i=0; i<MVlst.size(); i++){
            const auto& mv = MVlst[i];
            if(mv.M*mv.N == 0) continue;
            size += 1;
         }
         transA.resize(size);
         M.resize(size); N.resize(size);
         LDA.resize(size); INCX.resize(size,1), INCY.resize(size,1); // by default = 1
         locA.resize(size); locx.resize(size); locy.resize(size);
         offA.resize(size); offx.resize(size); offy.resize(size);
         cost = 0.0;
         size_t i = 0;
         for(size_t j=0; j<MVlst.size(); j++){
            const auto& mv = MVlst[j];
            if(mv.M*mv.N == 0) continue;
            cost += mv.cost();
            transA[i] = mv.transA; 
            M[i] = mv.M; N[i] = mv.N;
            LDA[i] = mv.LDA;
            locA[i] = mv.locA; locx[i] = mv.locx; locy[i] = mv.locy;
            offA[i] = mv.offA; offx[i] = mv.offx; offy[i] = mv.offy;
            i += 1;
         }
         Aptr.resize(size); xptr.resize(size); yptr.resize(size);
         alpha_vec.resize(size,1.0);
         beta_vec.resize(size,beta); // beta is allowed
         size_per_group_vec.resize(size,1);
      }

   template <typename Tm>
      void MVbatch<Tm>::xgemv_omp(Tm** ptrs){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(size_t i=0; i<size; i++){
            Tm* aptr = ptrs[locA[i]] + offA[i];
            Tm* xptr = ptrs[locx[i]] + offx[i];
            Tm* yptr = ptrs[locy[i]] + offy[i];
            linalg::xgemv(&transA[i], M[i], N[i], alpha_vec[i],
                  aptr, LDA[i], xptr, INCX[i], beta_vec[i],
                  yptr, INCY[i]);
            /*
            // debug
            std::cout << "M,N,LDA=" << M[i] << "," << N[i] << "," << LDA[i] << std::endl;
            linalg::matrix<Tm> amat(M[i],N[i],aptr);
            amat.print("amat");
            for(int j=0; j<N[i]; j++){
               std::cout << "lzd x: j=" << j << " x=" << xptr[j] << std::endl;
            }
            for(int j=0; j<M[i]; j++){
               std::cout << "lzd y: j=" << j << " y=" << yptr[j] << std::endl;
            }
            */
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
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            xptr[i] = ptrs[locx[i]] + offx[i];
            yptr[i] = ptrs[locy[i]] + offy[i];
         }
         if(size > 0){
            linalg::xgemv_batch_gpu(transA[0], M.data(), N.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), xptr.data(), INCX.data(), beta_vec.data(),
                  yptr.data(), INCY.data(), size);
         }
      }
#endif

} // ctns

#endif
