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
            // ordered by cost and then lexicographic ordering of {M,N}
            std::pair<int,int> get_dim2() const{ return std::make_pair(M,N); }
            bool operator >(const MVinfo<Tm>& mv) const{
               size_t mn = size_t(M)*size_t(N);
               size_t mn2 = size_t(mv.M)*size_t(mv.N);
               return (mn>mn2) || (mn==mn2 && this->get_dim2()>mv.get_dim2());
            }
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
            // save dimension for optimization
            void save(const std::string fname) const{
               std::ofstream fout(fname);
               if(size > 0){
                  // total batch size
                  fout << size << " " << transA[0] << " " << std::endl;
                  // group information
                  fout << gsta.size();
                  for(int i=0; i<gsta.size(); i++){
                     fout << " " << gsta[i];
                  }
                  fout << std::endl;
                  // (M,N)
                  for(int i=0; i<size; i++){
                     fout << M[i] << " " << N[i] << std::endl;
                  }
               }else{
                  fout << "empty" << std::endl;
               }
               fout.close();
            }
            void kernel(const int batchgemv, Tm** ptrs){
               if(batchgemv == 0){
                  this->xgemv_omp(ptrs);   
               }else if(batchgemv == 1){
                  this->xgemv_batch_cpu(ptrs);   
#ifdef GPU 
               }else if(batchgemv == 2){
                  this->xgemv_batch_gpu_magma(ptrs);
#ifndef HIP
               }else if(batchgemv == 3){
                  this->xgemv_batch_gpu_grouped(ptrs);
               }else if(batchgemv == 4){
                  this->xgemv_batch_gpu_stream(ptrs);
#endif
#endif 
               }else{
                  std::cout << "error: no such option in MVbatch::kernel batchgemv=" << batchgemv << std::endl;
                  exit(1);
               }
            }
            void xgemv_omp(Tm** ptrs);
            void xgemv_batch_cpu(Tm** ptrs);
#ifdef GPU
            void xgemv_batch_gpu_magma(Tm** ptrs);
#ifndef HIP
            void xgemv_batch_gpu_grouped(Tm** ptrs);
            void xgemv_batch_gpu_stream(Tm** ptrs);
#endif
#endif
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
            std::vector<int> gsta; // groups
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
         std::pair<int,int> dim2;
         for(size_t j=0; j<MVlst.size(); j++){
            const auto& mv = MVlst[j];
            if(mv.M*mv.N == 0) continue;
            // grouping
            if(i == 0){
               dim2 = mv.get_dim2();
               gsta.push_back(0);
            }else{
               auto dim2new = mv.get_dim2();
               if(dim2new != dim2){
                  dim2 = dim2new;
                  gsta.push_back(i);
               }
            }
            cost += mv.cost();
            transA[i] = mv.transA; 
            M[i] = mv.M; N[i] = mv.N;
            LDA[i] = mv.LDA;
            locA[i] = mv.locA; locx[i] = mv.locx; locy[i] = mv.locy;
            offA[i] = mv.offA; offx[i] = mv.offx; offy[i] = mv.offy;
            i += 1;
         }
         gsta.push_back(size);
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
      void MVbatch<Tm>::xgemv_batch_gpu_magma(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            xptr[i] = ptrs[locx[i]] + offx[i];
            yptr[i] = ptrs[locy[i]] + offy[i];
         }
         if(size > 0){
            linalg::xgemv_batch_gpu_magma(transA[0], M.data(), N.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), xptr.data(), INCX.data(), beta_vec.data(),
                  yptr.data(), INCY.data(), size);
         }
      }

#ifndef HIP
   template <typename Tm>
      void MVbatch<Tm>::xgemv_batch_gpu_grouped(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            xptr[i] = ptrs[locx[i]] + offx[i];
            yptr[i] = ptrs[locy[i]] + offy[i];
         }
         if(size > 0){
            linalg::xgemv_batch_gpu_grouped(transA[0], M.data(), N.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), xptr.data(), INCX.data(), beta_vec.data(),
                  yptr.data(), INCY.data(), size, gsta);
         }
      }

   template <typename Tm>
      void MVbatch<Tm>::xgemv_batch_gpu_stream(Tm** ptrs){
         // initialization 
         for(size_t i=0; i<size; i++){
            Aptr[i] = ptrs[locA[i]] + offA[i];
            xptr[i] = ptrs[locx[i]] + offx[i];
            yptr[i] = ptrs[locy[i]] + offy[i];
         }
         if(size > 0){
            linalg::xgemv_batch_gpu_stream(transA[0], M.data(), N.data(), alpha_vec.data(), 
                  Aptr.data(), LDA.data(), xptr.data(), INCX.data(), beta_vec.data(),
                  yptr.data(), INCY.data(), size, gsta);
         }
      }
#endif
#endif

} // ctns

#endif
