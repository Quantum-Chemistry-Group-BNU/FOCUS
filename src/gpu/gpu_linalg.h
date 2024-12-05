#ifdef GPU

#ifndef GPU_LINALG_H
#define GPU_LINALG_H

#include "../core/matrix.h"
#include "gpu_env.h"
#include <cusolverDn.h>
#include "cusolver_utils.h"

namespace linalg{

   // Eig_solver and SVD_solver using suSOLVER: 

   // eigendecomposition HU=Ue: order=0/1 small-large/large-small
   template <typename Tm>
      void eig_solver_gpu(const matrix<Tm>& A, std::vector<double>& e, 
            matrix<Tm>& U, const MKL_INT order=0){

         assert(A.rows() == A.cols());  
         assert(A.rows() <= e.size()); // allow larger space used for e 
         U.resize(A.rows(), A.cols());
         U = (order == 0)? A : -A;

         using data_type = typename std::conditional<tools::is_complex<Tm>(), cuDoubleComplex, double>::type;

         // adapted from
         // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xsyevd/cusolver_Xsyevd_example.cu
         const int m = A.rows();
         const int lda = m;

         cusolverDnHandle_t cusolverH = NULL;
         cudaStream_t stream = NULL;
         cusolverDnParams_t params = NULL;

         data_type *d_A = nullptr;
         double *d_e = nullptr;
         int *d_info = nullptr;
         void *d_work = nullptr;              /* device workspace */
         void *h_work = nullptr;              /* host workspace for */
         size_t workspaceInBytesOnDevice = 0; /* size of workspace */
         size_t workspaceInBytesOnHost = 0;   /* size of workspace */
         int info = 0;

         // step 1: create cusolver handle, bind a stream 
         CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
         CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
         CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
         CUSOLVER_CHECK(cusolverDnCreateParams(&params));

         // step 2: copy A to gpu 
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_e), sizeof(double) * m));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

         CUDA_CHECK(cudaMemcpyAsync(d_A, U.data(), sizeof(data_type) * A.size(), 
                  cudaMemcpyHostToDevice, stream));

         // step 3: query working space of syevd
         cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
         cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

         CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
                  cusolverH, params, jobz, uplo, m, 
                  traits<data_type>::cuda_data_type, d_A, lda,
                  traits<double>::cuda_data_type, d_e, 
                  traits<data_type>::cuda_data_type, 
                  &workspaceInBytesOnDevice, 
                  &workspaceInBytesOnHost));

         // allocate work space
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

         if (0 < workspaceInBytesOnHost) {
            h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
            if (h_work == nullptr) {
               throw std::runtime_error("Error: h_work not allocated.");
            }
         }

         // step 4: compute spectrum
         CUSOLVER_CHECK(cusolverDnXsyevd(
                  cusolverH, params, jobz, uplo, m, 
                  traits<data_type>::cuda_data_type, d_A, lda,
                  traits<double>::cuda_data_type, d_e, 
                  traits<data_type>::cuda_data_type, d_work, 
                  workspaceInBytesOnDevice, h_work, 
                  workspaceInBytesOnHost, d_info));

         // step 5: copy back
         CUDA_CHECK(cudaMemcpyAsync(U.data(), d_A, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost,
                  stream));
         CUDA_CHECK(cudaMemcpyAsync(e.data(), d_e, sizeof(double) * e.size(), cudaMemcpyDeviceToHost,
                  stream));
         CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

         CUDA_CHECK(cudaStreamSynchronize(stream));

         /* free resources */
         CUDA_CHECK(cudaFree(d_A));
         CUDA_CHECK(cudaFree(d_e));
         CUDA_CHECK(cudaFree(d_info));
         CUDA_CHECK(cudaFree(d_work));
         free(h_work);

         CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
         CUDA_CHECK(cudaStreamDestroy(stream));

         if(order == 1){ transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
         if(info){
            std::cout << "eig_gpu[d] failed with info=" << info << std::endl;
            exit(1);
         }
      }

   // singular value decomposition: 
   template <typename Tm>
      void svd_solver_gpu(const matrix<Tm>& A, std::vector<double>& S, 
            matrix<Tm>& U, matrix<Tm>& VT, const MKL_INT iop=3){

         using data_type = typename std::conditional<tools::is_complex<Tm>(), cuDoubleComplex, double>::type;

         // adapted from
         // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgesvdp/cusolver_Xgesvdp_example.cu
         int m = A.rows();
         int n = A.cols();
         int r = std::min(m,n); 
         int lda = m;  // lda >= m
         int econ;
         int ldu, ldv;
         S.resize(r);
         linalg::matrix<Tm> V;
         if(iop == 10){
            econ = 0; U.resize(m,m); ldu = m; V.resize(n,n); ldv = n;
         }else if(iop == 13){                           
            econ = 1; U.resize(m,r); ldu = m; V.resize(n,r); ldv = n;
         }else{
            std::cout << "error: no such option in svd_solver_gpu!" << std::endl;
            exit(1);
         } 
         cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
         double h_err_sigma;

         cusolverDnHandle_t cusolverH = NULL;
         cublasHandle_t cublasH = NULL;
         cudaStream_t stream = NULL;
         cusolverDnParams_t params = NULL;

         data_type *d_A = nullptr;
         double *d_S = nullptr;  // singular values
         data_type *d_U = nullptr;  // left singular vectors
         data_type *d_V = nullptr; // right singular vectors
         int *d_info = nullptr;
         void *d_work = nullptr; // Device workspace
         void *h_work = nullptr; // Host workspace
         size_t workspaceInBytesOnDevice = 0;
         size_t workspaceInBytesOnHost = 0;
         int info = 0;
         
         /* step 1: create cusolver handle, bind a stream */
         CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
         CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
         CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
         CUSOLVER_CHECK(cusolverDnCreateParams(&params));

         /* step 2: copy A to device */
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * U.size()));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * V.size()));
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

         CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * lda * n, cudaMemcpyHostToDevice,
                  stream));

         /* step 3: query working space of SVD */
         CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
                  cusolverH, params, jobz, econ, m, n, 
                  traits<data_type>::cuda_data_type, d_A, lda,
                  traits<double>::cuda_data_type, d_S, 
                  traits<data_type>::cuda_data_type, d_U, ldu, 
                  traits<data_type>::cuda_data_type, d_V, ldv,
                  traits<data_type>::cuda_data_type,
                  &workspaceInBytesOnDevice,
                  &workspaceInBytesOnHost));

         // allocate work space
         CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

         if (0 < workspaceInBytesOnHost) {
            h_work = reinterpret_cast<data_type *>(malloc(workspaceInBytesOnHost));
            if (h_work == nullptr) {
               throw std::runtime_error("Error: h_work not allocated.");
            }
         }

         /* step 4: compute SVD */
         CUSOLVER_CHECK(cusolverDnXgesvdp(
                  cusolverH, params, jobz, econ, m, n, 
                  traits<data_type>::cuda_data_type, d_A, lda,
                  traits<double>::cuda_data_type, d_S, 
                  traits<data_type>::cuda_data_type, d_U, ldu,
                  traits<data_type>::cuda_data_type, d_V, ldv,
                  traits<data_type>::cuda_data_type, d_work,
                  workspaceInBytesOnDevice, h_work,
                  workspaceInBytesOnHost, d_info,
                  &h_err_sigma));

         // step 5: copy back
         CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost,
                  stream));
         CUDA_CHECK(cudaMemcpyAsync(V.data(), d_V, sizeof(data_type) * V.size(),
                  cudaMemcpyDeviceToHost, stream));
         CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost,
                  stream));
         CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

         CUDA_CHECK(cudaStreamSynchronize(stream));

         /* free resources */
         CUDA_CHECK(cudaFree(d_A));
         CUDA_CHECK(cudaFree(d_S));
         CUDA_CHECK(cudaFree(d_U));
         CUDA_CHECK(cudaFree(d_V));
         CUDA_CHECK(cudaFree(d_info));
         CUDA_CHECK(cudaFree(d_work));
         free(h_work);

         CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
         CUDA_CHECK(cudaStreamDestroy(stream));

         if(info){
            std::cout << "svd[d] failed with info=" << info << " for iop=" << iop << std::endl;
            exit(1);
         }
         VT = V.H();
      }

   /*
   // Input: a vector of matrices {c[l,r]}
   // Output: the reduced basis U[r,alpha] for the right space
   template <typename Tm> 
   void get_renorm_states_nkr_gpu(const std::vector<matrix<Tm>>& clr,
   std::vector<double>& sigs2,
   matrix<Tm>& U,
   const double rdm_svd,
   const bool debug_basis=false,
   const double thresh_Uortho=1.e-8){
   int nroots = clr.size();
   int diml = clr[0].rows();
   int dimr = clr[0].cols();
   if(dimr <= static_cast<int>(rdm_svd*diml)){ 

   if(debug_basis){ 
   std::cout << " RDM-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
   }
   matrix<Tm> rhor(dimr,dimr);
   for(int iroot=0; iroot<nroots; iroot++){
   rhor += xgemm("T","N",clr[iroot],clr[iroot].conj());
   } // iroot
   rhor *= 1.0/nroots;   
   sigs2.resize(dimr);
   eig_solver_gpu(rhor, sigs2, U, 1);

   }else{

   if(debug_basis){ 
   std::cout << " SVD-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
   }
   matrix<Tm> vrl(dimr,diml*nroots);
   for(int iroot=0; iroot<nroots; iroot++){
   auto crl = clr[iroot].T();
   xcopy(dimr*diml, crl.data(), vrl.col(iroot*diml));
   } // iroot
   vrl *= 1.0/std::sqrt(nroots);
   matrix<Tm> vt; // size of sig2,U,vt will be determined inside svd_solver!
   svd_solver_gpu(vrl, sigs2, U, vt);
   std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
   [](const double& x){ return x*x; });

   }
   if(debug_basis){ 
   std::cout << " sigs2[final]: ";
   for(const auto sig2 : sigs2) std::cout << sig2 << " ";
   std::cout << std::endl;                               
//U.print("U"); 
}
check_orthogonality(U, thresh_Uortho); // orthonormality is essential for variational calculations
}
*/

} // linalg

#endif

#endif
