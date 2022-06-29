#ifndef PREPROCESS_SIGMA_BATCH_H
#define PREPROCESS_SIGMA_BATCH_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

extern "C" {

void dgemm_batch_(const char *transa_array, const char *transb_array, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha_array, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta_array, double **c_array, const int *ldc_array, 
        const int *group_count, const int *group_size);

void zgemm_batch_(const char *transa_array, const char *transb_array, 
        const int *m_array, const int *n_array, const int *k_array,
        const std::complex<double> *alpha_array, const std::complex<double> **a_array, 
        const int *lda_array,
        const std::complex<double> **b_array, const int *ldb_array,
        const std::complex<double> *beta_array, std::complex<double> **c_array, 
        const int *ldc_array, 
        const int *group_count, const int *group_size);

}


namespace ctns{

inline void xgemm_batch(const char *transa_array, const char *transb_array, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha_array, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta_array, double **c_array, const int *ldc_array, 
        const int *group_count, const int *group_size)
{
    return ::dgemm_batch_(
            transa_array, transb_array, 
            m_array, n_array, k_array,
            alpha_array, a_array, lda_array, 
            b_array, ldb_array,
            beta_array, c_array, ldc_array, 
            group_count, group_size
            );
}
inline void xgemm_batch(const char *transa_array, const char *transb_array, 
        const int *m_array, const int *n_array, const int *k_array,
        const std::complex<double> *alpha_array, 
        const std::complex<double> **a_array, const int *lda_array,
        const std::complex<double> **b_array, const int *ldb_array,
        const std::complex<double> *beta_array, std::complex<double> **c_array, 
        const int *ldc_array, 
        const int *group_count, const int *group_size)
{
    return ::zgemm_batch_(
            transa_array, transb_array, 
            m_array, n_array, k_array,
            alpha_array, a_array, lda_array, 
            b_array, ldb_array,
            beta_array, c_array, ldc_array, 
            group_count, group_size
            );
}

template <typename Tm>
void batchGEMM(const std::vector<char>& transA,
	       const std::vector<char>& transB,
	       const std::vector<int>& Mlst,
	       const std::vector<int>& Nlst,
	       const std::vector<int>& Klst,
	       const std::vector<int>& LDAlst,
	       const std::vector<int>& LDBlst,
	       std::vector<const Tm*>& Aptr,
	       std::vector<const Tm*>& Bptr,
	       std::vector<Tm*>& Cptr){
   int group_count = transA.size();
   const std::vector<Tm> alpha_vector(group_count,1.0);
   const std::vector<Tm> beta_vector(group_count,0.0);
   const std::vector<int> size_per_group_vector(group_count,1);
   xgemm_batch(transA.data(), transB.data(),
	       Mlst.data(), Nlst.data(), Klst.data(), 
	       alpha_vector.data(), Aptr.data(), LDAlst.data(),
	       Bptr.data(), LDBlst.data(), beta_vector.data(),
	       Cptr.data(), Mlst.data(), 
	       &group_count, size_per_group_vector.data());
/*
   const Tm alpha = 1.0, beta = 0.0;
   for(int i=0; i<transA.size(); i++){
      linalg::xgemm(&transA[i], &transB[i], &Mlst[i], &Nlst[i], &Klst[i], &alpha,
		    Aptr[i], &LDAlst[i], Bptr[i], &LDBlst[i], &beta,
		    Cptr[i], &Mlst[i]);
   }
*/
}

// for Davidson diagonalization
template <typename Tm> 
void preprocess_Hx_batch(Tm* y,
	                 const Tm* x,
		         const Tm& scale,
		         const int& size,
	                 const int& rank,
		         const size_t& ndim,
	                 const size_t& blksize,
			 const size_t& batchsize,
	                 Hxlist2<Tm>& Hxlst2,
		         Tm* workspace){
   const bool debug = false;
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::preprocess_Hx_batch"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }

   // initialization
   memset(y, 0, ndim*sizeof(Tm));

   std::vector<Tm*> xaddr(batchsize);
   std::vector<Tm*> yaddr(batchsize);
   std::vector<int[4]> din(batchsize);
   std::vector<int> nt(batchsize); 

   // loop over nonzero blocks
   for(int i=0; i<Hxlst2.size(); i++){
 
      int fsize = Hxlst2[i].size();
      int nbatch = fsize/batchsize;
      if(fsize%batchsize != 0) nbatch += 1;

      // partition formulae into batches
      for(int k=0; k<nbatch; k++){
	 
	 // initialization
	 int jlen = std::min(fsize-k*batchsize, batchsize);
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    xaddr[j] = const_cast<Tm*>(x)+Hxblk.offin;
	    yaddr[j] = &workspace[j*blksize*2];
	    din[j][0] = Hxblk.dimin[0];
	    din[j][1] = Hxblk.dimin[1];
	    din[j][2] = Hxblk.dimin[2];
	    din[j][3] = Hxblk.dimin[3];
	 }
	 std::fill_n(nt.begin(), jlen, 0);
	 
	 // 1. Oc2
	 std::vector<char> transA, transB;
	 std::vector<int> Mlst, Nlst, Klst, LDAlst, LDBlst;
	 std::vector<const Tm*> Aptr, Bptr;
	 std::vector<Tm*> Cptr;
	 // form batched GEMM
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    if(Hxblk.identity(3)) continue;
	    int M = din[j][0]*din[j][1]*din[j][2];
	    int N = Hxblk.dimout[3];
	    int K = din[j][3];
	    Mlst.push_back(M);
	    Nlst.push_back(N);
	    Klst.push_back(K);
            LDAlst.push_back(M);
	    transA.push_back('N');
	    LDBlst.push_back(Hxblk.dagger[3]? K : N);
	    transB.push_back(Hxblk.dagger[3]? 'N' : 'T');
            Aptr.push_back(xaddr[j]);
            Bptr.push_back(Hxblk.addr[3]);
	    Cptr.push_back(yaddr[j]);
	    xaddr[j] = &workspace[j*blksize*2+(nt[j]%2)*Hxblk.blksize];
	    yaddr[j] = &workspace[j*blksize*2+(1-nt[j]%2)*Hxblk.blksize];
	    din[j][3] = Hxblk.dimout[3];
	    nt[j] += 1;
	 }
	 batchGEMM(transA, transB, Mlst, Nlst, Klst, 
		   LDAlst, LDBlst, Aptr, Bptr, Cptr);
	 
	 // 2. Oc1
	 transA.clear(); transB.clear();
         Mlst.clear(); Nlst.clear(); Klst.clear(); LDAlst.clear(); LDBlst.clear();
	 Aptr.clear(); Bptr.clear(); Cptr.clear();
	 // form batched GEMM
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    if(Hxblk.identity(2)) continue;
	    for(int iv=0; iv<din[j][3]; iv++){
	       int M = din[j][0]*din[j][1];
	       int N = Hxblk.dimout[2];
	       int K = din[j][2];
	       Mlst.push_back(M);
	       Nlst.push_back(N);
	       Klst.push_back(K);
               LDAlst.push_back(M);
	       transA.push_back('N');
	       LDBlst.push_back(Hxblk.dagger[2]? K : N);
	       transB.push_back(Hxblk.dagger[2]? 'N' : 'T');
               Aptr.push_back(xaddr[j]+iv*M*K);
               Bptr.push_back(Hxblk.addr[2]);
	       Cptr.push_back(yaddr[j]+iv*M*N);
	    }
	    xaddr[j] = &workspace[j*blksize*2+(nt[j]%2)*Hxblk.blksize];
	    yaddr[j] = &workspace[j*blksize*2+(1-nt[j]%2)*Hxblk.blksize];
	    din[j][2] = Hxblk.dimout[2];
	    nt[j] += 1;
	 }
	 batchGEMM(transA, transB, Mlst, Nlst, Klst, 
		   LDAlst, LDBlst, Aptr, Bptr, Cptr);
	 
	 // 3. Or
	 transA.clear(); transB.clear();
         Mlst.clear(); Nlst.clear(); Klst.clear(); LDAlst.clear(); LDBlst.clear();
	 Aptr.clear(); Bptr.clear(); Cptr.clear();
	 // form batched GEMM
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    if(Hxblk.identity(1)) continue;
	    for(int iv=0; iv<din[j][3]; iv++){
	       for(int im=0; im<din[j][2]; im++){
	          int M = din[j][0];
	          int N = Hxblk.dimout[1];
	          int K = din[j][1];
	          Mlst.push_back(M);
	          Nlst.push_back(N);
	          Klst.push_back(K);
                  LDAlst.push_back(M);
	          transA.push_back('N');
	          LDBlst.push_back(Hxblk.dagger[1]? K : N);
	          transB.push_back(Hxblk.dagger[1]? 'N' : 'T');
                  Aptr.push_back(xaddr[j]+(iv*din[j][2]+im)*M*K);
                  Bptr.push_back(Hxblk.addr[1]);
	          Cptr.push_back(yaddr[j]+(iv*din[j][2]+im)*M*N);
	       }
	    }
	    xaddr[j] = &workspace[j*blksize*2+(nt[j]%2)*Hxblk.blksize];
	    yaddr[j] = &workspace[j*blksize*2+(1-nt[j]%2)*Hxblk.blksize];
	    din[j][1] = Hxblk.dimout[1];
	    nt[j] += 1;
	 }
	 batchGEMM(transA, transB, Mlst, Nlst, Klst, 
		   LDAlst, LDBlst, Aptr, Bptr, Cptr);
	 
	 // 4. Ol
	 transA.clear(); transB.clear();
         Mlst.clear(); Nlst.clear(); Klst.clear(); LDAlst.clear(); LDBlst.clear();
	 Aptr.clear(); Bptr.clear(); Cptr.clear();
	 // form batched GEMM
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    if(Hxblk.identity(0)) continue;
	    int M = Hxblk.dimout[0];
	    int N = din[j][1]*din[j][2]*din[j][3];
	    int K = din[j][0];
	    Mlst.push_back(M);
	    Nlst.push_back(N);
	    Klst.push_back(K);
            LDAlst.push_back(Hxblk.dagger[0]? K : M);
	    transA.push_back(Hxblk.dagger[0]? 'T' : 'N');
	    LDBlst.push_back(K);
	    transB.push_back('N');
            Aptr.push_back(Hxblk.addr[0]);
            Bptr.push_back(xaddr[j]);
	    Cptr.push_back(yaddr[j]);
	    xaddr[j] = &workspace[j*blksize*2+(nt[j]%2)*Hxblk.blksize];
	    yaddr[j] = &workspace[j*blksize*2+(1-nt[j]%2)*Hxblk.blksize];
	    din[j][0] = Hxblk.dimout[0];
	    nt[j] += 1;
	 }
	 batchGEMM(transA, transB, Mlst, Nlst, Klst, 
		   LDAlst, LDBlst, Aptr, Bptr, Cptr);

         // reduction
         for(int j=0; j<jlen; j++){
	    int jdx = k*batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    assert(din[j][0]==Hxblk.dimout[0] && 
		   din[j][1]==Hxblk.dimout[1] &&
		   din[j][2]==Hxblk.dimout[2] &&
		   din[j][3]==Hxblk.dimout[3]);
            linalg::xaxpy(Hxblk.size, Hxblk.coeff, xaddr[j], y+Hxblk.offout);
	 }

      } // k   
   } // i

   // add const term
   if(rank == 0){
      linalg::xaxpy(ndim, scale, x, y);
   }
}


} // ctns

#endif
