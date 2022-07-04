#ifndef PREPROCESS_GEMM_BATCH_H
#define PREPROCESS_GEMM_BATCH_H

#include "blas_batch.h"
#include "preprocess_hxlist.h"

namespace ctns{

// Matrix-matrix operations: interface to XGEMM_BATCH
template <typename Tm>
struct MMbatch{
public:
   void init(const MMlist<Tm>& MMlst);
   void kernel(const int batchgemm, Tm** ptrs){
      if(batchgemm == 0){
	 this->xgemm(ptrs);   
      }else if(batchgemm == 1){
	 this->xgemm_batch(ptrs);    
      }
   }
   void xgemm(Tm** ptrs);
   void xgemm_batch(Tm** ptrs);
public:
   size_t size;
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
   for(int i=0; i<size; i++){
      const auto& mm = MMlst[i];
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
void MMbatch<Tm>::xgemm(Tm** ptrs){
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
void MMbatch<Tm>::xgemm_batch(Tm** ptrs){
   // initialization 
   for(int i=0; i<size; i++){
      Aptr[i] = ptrs[locA[i]] + offA[i];
      Bptr[i] = ptrs[locB[i]] + offB[i];
      Cptr[i] = ptrs[locC[i]] + offC[i];
   }
   int group_count = size; 
   linalg::xgemm_batch(transA.data(), transB.data(), M.data(), N.data(), K.data(), alpha_vec.data(), 
               	       Aptr.data(), LDA.data(), Bptr.data(), LDB.data(), beta_vec.data(),
               	       Cptr.data(), M.data(), &group_count, size_per_group_vec.data());
}

// Interface to Hxlist
template <typename Tm>
struct MMtask{
public:
   void init(const Hxlist<Tm>& Hxlst, 
	     const int _batchgemm,
	     const size_t _batchsize,
	     const size_t offset,
	     const int hdxorder,
	     const int iop=0);
   void kernel(const int k, Tm** ptrs){
      // perform GEMMs [c2,c1,r,l]
      for(int i=0; i<mmbatch2[k].size(); i++){
         mmbatch2[k][i].kernel(batchgemm, ptrs);
      }
   }
public:
   int batchgemm;
   size_t totsize, batchsize, nbatch;
   std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][iop]
};
template <typename Tm>
using MMtasks = std::vector<MMtask<Tm>>;

template <typename Tm>
void MMtask<Tm>::init(const Hxlist<Tm>& Hxlst,
		      const int _batchgemm,
		      const size_t _batchsize,
		      const size_t offset,
	 	      const int hxorder,
		      const int iop){
   batchgemm = _batchgemm;
   batchsize = _batchsize;
   totsize = Hxlst.size();
   nbatch = totsize/batchsize;
   if(totsize%batchsize != 0) nbatch += 1;
   mmbatch2.resize(nbatch);
   // process Hxlst
   for(int k=0; k<nbatch; k++){
      size_t off = k*batchsize;
      int jlen = std::min(totsize-off, batchsize);
      // initialization
      int nd = (iop==0)? 4 : 8;
      int pos[4] = {0,1,2,3};
      std::vector<size_t> dims(nd,0);
      for(int j=0; j<jlen; j++){
	 int jdx = off+j;
         auto& Hxblk = Hxlst[jdx];
	 if(iop == 1){
	    pos[0] = Hxblk.dagger[0]? 0 : 1;
	    pos[1] = Hxblk.dagger[1]? 2 : 3;
	    pos[2] = Hxblk.dagger[2]? 4 : 5;
	    pos[3] = Hxblk.dagger[3]? 6 : 7;
	 }
         dims[pos[0]] += Hxblk.identity(3)? 0 : 1; 
         dims[pos[1]] += Hxblk.identity(2)? 0 : Hxblk.dimout[3]; // c1
         dims[pos[2]] += Hxblk.identity(1)? 0 : Hxblk.dimout[3]*Hxblk.dimout[2]; // r
         dims[pos[3]] += Hxblk.identity(0)? 0 : 1; // l
      }
      // generation of mmlst2
      MMlist2<Tm> mmlst2(nd);
      for(int i=0; i<nd; i++){
         mmlst2[i].resize(dims[i]); // c2,c1,r,l
      }
      std::vector<size_t> idx(nd,0);
      for(int j=0; j<jlen; j++){
	 int jdx = off+j;
	 auto& Hxblk = Hxlst[jdx];
	 if(iop == 1){
	    pos[0] = Hxblk.dagger[0]? 0 : 1;
	    pos[1] = Hxblk.dagger[1]? 2 : 3;
	    pos[2] = Hxblk.dagger[2]? 4 : 5;
	    pos[3] = Hxblk.dagger[3]? 6 : 7;
	 }
         MMlist2<Tm> mmtmp2(4);
         Hxblk.get_MMlist_twodot(mmtmp2, j*offset);
         for(int i=0; i<mmtmp2.size(); i++){
            int ipos = pos[i];
	    // copy the mmlst to the correct place
            for(int k=0; k<mmtmp2[i].size(); k++){
               mmlst2[ipos][idx[ipos]] = mmtmp2[i][k];
               idx[ipos]++;
            } //k
         } // i
      } // j
      // convert to batch list
      mmbatch2[k].resize(nd);
      for(int i=0; i<nd; i++){
         if(hxorder == 1){ // sort by cost
            std::stable_sort(mmlst2[i].begin(), mmlst2[i].end(),
           		     [](const MMinfo<Tm>& mm1, const MMinfo<Tm>& mm2){
           		        return mm1.cost() > mm2.cost();
           		     });
         }
         mmbatch2[k][i].init(mmlst2[i]);
     } // i
   } // k
}

} // ctns

#endif
