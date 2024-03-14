#ifndef PREPROCESS_RLIST_H
#define PREPROCESS_RLIST_H

#include "preprocess_header.h"
#include "preprocess_mmbatch.h"

namespace ctns{

   // information for sigma = H*x with given symmetry blocks
   // then perform contraction to form renormalized operators
   // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] 
   // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] 
   // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
   template <typename Tm>
      struct Rblock{
         public:
            Rblock(const int _terms,
                  const int _cterms,
                  const int _alg_rcoper){
               terms = _terms;
               cterms = _cterms;
               alg_rcoper = _alg_rcoper;
            }
            bool identity(const int i) const{ return loc[i]==-1; }
            void display() const{
               std::cout << "size=" << size << " terms=" << terms 
                  << " offrop=" << offrop << " offin=" << offin << " offin2=" << offin2
                  << " dimin=" << dimin[0] << "," << dimin[1] << "," << dimin[2]
                  << " dimout=" << dimout[0] << "," << dimout[1] << "," << dimout[2]
                  << " dimin2=" << dimin2[0] << "," << dimin2[1] << "," << dimin2[2] 
                  << " identity=" << this->identity(0) << "," << this->identity(1) << ","
                  << this->identity(2) 
                  << " dagger=" << dagger[0] << "," << dagger[1] << "," << dagger[2]
                  << " loc=" << loc[0] << "," << loc[1] << "," << loc[2]
                  << " off=" << off[0] << "," << off[1] << "," << off[2]
                  << " terms=" << terms << " cterms=" << cterms << " alg_rcoper=" << alg_rcoper 
                  << " coeff=" << coeff
                  << " cost=" << cost
                  << std::endl;
            }
            // cost for contractions
            void setup(){
               int dfac = (icase == 1)? dimin2[2] : 1; // for batchGEMM
               std::vector<size_t> dimsInter = {dimin[0] *dimin[1] *dimout[2],
                  dimin[0] *dimout[1]*dimout[2],
                  dimout[0]*dimout[1]*dimout[2],
                  dimin2[icase]*dimout[icase]*dfac // renormalized operators
               };
               blksize = *std::max_element(dimsInter.begin(), dimsInter.end());
               if(!this->identity(2)) cost += 2*double(dimin[0])*dimin[1]*dimin[2]*dimout[2];
               if(!this->identity(1)) cost += 2*double(dimin[0])*dimin[1]*dimout[2]*dimout[1];
               if(!this->identity(0)) cost += 2*double(dimin[0])*dimout[1]*dimout[2]*dimout[0];
               // Additional information for psi*[br,bc,bm]
               cost += 2*double(dimin2[0])*dimin2[1]*dimin2[2]*dimout[icase];
            }
            void get_MMlist2_onedot(MMlist2<Tm>& MMlst2, const size_t offset=0, const bool ifbatch=false) const;
            void get_MMlist2(){
               mmlst2.resize(4);
               get_MMlist2_onedot(mmlst2);
            }
            bool kernel(const Tm* x, Tm** opaddr, Tm* workspace) const;
         public:
            int icase = -1; // 0:cr, 1:lc, 2:lr 
            int terms = 0; // no. of terms in Hmu 
            int cterms = 0, alg_rcoper = 0; // special treatment of coper
            // information of o1 and o2
            bool dagger[3] = {false,false,false};
            int loc[3] = {-1,-1,-1};
            size_t off[3] = {0,0,0};
            // information of psi*[br,bc,bm],psi[br',bc',bm']
            size_t dimin[3] = {0,0,0};  // psi[br',bc',bm']
            size_t dimout[3] = {0,0,0}; // sigma
            size_t dimin2[3] = {0,0,0}; // psi*[br,bc,bm]
            size_t offin = 0, offin2 = 0;
            size_t offrop = 0, size = 0; // size of output operator
            Tm coeff = 1.0;
            // for Matrix-Matrix multiplications
            size_t blksize = 0; // blksize of GEMM (can be different from size)
            double cost = 0.0;
            MMlist2<Tm> mmlst2;
            // intermediates [direct]
            int posInter = -1, lenInter = -1;
            size_t offInter = 0, ldaInter = 0; 
      };
   template <typename Tm>
      using Rlist = std::vector<Rblock<Tm>>;
   template <typename Tm>
      using Rlist2 = std::vector<std::vector<Rblock<Tm>>>; 

   template <typename Tm>
      void get_MMlist2(Rlist<Tm>& Rlst){
         // generate MMlist 
         for(int i=0; i<Rlst.size(); i++){
            Rlst[i].get_MMlist2();
         }
      }

   template <typename Tm>
      void get_MMlist2(Rlist2<Tm>& Rlst2){
         for(int i=0; i<Rlst2.size(); i++){
            auto& Rlst = Rlst2[i];
            get_MMlist2(Rlst);
         } // i
      }

   // Generation of MMlst following qtensor/contract_qt3_qt2.h
   // sigma[br,bc,bm] = coeff Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 	             Oc1^dagger2[bm,bm'] wf[br',bc',bm']
   // Additional information for psi*[br,bc,bm]
   // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] 
   // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] 
   // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
   template <typename Tm>
      void Rblock<Tm>::get_MMlist2_onedot(MMlist2<Tm>& MMlst2,
            const size_t offset,
            const bool ifbatch) const{
         // wf[br',bc',bm']
         int xloc = locIn, yloc = locOut;
         // ZL@20230519: whether perform contraction for op[c2/c1]
         // NOTE: no need to perform contraction for op[c] if alg_rcoper,
         //       because there is a last contraction with psi*. 
         const bool ifcntr = alg_rcoper==0;
         // ZL@20230228: ensure the output is always at the first part of 2*blksize
         int nt = ifcntr? terms : terms-cterms; 
         size_t xoff = offin, yoff = offset+(nt%2)*blksize;
         // Oc1^dagger2[bm,bm']: out(r,c,m) = o[d](m,x) in(r,c,x)
         if(!this->identity(2) && ifcntr){
            int p = 2;
            MMinfo<Tm> mm;
            mm.M = dimin[0]*dimin[1];
            mm.N = dimout[p];
            mm.K = dimin[p];
            mm.LDA = mm.M;
            mm.transA = 'N';
            mm.LDB = dagger[p]? mm.K : mm.N; // o(x,v) or o(v,x)
            mm.transB = dagger[p]? 'N' : 'T';
            mm.locA = xloc;   mm.offA = xoff;
            mm.locB = loc[p]; mm.offB = off[p];
            mm.locC = yloc;   mm.offC = yoff; 
            MMlst2[0].push_back(mm); 
            // update x & y  
            xloc = locOut; xoff = offset+(nt%2)*blksize; 
            yloc = locOut; yoff = offset+(1-nt%2)*blksize;
            nt -= 1;
         }
         // Or^dagger1[bc,bc']: out(r,c,m) = o[d](c,x) in(r,x,m) 
         if(!this->identity(1)){
            int p = 1;
            for(int im=0; im<dimout[2]; im++){
               MMinfo<Tm> mm;
               mm.M = dimin[0];
               mm.N = dimout[p];
               mm.K = dimin[p];
               mm.LDA = mm.M;
               mm.transA = 'N';
               mm.LDB = dagger[p]? mm.K : mm.N;
               mm.transB = dagger[p]? 'N' : 'T';
               mm.locA = xloc;   mm.offA = xoff+im*mm.M*mm.K;
               mm.locB = loc[p]; mm.offB = off[p];
               mm.locC = yloc;   mm.offC = yoff+im*mm.M*mm.N;
               MMlst2[1].push_back(mm);
            }
            // update x & y
            xloc = locOut; xoff = offset+(nt%2)*blksize;
            yloc = locOut; yoff = offset+(1-nt%2)*blksize;
            nt -= 1;
         }
         // Ol^dagger0[br,br']: out(r,c,m) = o[d](r,x) in(x,c,m)
         if(!this->identity(0)){
            int p = 0;
            MMinfo<Tm> mm;
            mm.M = dimout[p];
            mm.N = dimout[1]*dimout[2];
            mm.K = dimin[p];
            mm.LDA = dagger[p]? mm.K : mm.M;
            mm.transA = dagger[p]? 'T' : 'N';
            mm.LDB = mm.K;
            mm.transB = 'N';
            mm.locA = loc[p]; mm.offA = off[p];
            mm.locB = xloc;   mm.offB = xoff;
            mm.locC = yloc;   mm.offC = yoff;
            MMlst2[2].push_back(mm);
            // update x & y
            xloc = locOut; xoff = offset+(nt%2)*blksize;
            yloc = locOut; yoff = offset+(1-nt%2)*blksize;
            nt -= 1;
         }
         assert(nt == 0);
         //
         // Final contraction with psi*[br,bc,bm] to form operators
         // 
         if(icase == 0){
            // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] => (M,K) * (N,K)^T
            //
            // TODOs: need to add conj on the first part for complex object in future !
            //
            MMinfo<Tm> mm;
            mm.M = dimin2[0];
            mm.N = dimout[0];
            mm.K = dimin2[1]*dimin2[2];
            mm.LDA = mm.M;
            mm.transA = 'N';
            mm.LDB = mm.N;
            mm.transB = 'T';
            mm.locA = locIn; mm.offA = offin2;
            mm.locB = xloc;  mm.offB = xoff;
            mm.locC = yloc;  mm.offC = yoff;
            MMlst2[3].push_back(mm);
         }else if(icase == 1){
            // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] => (K,M)^H * (K,N) 
            for(int im=0; im<dimin2[2]; im++){
               MMinfo<Tm> mm;
               mm.M = dimin2[1];
               mm.N = dimout[1];
               mm.K = dimin2[0];
               mm.LDA = mm.K;
               mm.transA = 'C';
               mm.LDB = mm.K;
               mm.transB = 'N';
               mm.locA = locIn; mm.offA = offin2+im*mm.M*mm.K;
               mm.locB = xloc;  mm.offB = xoff+im*mm.N*mm.K;
               mm.locC = yloc;  mm.offC = yoff;
               if(ifbatch) mm.offC += im*mm.M*mm.N;
               MMlst2[3].push_back(mm);
            }
         }else if(icase == 2){
            // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
            MMinfo<Tm> mm;
            mm.M = dimin2[2];
            mm.N = dimout[2];
            mm.K = dimin2[0]*dimin2[1];
            mm.LDA = mm.K;
            mm.transA = 'C';
            mm.LDB = mm.K;
            mm.transB = 'N';
            mm.locA = locIn; mm.offA = offin2;
            mm.locB = xloc;  mm.offB = xoff;
            mm.locC = yloc;  mm.offC = yoff;
            MMlst2[3].push_back(mm);
         }
      }

   // Perform the actual matrix-matrix multiplication
   template <typename Tm>
      bool Rblock<Tm>::kernel(const Tm* x, Tm** opaddr, Tm* workspace) const{
         const Tm alpha = 1.0; 
         Tm* ptrs[7];
         ptrs[0] = opaddr[0]; // l
         ptrs[1] = opaddr[1]; // r
         ptrs[2] = opaddr[2]; // c
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4]; // inter
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;
         bool ifcal = false;
         for(int i=0; i<mmlst2.size(); i++){
            for(int j=0; j<mmlst2[i].size(); j++){
               ifcal = true;
               const auto& mm = mmlst2[i][j];
               Tm* Aptr = ptrs[mm.locA] + mm.offA;
               Tm* Bptr = ptrs[mm.locB] + mm.offB;
               Tm* Cptr = ptrs[mm.locC] + mm.offC;
               Tm beta = 0.0;
               if(icase == 1 && i==3 && j>0) beta = 1.0; // special treatment for "lc" with large branch
               linalg::xgemm(&mm.transA, &mm.transB, mm.M, mm.N, mm.K, alpha,
                     Aptr, mm.LDA, Bptr, mm.LDB, beta,
                     Cptr, mm.M);
            } // j
         } // i
         return ifcal;
      }

} // ctns

#endif
