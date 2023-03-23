#ifndef PREPROCESS_HXLIST_H
#define PREPROCESS_HXLIST_H

#include "preprocess_header.h"
#include "preprocess_mmbatch.h"

namespace ctns{

   // information for sigma = H*x with given symmetry blocks
   template <typename Tm>
      struct Hxblock{
         public:
            Hxblock(const int _dims, const int _terms): dims(_dims), terms(_terms) {};
            bool identity(const int i) const{ return loc[i]==-1; }    
            void display() const{
               std::cout << "offout=" << offout << " offin=" << offin 
                  << " dimout=" << dimout[0] << "," << dimout[1] << "," << dimout[2] << "," << dimout[3]
                  << " dimin=" << dimin[0] << "," << dimin[1] << "," << dimin[2] << "," << dimin[3]
                  << " identity=" << this->identity(0) << "," << this->identity(1) << "," 
                  << this->identity(2) << "," << this->identity(3) 
                  << " dagger=" << dagger[0] << "," << dagger[1] << "," << dagger[2] << "," << dagger[3]
                  << " loc=" << loc[0] << "," << loc[1] << "," << loc[2] << "," << loc[3]
                  << " off=" << off[0] << "," << off[1] << "," << off[2] << "," << off[3]
                  << " coeff=" << coeff
                  << " cost=" << cost
                  << std::endl;  
            }
            // cost for contractions
            void setup(){         
               if(dims == 4){
                  std::vector<size_t> dimsInter = {dimin[0] *dimin[1] *dimin[2] *dimout[3],
                     dimin[0] *dimin[1] *dimout[2]*dimout[3],
                     dimin[0] *dimout[1]*dimout[2]*dimout[3],
                     dimout[0]*dimout[1]*dimout[2]*dimout[3]};
                  blksize = *std::max_element(dimsInter.begin(), dimsInter.end());
                  if(!this->identity(3)) cost += 2*double(dimin[0])*dimin[1]*dimin[2]*dimin[3]*dimout[3]; // Oc2
                  if(!this->identity(2)) cost += 2*double(dimin[0])*dimin[1]*dimin[2]*dimout[3]*dimout[2]; // Oc1
                  if(!this->identity(1)) cost += 2*double(dimin[0])*dimin[1]*dimout[2]*dimout[3]*dimout[1]; // Or
                  if(!this->identity(0)) cost += 2*double(dimin[0])*dimout[1]*dimout[2]*dimout[3]*dimout[0]; // Ol
               }else if(dims == 3){
                  std::vector<size_t> dimsInter = {dimin[0] *dimin[1] *dimout[2],
                     dimin[0] *dimout[1]*dimout[2],
                     dimout[0]*dimout[1]*dimout[2]};
                  blksize = *std::max_element(dimsInter.begin(), dimsInter.end());
                  if(!this->identity(2)) cost += 2*double(dimin[0])*dimin[1]*dimin[2]*dimout[2];
                  if(!this->identity(1)) cost += 2*double(dimin[0])*dimin[1]*dimout[2]*dimout[1];
                  if(!this->identity(0)) cost += 2*double(dimin[0])*dimout[1]*dimout[2]*dimout[0];
               }else{
                  std::cout << "error: no such option for dims=" << dims << std::endl;
                  exit(1);
               }
            }
            void get_MMlist(){
               MMlist2<Tm> MMlst2(4);
               if(dims == 4){
                  get_MMlist_twodot(MMlst2);
               }else if(dims == 3){
                  get_MMlist_onedot(MMlst2);
               }else{
                  std::cout << "error: no such option for dims=" << dims << std::endl;
                  exit(1);
               }
               // flatten MMlst2 to MMlst
               size_t size = MMlst2[0].size() + MMlst2[1].size() + MMlst2[2].size() + MMlst2[3].size();
               assert(size > 0);
               MMlst.resize(size);
               int idx = 0;
               for(int i=0; i<dims; i++){
                  for(int j=0; j<MMlst2[i].size(); j++){
                     MMlst[idx] = MMlst2[i][j];
                     idx++;
                  } // j
               } // i
            }
            void get_MMlist_twodot(MMlist2<Tm>& MMlst2, const size_t offset=0) const;
            void get_MMlist_onedot(MMlist2<Tm>& MMlst2, const size_t offset=0) const;
            void kernel(const Tm* x, Tm** opaddr, Tm* workspace) const;
         public:
            int dims  = 0; // 3/4 for onedot/twodot
            int terms = 0; // no. of terms in Hmu 
            size_t dimout[4] = {0,0,0,0};
            size_t dimin[4] = {0,0,0,0};
            bool dagger[4] = {false,false,false,false};
            int loc[4] = {-1,-1,-1,-1};
            size_t off[4] = {0,0,0,0};
            size_t offout = 0, offin = 0, size = 0; // size of output block 
            Tm coeff = 1.0;
            // for Matrix-Matrix multiplications
            size_t blksize = 0; // blksize of GEMM (can be different from size)
            double cost = 0.0;
            MMlist<Tm> MMlst;
            // intermediates [direct]
            int posInter = -1;
            size_t ldaInter = 0; 
            std::vector<Tm> alpha_vec;
      };
   template <typename Tm>
      using Hxlist = std::vector<Hxblock<Tm>>;
   template <typename Tm>
      using Hxlist2 = std::vector<std::vector<Hxblock<Tm>>>; 

   template <typename Tm>
      void get_MMlist(Hxlist<Tm>& Hxlst, const int hxorder){
         // sort Hxlst
         if(hxorder == 1){ // sort by cost
            std::stable_sort(Hxlst.begin(), Hxlst.end(),
                  [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
                  return t1.cost > t2.cost; });
         }else if(hxorder == 2){ // sort by cost
            std::stable_sort(Hxlst.begin(), Hxlst.end(),
                  [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
                  return t1.cost < t2.cost; });
         }else if(hxorder == 3){ // sort by offout
            std::stable_sort(Hxlst.begin(), Hxlst.end(),
                  [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
                  return t1.offout < t2.offout; });
         }else if(hxorder == 4){ // sort by offin
            std::stable_sort(Hxlst.begin(), Hxlst.end(),
                  [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
                  return t1.offin < t2.offin; });
         } // hxorder
           // generate MMlist 
         for(int i=0; i<Hxlst.size(); i++){
            Hxlst[i].get_MMlist();
         }
      }

   template <typename Tm>
      void get_MMlist(Hxlist2<Tm>& Hxlst2, const int hxorder){
         for(int i=0; i<Hxlst2.size(); i++){
            auto& Hxlst = Hxlst2[i];
            get_MMlist(Hxlst, hxorder);
         } // i
      }

   // Generation of MMlst following qtensor/contract_qt4_qt2.h
   // sigma[br,bc,bm,bv] = coeff Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 			Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv']
   // 			wf[br',bc',bm',bv']
   template <typename Tm>
      void Hxblock<Tm>::get_MMlist_twodot(MMlist2<Tm>& MMlst2,
            const size_t offset) const{
         // wf[br',bc',bm',bv']
         int xloc = locIn, yloc = locOut;
         int nt = terms+1; // ZL@20230228: ensure the output is always at the first part of 2*blksize
         size_t xoff = offin, yoff = offset+(nt%2)*blksize;
         // Oc2^dagger3[bv,bv']: out(r,c,m,v) = o[d](v,x) in(r,c,m,x) 
         if(!this->identity(3)){
            int p = 3;
            MMinfo<Tm> mm;
            mm.M = dimin[0]*dimin[1]*dimin[2];
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
         // Oc1^dagger2[bm,bm']: out(r,c,m,v) = o[d](m,x) in(r,c,x,v)
         if(!this->identity(2)){
            int p = 2;
            for(int iv=0; iv<dimout[3]; iv++){
               MMinfo<Tm> mm;
               mm.M = dimin[0]*dimin[1];
               mm.N = dimout[p];
               mm.K = dimin[p];
               mm.LDA = mm.M;
               mm.transA = 'N';
               mm.LDB = dagger[p]? mm.K : mm.N;
               mm.transB = dagger[p]? 'N' : 'T';
               mm.locA = xloc;   mm.offA = xoff+iv*mm.M*mm.K;
               mm.locB = loc[p]; mm.offB = off[p];
               mm.locC = yloc;   mm.offC = yoff+iv*mm.M*mm.N;
               MMlst2[1].push_back(mm);
            }
            // update x & y
            xloc = locOut; xoff = offset+(nt%2)*blksize;
            yloc = locOut; yoff = offset+(1-nt%2)*blksize;
            nt -= 1;
         }
         // Or^dagger1[bc,bc']: out(r,c,m,v) = o[d](c,x) in(r,x,m,v) 
         if(!this->identity(1)){
            int p = 1;
            for(int iv=0; iv<dimout[3]; iv++){
               for(int im=0; im<dimout[2]; im++){
                  MMinfo<Tm> mm;
                  mm.M = dimin[0];
                  mm.N = dimout[p];
                  mm.K = dimin[p];
                  mm.LDA = mm.M;
                  mm.transA = 'N';
                  mm.LDB = dagger[p]? mm.K : mm.N;
                  mm.transB = dagger[p]? 'N' : 'T';
                  mm.locA = xloc;   mm.offA = xoff+(iv*dimout[2]+im)*mm.M*mm.K;
                  mm.locB = loc[p]; mm.offB = off[p];
                  mm.locC = yloc;   mm.offC = yoff+(iv*dimout[2]+im)*mm.M*mm.N;
                  MMlst2[2].push_back(mm);
               }
            }
            // update x & y
            xloc = locOut; xoff = offset+(nt%2)*blksize;
            yloc = locOut; yoff = offset+(1-nt%2)*blksize;
            nt -= 1;	  
         }
         // Ol^dagger0[br,br']: out(r,c,m,v) = o[d](r,x) in(x,c,m,v)
         if(!this->identity(0)){
            int p = 0;	   
            MMinfo<Tm> mm;
            mm.M = dimout[p];
            mm.N = dimout[1]*dimout[2]*dimout[3];
            mm.K = dimin[p];
            mm.LDA = dagger[p]? mm.K : mm.M;
            mm.transA = dagger[p]? 'T' : 'N';
            mm.LDB = mm.K;
            mm.transB = 'N';
            mm.locA = loc[p]; mm.offA = off[p];
            mm.locB = xloc;   mm.offB = xoff;
            mm.locC = yloc;   mm.offC = yoff;
            MMlst2[3].push_back(mm);
            nt -= 1;
         }
         assert(nt == 1);
      }

   // Generation of MMlst following qtensor/contract_qt3_qt2.h
   // sigma[br,bc,bm] = coeff Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 	             Oc1^dagger2[bm,bm'] 
   // 		     wf[br',bc',bm']
   template <typename Tm>
      void Hxblock<Tm>::get_MMlist_onedot(MMlist2<Tm>& MMlst2,
            const size_t offset) const{
         // wf[br',bc',bm']
         int xloc = locIn, yloc = locOut;
         int nt = terms+1; // ZL@20230228: ensure the output is always at the first part of 2*blksize
         size_t xoff = offin, yoff = offset+(nt%2)*blksize;
         // Oc1^dagger2[bm,bm']: out(r,c,m) = o[d](m,x) in(r,c,x)
         if(!this->identity(2)){
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
            nt -= 1;
         }
         assert(nt == 1);
      }

   // Perform the actual matrix-matrix multiplication
   template <typename Tm>
      void Hxblock<Tm>::kernel(const Tm* x, Tm** opaddr, Tm* workspace) const{ 
         const Tm alpha = 1.0, beta = 0.0;
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;
         for(int i=0; i<MMlst.size(); i++){
            const auto& mm = MMlst[i];
            Tm* Aptr = ptrs[mm.locA] + mm.offA;
            Tm* Bptr = ptrs[mm.locB] + mm.offB;
            Tm* Cptr = ptrs[mm.locC] + mm.offC;
            linalg::xgemm(&mm.transA, &mm.transB, mm.M, mm.N, mm.K, alpha,
                  Aptr, mm.LDA, Bptr, mm.LDB, beta,
                  Cptr, mm.M);
         }
      }

} // ctns

#endif
