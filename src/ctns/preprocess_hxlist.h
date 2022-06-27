#ifndef PREPROCESS_HXLIST_H
#define PREPROCESS_HXLIST_H

namespace ctns{

// Matrix-matrix operations: interface to XGEMM
template <typename Tm>
struct MMinfo{
public:
   char transA, transB;
   int M, N, K, LDA, LDB;
   int locA, locB, locC;
   size_t offA, offB, offC;
};
template <typename Tm>
using MMlist = std::vector<MMinfo<Tm>>;  

// information for sigma[br,bc,bm,bv] 
template <typename Tm>
struct Hxblock{
public:
   void display() const;
   bool identity(const int i) const{ return addr[i]==nullptr; }    
   void gen_MMlist(const int cases);   
   void gen_MMlist_twodot();
   void gen_MMlist_onedot();
   void kernel(const Tm* x, Tm* workspace);
public:
   size_t dimout[4] = {0,0,0,0};
   size_t dimin[4] = {0,0,0,0};
   bool dagger[4] = {false,false,false,false};
   Tm* addr[6] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
   size_t offout = 0, offin = 0;
   size_t size = 0; // dimout[0]*dimout[1]*dimout[2]*dimout[3]
   Tm coeff = 1.0;
   // for Matrix-Matrix multiplications
   size_t tmpsize = 0, offres = 0;
   double cost = 0.0;
   MMlist<Tm> MMlst;
};
template <typename Tm>
using Hxlist = std::vector<Hxblock<Tm>>;  

template <typename Tm>
void Hxblock<Tm>::display() const{
   std::cout << "offout=" << offout << " offin=" << offin 
	     << " dimout=" << dimout[0] << "," << dimout[1] << "," << dimout[2] << "," << dimout[3]
	     << " dimin=" << dimin[0] << "," << dimin[1] << "," << dimin[2] << "," << dimin[3]
	     << " identity=" << this->identity(0) << "," << this->identity(1) << "," 
	     		     << this->identity(2) << "," << this->identity(3) 
	     << " dagger=" << dagger[0] << "," << dagger[1] << "," << dagger[2] << "," << dagger[3]
	     << " coeff=" << coeff
	     << " cost=" << cost
	     << std::endl;  
}

template <typename Tm>
void Hxblock<Tm>::gen_MMlist(const int cases){
   if(cases == 4){
      gen_MMlist_twodot();
   }else if(cases == 3){
      gen_MMlist_onedot();
   }else{
      std::cout << "error: no such option for cases=" << cases << std::endl;
      exit(1);
   }
}

// compute sigma[br,bc,bm,bv] = coeff Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
// 			        Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv']
// 			        wf[br',bc',bm',bv']
template <typename Tm>
void Hxblock<Tm>::gen_MMlist_twodot(){
   std::vector<size_t> dims = {dimin[0] *dimin[1] *dimin[2] *dimout[3],
		               dimin[0] *dimin[1] *dimout[2]*dimout[3],
		      	       dimin[0] *dimout[1]*dimout[2]*dimout[3],
		      	       dimout[0]*dimout[1]*dimout[2]*dimout[3]};
   size_t blksize = *std::max_element(dims.begin(), dims.end());
   tmpsize = 2*blksize;
   // wf[br',bc',bm',bv']
   int xloc = 4, yloc = 5; 
   size_t xoff = offin, yoff = 0;
   int din[4] = {dimin[0],dimin[1],dimin[2],dimin[3]}; 
   int nt = 0;
   // Oc2^dagger3[bv,bv']: out(r,c,m,v) = o[d](v,x) in(r,c,m,x) 
   if(!this->identity(3)){
      int p = 3;
      MMinfo<Tm> mm;
      mm.M = din[0]*din[1]*din[2];
      mm.N = dimout[p];
      mm.K = din[p];
      mm.LDA = mm.M;
      mm.transA = 'N';
      mm.LDB = dagger[p]? mm.K : mm.N; // o(x,v) or o(v,x)
      mm.transB = dagger[p]? 'N' : 'T';
      mm.locA = xloc; mm.offA = xoff;
      mm.locB = p;    mm.offB = 0;
      mm.locC = yloc; mm.offC = yoff; 
      MMlst.push_back(mm); 
      // update x & y  
      xloc = 5; xoff = (nt%2)*blksize; 
      yloc = 5; yoff = (1-nt%2)*blksize;
      cost += double(din[0])*din[1]*din[2]*din[3]*dimout[p];
      din[p] = dimout[p];
      nt += 1;
   }
   // Oc1^dagger2[bm,bm']: out(r,c,m,v) = o[d](m,x) in(r,c,x,v)
   if(!this->identity(2)){
      int p = 2;
      for(int iv=0; iv<din[3]; iv++){
	 MMinfo<Tm> mm;
	 mm.M = din[0]*din[1];
	 mm.N = dimout[p];
	 mm.K = din[p];
	 mm.LDA = mm.M;
	 mm.transA = 'N';
	 mm.LDB = dagger[p]? mm.K : mm.N;
	 mm.transB = dagger[p]? 'N' : 'T';
	 mm.locA = xloc; mm.offA = xoff+iv*mm.M*mm.K;
	 mm.locB = p;    mm.offB = 0;
	 mm.locC = yloc; mm.offC = yoff+iv*mm.M*mm.N;
	 MMlst.push_back(mm);
      }
      // update x & y
      xloc = 5; xoff = (nt%2)*blksize;
      yloc = 5; yoff = (1-nt%2)*blksize;
      cost += double(din[0])*din[1]*din[2]*din[3]*dimout[p];
      din[p] = dimout[p];
      nt += 1;
   }
   // Or^dagger1[bc,bc']: out(r,c,m,v) = o[d](c,x) in(r,x,m,v) 
   if(!this->identity(1)){
      int p = 1;
      for(int iv=0; iv<din[3]; iv++){
         for(int im=0; im<din[2]; im++){
	    MMinfo<Tm> mm;
	    mm.M = din[0];
	    mm.N = dimout[p];
	    mm.K = din[p];
	    mm.LDA = mm.M;
	    mm.transA = 'N';
	    mm.LDB = dagger[p]? mm.K : mm.N;
	    mm.transB = dagger[p]? 'N' : 'T';
	    mm.locA = xloc; mm.offA = xoff+(iv*din[2]+im)*mm.M*mm.K;
	    mm.locB = p;    mm.offB = 0;
	    mm.locC = yloc; mm.offC = yoff+(iv*din[2]+im)*mm.M*mm.N;
	    MMlst.push_back(mm);
	 }
      }
      // update x & y
      xloc = 5; xoff = (nt%2)*blksize;
      yloc = 5; yoff = (1-nt%2)*blksize;
      cost += double(din[0])*din[1]*din[2]*din[3]*dimout[p];
      din[p] = dimout[p];
      nt += 1;	  
   }
   // Ol^dagger0[br,br']: out(r,c,m,v) = o[d](r,x) in(x,c,m,v)
   if(!this->identity(0)){
      int p = 0;	   
      MMinfo<Tm> mm;
      mm.M = dimout[p];
      mm.N = din[1]*din[2]*din[3];
      mm.K = din[p];
      mm.LDA = dagger[p]? mm.K : mm.M;
      mm.transA = dagger[p]? 'T' : 'N';
      mm.LDB = mm.K;
      mm.transB = 'N';
      mm.locA = p;    mm.offA = 0;
      mm.locB = xloc; mm.offB = xoff;
      mm.locC = yloc; mm.offC = yoff;
      MMlst.push_back(mm);
      xoff = (nt%2)*blksize;
      assert(xoff == yoff);
      cost += double(din[0])*din[1]*din[2]*din[3]*dimout[p];
      din[p] = dimout[p];
   }
   assert(din[0]==dimout[0] && din[1]==dimout[1] &&
          din[2]==dimout[2] && din[3]==dimout[3]);		  
   offres = xoff;
}

// compute sigma[br,bc,bm] = coeff Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
// 			     Oc1^dagger2[bm,bm'] 
// 			     wf[br',bc',bm']
template <typename Tm>
void Hxblock<Tm>::gen_MMlist_onedot(){
   std::vector<size_t> dims = {dimin[0] *dimin[1] *dimout[2],
		               dimin[0] *dimout[1]*dimout[2],
		      	       dimout[0]*dimout[1]*dimout[2]};
   size_t blksize = *std::max_element(dims.begin(), dims.end());
   tmpsize = 2*blksize;
   // wf[br',bc',bm']
   int xloc = 4, yloc = 5;
   size_t xoff = offin, yoff = 0;
   int din[3] = {dimin[0],dimin[1],dimin[2]};
   int nt = 0;
   // Oc1^dagger2[bm,bm']: out(r,c,m) = o[d](m,x) in(r,c,x)
   if(!this->identity(2)){
      int p = 2;
      MMinfo<Tm> mm;
      mm.M = din[0]*din[1];
      mm.N = dimout[p];
      mm.K = din[p];
      mm.LDA = mm.M;
      mm.transA = 'N';
      mm.LDB = dagger[p]? mm.K : mm.N; // o(x,v) or o(v,x)
      mm.transB = dagger[p]? 'N' : 'T';
      mm.locA = xloc; mm.offA = xoff;
      mm.locB = p;    mm.offB = 0;
      mm.locC = yloc; mm.offC = yoff; 
      MMlst.push_back(mm); 
      // update x & y  
      xloc = 5; xoff = (nt%2)*blksize; 
      yloc = 5; yoff = (1-nt%2)*blksize;
      cost += double(din[0])*din[1]*din[2]*dimout[p];
      din[p] = dimout[p];
      nt += 1;
   }
   // Or^dagger1[bc,bc']: out(r,c,m) = o[d](c,x) in(r,x,m) 
   if(!this->identity(1)){
      int p = 1;
      for(int im=0; im<din[2]; im++){
	 MMinfo<Tm> mm;
	 mm.M = din[0];
	 mm.N = dimout[p];
	 mm.K = din[p];
	 mm.LDA = mm.M;
	 mm.transA = 'N';
	 mm.LDB = dagger[p]? mm.K : mm.N;
	 mm.transB = dagger[p]? 'N' : 'T';
	 mm.locA = xloc; mm.offA = xoff+im*mm.M*mm.K;
	 mm.locB = p;    mm.offB = 0;
	 mm.locC = yloc; mm.offC = yoff+im*mm.M*mm.N;
	 MMlst.push_back(mm);
      }
      // update x & y
      xloc = 5; xoff = (nt%2)*blksize;
      yloc = 5; yoff = (1-nt%2)*blksize;
      cost += double(din[0])*din[1]*din[2]*dimout[p];
      din[p] = dimout[p];
      nt += 1;
   }
   // Ol^dagger0[br,br']: out(r,c,m) = o[d](r,x) in(x,c,m)
   if(!this->identity(0)){
      int p = 0;
      MMinfo<Tm> mm;
      mm.M = dimout[p];
      mm.N = din[1]*din[2];
      mm.K = din[p];
      mm.LDA = dagger[p]? mm.K : mm.M;
      mm.transA = dagger[p]? 'T' : 'N';
      mm.LDB = mm.K;
      mm.transB = 'N';
      mm.locA = p;    mm.offA = 0;
      mm.locB = xloc; mm.offB = xoff;
      mm.locC = yloc; mm.offC = yoff;
      MMlst.push_back(mm);
      xoff = (nt%2)*blksize;
      assert(xoff == yoff);
      cost += double(din[0])*din[1]*din[2]*dimout[p];
      din[p] = dimout[p];
   }
   assert(din[0]==dimout[0] && din[1]==dimout[1] &&
          din[2]==dimout[2]);
   offres = xoff;
}

// perform the actual matrix-matrix multiplication
template <typename Tm>
void Hxblock<Tm>::kernel(const Tm* x,
	       		 Tm* workspace){
   const Tm alpha = 1.0, beta = 0.0;
   addr[4] = const_cast<Tm*>(x);
   addr[5] = workspace;
   assert(MMlst.size()>0);
   for(int i=0; i<MMlst.size(); i++){
      const auto& mm = MMlst[i];
      Tm* Aptr = addr[mm.locA] + mm.offA;
      Tm* Bptr = addr[mm.locB] + mm.offB;
      Tm* Cptr = addr[mm.locC] + mm.offC;
      linalg::xgemm(&mm.transA, &mm.transB, &mm.M, &mm.N, &mm.K, &alpha,
		    Aptr, &mm.LDA, Bptr, &mm.LDB, &beta,
		    Cptr, &mm.M);
   }
}

} // ctns

#endif
