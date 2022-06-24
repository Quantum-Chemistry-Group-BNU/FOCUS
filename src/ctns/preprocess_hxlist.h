#ifndef PREPROCESS_HXLIST_H
#define PREPROCESS_HXLIST_H

namespace ctns{

// information for sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
// 					Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv']
// 					wf[br',bc',bm',bv']
template <typename Tm>
struct Hxblock{
public:
   void display() const;
   Tm* kernel(const Tm* x,
	      const size_t& blksize,
	      Tm** qops_addr,
	      Tm* workspace) const;
public:
   bool dagger[4] = {false,false,false,false};
   int location[4] = {-1,-1,-1,-1}; 
   size_t offop[4] = {0,0,0,0};
   size_t dimout[4] = {0,0,0,0};
   size_t dimin[4] = {0,0,0,0};
   size_t offout = 0, offin = 0, size = 0;
   Tm coeff = 1.0;
};
template <typename Tm>
using Hxlist = std::vector<Hxblock<Tm>>;  

template <typename Tm>
void Hxblock<Tm>::display() const{
   std::cout << "offout=" << offout << " size=" << size
	     << " dimout=" << dimout[0] << "," << dimout[1] << "," << dimout[2] << "," << dimout[3]
	     << " offin=" << offin 
	     << " dimin=" << dimin[0] << "," << dimin[1] << "," << dimin[2] << "," << dimin[3]
	     << " coeff=" << coeff
	     << " dagger=" << dagger[0] << "," << dagger[1] << "," << dagger[2] << "," << dagger[3]
	     << " location=" << location[0] << "," << location[1] << "," << location[2] << "," << location[3]
	     << " offop=" << offop[0] << "," << offop[1] << "," << offop[2] << "," << offop[3]
	     << std::endl;  
}

// compute sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
// 			        Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv']
// 			        wf[br',bc',bm',bv']
template <typename Tm>
Tm* Hxblock<Tm>::kernel(const Tm* x,
			const size_t& blksize,
	                Tm** qops_addr,
	       		Tm* workspace) const{
   const Tm alpha = 1.0, beta = 0.0;
   Tm *optr, *xptr, *yptr;
   int din[4];
   int nt = 0;
   // wf[br',bc',bm',bv']
   yptr = workspace;
   xptr = const_cast<Tm*>(x)+offin;
   din[0] = dimin[0];
   din[1] = dimin[1];
   din[2] = dimin[2];
   din[3] = dimin[3];
   // Oc2^dagger3[bv,bv']: out(r,c,m,v) = o(v,x) in(r,c,m,x) 
   if(location[3] != -1){
      optr = qops_addr[location[3]]+offop[3];
      int vdim = dimout[3];
      int xdim = din[3];
      const char* transb = dagger[3]? "N" : "T";
      int LDB = dagger[3]? xdim : vdim; // o(x,v) or o(v,x)
      int rcmdim = din[0]*din[1]*din[2];
      linalg::xgemm("N", transb, &rcmdim, &vdim, &xdim, &alpha,
		    xptr, &rcmdim, optr, &LDB, &beta,
		    yptr, &rcmdim);
      xptr = workspace+(nt%2)*blksize;
      yptr = workspace+(1-nt%2)*blksize;
      din[3] = dimout[3];
      nt += 1;	  
   }
   // Oc1^dagger2[bm,bm']: out(r,c,m,v) = o(m,x) in(r,c,x,v)
   if(location[2] != -1){
      optr = qops_addr[location[2]]+offop[2];
      int mdim = dimout[2];
      int xdim = din[2];
      const char* transb = dagger[2]? "N" : "T";
      int LDB = dagger[2]? xdim : mdim;
      int vdim = din[3];
      int rcdim = din[0]*din[1];
      int rcmdim = rcdim*mdim;
      int rcxdim = rcdim*xdim;
      for(int iv=0; iv<vdim; iv++){
         Tm* xptr_iv = xptr + iv*rcxdim;
	 Tm* yptr_iv = yptr + iv*rcmdim;
	 linalg::xgemm("N", transb, &rcdim, &mdim, &xdim, &alpha,
                       xptr_iv, &rcdim, optr, &LDB, &beta,
                       yptr_iv, &rcdim);
      }
      xptr = workspace+(nt%2)*blksize;
      yptr = workspace+(1-nt%2)*blksize;
      din[2] = dimout[2];
      nt += 1;	   
   }
   // Or^dagger1[bc,bc']: out(r,c,m,v) = o(c,x) in(r,x,m,v) 
   if(location[1] != -1){
      optr = qops_addr[location[1]]+offop[1];
      int cdim = dimout[1];
      int xdim = din[1];
      const char* transb = dagger[1]? "N" : "T";
      int LDB = dagger[1]? xdim : cdim;
      int rdim = din[0];
      int rcdim = rdim*cdim;
      int rxdim = rdim*xdim;
      int mdim = din[2];
      int vdim = din[3];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
	    Tm* xptr_imv = xptr + (iv*mdim+im)*rxdim;
	    Tm* yptr_imv = yptr + (iv*mdim+im)*rcdim;
            linalg::xgemm("N", transb, &rdim, &cdim, &xdim, &alpha,
                          xptr_imv, &rdim, optr, &LDB, &beta,
                          yptr_imv, &rdim);
	 }
      }
      xptr = workspace+(nt%2)*blksize;
      yptr = workspace+(1-nt%2)*blksize;
      din[1] = dimout[1];
      nt += 1;	  
   }
   // Ol^dagger0[br,br']: out(r,c,m,v) = o(r,x) in(x,c,m,v)
   if(location[0] != -1){
      optr = qops_addr[location[0]]+offop[0];
      int rdim = dimout[0];
      int xdim = din[0];
      const char* transa = dagger[0]? "T" : "N";
      int LDA = dagger[0]? xdim : rdim;
      int cmvdim = din[1]*din[2]*din[3];
      linalg::xgemm(transa, "N", &rdim, &cmvdim, &xdim, &alpha,
                    optr, &LDA, xptr, &xdim, &beta,
                    yptr, &rdim);
      xptr = yptr;
      din[0] = dimout[0];
      nt += 1;
   }
   assert(din[0]==dimout[0] && din[1]==dimout[1] &&
	  din[2]==dimout[2] && din[3]==dimout[3]);
   return xptr;
}

} // ctns

#endif
