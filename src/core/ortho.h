#ifndef ORTHO_H
#define ORTHO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include "tools.h"
#include "matrix.h"
#include "linalg.h"

namespace linalg{

// orthogonality of vbas(ndim,mstate) as in Fortran: ||V^+V-I||_F < thresh
template <typename Tm>
double check_orthogonality(const linalg::matrix<Tm>& V,
  		           const double thresh=1.e-10){
   int n = V.cols();
   linalg::matrix<Tm> dev = xgemm("C","N",V,V) - identity_matrix<Tm>(n);
   double diff = dev.normF()/static_cast<double>(n);
   if(diff > thresh){
      std::cout << "error in check_orthogonality: dim=" << n 
	        << " diff=" << std::scientific << diff 
		<< " thresh=" << thresh
	        << std::endl;
      dev.print("dev");
      std::exit(1);
   }
   return diff;
}
template <typename Tm>
double check_orthogonality(const int m, const int n,
   	      	           const std::vector<Tm>& vbas,
  		           const double thresh=1.e-10){
   linalg::matrix<Tm> V(m,n,vbas.data());
   return check_orthogonality(V,thresh);
}

// (1-V*V^+)*R = R-V*(V^+R)
template <typename Tm>
void ortho_projection(const int ndim,
  	      	      const int neig,
  		      const int nres,
  		      const Tm* vbas,
  		      Tm* rbas){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   std::vector<Tm> vdr(neig*nres);
   linalg::xgemm("C","N",&neig,&nres,&ndim,
	         &one,vbas,&ndim,rbas,&ndim,
	         &zero,vdr.data(),&neig); // V^+r
   linalg::xgemm("N","N",&ndim,&nres,&neig,
	         &mone,vbas,&ndim,vdr.data(),&neig,
	         &one,rbas,&ndim); // R = R-V*(V^+r)
}

// modified Gram-Schmidt orthogonalization of 
// rbas(ndim,nres) against vbas(ndim,neig)
template <typename Tm>
int get_ortho_basis(const int ndim,
  	      	    const int neig,
  		    const int nres,
  		    const Tm* vbas,
  		    Tm* rbas,
  		    const double crit_indp=1.e-12){
   const int maxtimes = 2;
   // 1. projection (1-V*V^+)*R = R-V*(V^+R)
   for(int repeat=0; repeat<maxtimes; repeat++){
      ortho_projection(ndim,neig,nres,vbas,rbas);
   }
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
         linalg::xscal(ndim, 1.0/rii, &rbas[i*ndim]);
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      linalg::xcopy(ndim, &rbas[i*ndim], &rbas[nindp*ndim]); 
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int nleft = nres-1-i;
      if(nleft == 0) break;
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-V*V^+)*R_rest
	 ortho_projection(ndim,neig,nleft,vbas,&rbas[(i+1)*ndim]);
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         ortho_projection(ndim,nindp,nleft,&rbas[0],&rbas[(i+1)*ndim]);
      } // repeat
   } // i
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm>
int get_ortho_basis(const int ndim,
		    const int nres,
		    Tm* rbas,
		    const double crit_indp=1.e-12){
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
         linalg::xscal(ndim, 1.0/rii, &rbas[i*ndim]);
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      linalg::xcopy(ndim, &rbas[i*ndim], &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int nleft = nres-1-i;
      if(nleft == 0) break;
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // R_rest = (1-Rnew*Rnew^+)*R_rest
         ortho_projection(ndim,nindp,nleft,&rbas[0],&rbas[(i+1)*ndim]); 
      } // repeat
   } // i
   return nindp;
}

} // linalg

#endif
