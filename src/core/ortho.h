#ifndef ORTHO_H
#define ORTHO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include "tools.h"
#include "matrix.h"
#include "linalg.h"

namespace linalg{

// orthogonality of vbas(ndim,mstate) as in Fortran
template <typename Tm>
double check_orthogonality(const linalg::matrix<Tm>& V,
  		           const double thresh=1.e-10){
   int n = V.cols();
   linalg::matrix<Tm> dev = xgemm("C","N",V,V) - identity_matrix<Tm>(n);
   double diff = dev.normF()/static_cast<double>(n);
   if(diff > thresh){
      //dev.print("dev");
      std::cout << "error in check_orthogonality: dim=" << n 
	        << " diff=" << std::scientific << diff 
		<< " thresh=" << thresh
	        << std::endl;
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

// modified Gram-Schmidt orthogonalization of 
// rbas(ndim,nres) against vbas(ndim,neig)
template <typename Tm>
int get_ortho_basis(const int ndim,
  	      	    const int neig,
  		    const int nres,
  		    const std::vector<Tm>& vbas,
  		    std::vector<Tm>& rbas,
  		    const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 1. projection (1-V*V^+)*R = R-V*(V^+R)
   std::vector<Tm> vtr(neig*nres);
   for(int repeat=0; repeat<maxtimes; repeat++){
      linalg::xgemm("C","N",&neig,&nres,&ndim,
	            &one,vbas.data(),&ndim,rbas.data(),&ndim,
	            &zero,vtr.data(),&neig);
      linalg::xgemm("N","N",&ndim,&nres,&neig,
	            &mone,vbas.data(),&ndim,vtr.data(),&neig,
	            &one,rbas.data(),&ndim);
   }
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      // std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      linalg::xcopy(ndim, &rbas[i*ndim], &rbas[nindp*ndim]); 
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-V*V^+)*R_rest
	 linalg::xgemm("C","N",&neig,&N,&ndim,
                       &one,vbas.data(),&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,vtr.data(),&neig);
         linalg::xgemm("N","N",&ndim,&N,&neig,
                       &mone,vbas.data(),&ndim,vtr.data(),&neig,
                       &one,&rbas[(i+1)*ndim],&ndim);
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         linalg::xgemm("C","N",&nindp,&N,&ndim,
                       &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,rtr.data(),&nindp);
         linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm>
int get_ortho_basis(const int ndim,
		    const int nres,
		    std::vector<Tm>& rbas,
		    const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      // std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      linalg::xcopy(ndim, &rbas[i*ndim], &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-Rnew*Rnew^+)*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
               	       &zero,rtr.data(),&nindp);
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}

// Plain version for matrix
template <typename Tm>
int get_ortho_basis(linalg::matrix<Tm>& rbas,
	            const int nres,	
		    const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   int ndim = rbas.rows(), nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, rbas.col(i)); // normalization constant
      // std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(rbas.col(i), rbas.col(i)+ndim, rbas.col(i),
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, rbas.col(i));
      }
      // copy
      linalg::xcopy(ndim, rbas.col(i), rbas.col(nindp));
      nindp += 1;
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      // R_rest = (1-Rnew*Rnew^+)*R_rest
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // rtr = Rnew^+*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,rbas.col(0),&ndim,rbas.col(i+1),&ndim,
               	       &zero,rtr.data(),&nindp);
	 // R_rest -= Rnew*rtr
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,rbas.col(0),&ndim,rtr.data(),&nindp,
                       &one,rbas.col(i+1),&ndim);
      } // repeat
   } // i
   return nindp;
}

} // linalg

#endif
