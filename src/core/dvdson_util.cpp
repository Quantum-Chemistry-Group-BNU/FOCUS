#include "dvdson.h"
#include "linalg.h"
#include <iomanip>
#include <string>
#include "../settings/global.h"

using namespace std;
using namespace linalg;

// perform H*x for a set of input vectors
void dvdsonSolver::HVecs(const int nstate, double* y, const double* x){
   auto t0 = global::get_time();
   for(int istate=0; istate<nstate; istate++){
      HVec(y+istate*ndim, x+istate*ndim); // y=H*x
   }   
   nmvp += nstate;
   auto t1 = global::get_time();
   cout << "timing for HVecs=" << setprecision(2)  
        << global::get_duration(t1-t0)/nstate << " s" << endl;
}

void dvdsonSolver::subspace_solver(const int ndim, 
				   const int nsub,
				   const int nt,
	       			   vector<double>& vbas,
	       			   vector<double>& wbas,
	       			   vector<double>& tmpE,
	       			   vector<double>& tmpV,
	       			   vector<double>& rbas){
   // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
   double alpha = 1.0, beta=0.0;
   matrix tmpH(nsub,nsub);
   ::dgemm_("T","N",&nsub,&nsub,&ndim,
            &alpha,vbas.data(),&ndim,wbas.data(),&ndim,
	    &beta,tmpH.data(),&nsub);
   // 2. check symmetry property
   double diff = symmetric_diff(tmpH);
   if(diff > 1.e-10){
      cout << "error in dvdsonSolver::subspace_solver: diff_skewH=" 
	   << diff << endl;
      tmpH.print("tmpH");
      exit(1);
   }
   // 3. solve eigenvalue problem
   eigen_solver(tmpH, tmpE);
   copy(tmpH.data(), tmpH.data()+nsub*nt, tmpV.data());
   // 4. form full residuals: Res[i]=HX[i]-e[i]*X[i]
   // vbas = X[i]
   copy(vbas.data(), vbas.data()+ndim*nsub, rbas.data()); 
   ::dgemm_("N","N",&ndim,&nt,&nsub,
            &alpha,rbas.data(),&ndim,tmpV.data(),&nsub,
	    &beta,vbas.data(),&ndim);
   // wbas = HX[i]
   copy(wbas.data(), wbas.data()+ndim*nsub, rbas.data()); 
   ::dgemm_("N","N",&ndim,&nt,&nsub,
	    &alpha,rbas.data(),&ndim,tmpV.data(),&nsub,
	    &beta,wbas.data(),&ndim);
   // rbas = HX[i]-e[i]*X[i]
   for(int i=0; i<nt; i++){
      transform(&wbas[i*ndim],&wbas[i*ndim]+ndim,&vbas[i*ndim],&rbas[i*ndim],
	        [i,&tmpE](const double& w, const double& x){ return w-x*tmpE[i]; }); 
   }
}

void dvdsonSolver::print_iter(const int iter,
			      const int nsub,
			      const matrix& eigs,
			      const matrix& rnorm,
			      const double t){
   string ifconverge = "-+";
   if(iter == 1){
      cout << defaultfloat; 
      cout << "settings: ndim=" << ndim 
	   << " neig=" << neig
	   << " nbuff=" << nbuff  
	   << " maxcycle= " << maxcycle << endl;
      cout << "          damping=" << damping 
	   << " crit_v=" << crit_v 
           << " crit_e=" << crit_e 
	   << " crit_indp=" << crit_indp << endl;
      cout << "iter   ieig        eigenvalue        ediff      rnorm   nsub  nmvp  time/s" << endl;
   }
   for(int i=0; i<neig; i++){
      cout << setw(5) << iter << " " 
           << setw(3) << i << " "
  	   << setw(1) << ifconverge[(rnorm(i,iter)<crit_v)] << " "
  	   << setw(20) << setprecision(12) << fixed << eigs(i,iter) << " "
  	   << setw(10) << setprecision(2) << scientific << eigs(i,iter)-eigs(i,iter-1) << " "
  	   << setw(10) << setprecision(2) << scientific << rnorm(i,iter) << " "
    	   << setw(4) << nsub << " " 
  	   << setw(5) << nmvp << " "
  	   << setw(10) << setprecision(2) << scientific << t << endl;
   } // i 
}

// column major matrix - vbas(ndim,mstate) as in Fortran
void dvdsonSolver::check_orthogonality(const int n, const int m, 
				       const vector<double>& vbas, 
				       const double thresh){
   matrix V(n,m,vbas.data());
   matrix Vt = transpose(V);
   matrix dev = dgemm("N","N",Vt,V) - identity_matrix(m);
   double diff = normF(dev);
   if(iprt > 1) cout << "dvdsonSolver::check_orthogonality diff=" << diff << endl;
   if(diff > thresh){
      cout << "error: deviation from orthonormal basis exceed thresh=" 
	   << thresh << endl;      
      V.print("V");
      dev.print("dev");
   }
}

// modified Gram-Schmidt orthogonalization - vbas(ndim,neig), rbas(ndim,nres) 
int dvdsonSolver::gen_ortho_basis(const int ndim,
		    		  const int neig,
		    		  const int nres,
		    		  const vector<double>& vbas,
		    		  vector<double>& rbas,
		    		  const double crit_indp){
   double one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 1. projection (1-V*V^+)*R = R-V*(V^+R)
   vector<double> vtr(neig*nres);
   for(int repeat=0; repeat<maxtimes; repeat++){
      ::dgemm_("T","N",&neig,&nres,&ndim,
	       &one,vbas.data(),&ndim,rbas.data(),&ndim,
	       &zero,vtr.data(),&neig);
      ::dgemm_("N","N",&ndim,&nres,&neig,
	       &mone,vbas.data(),&ndim,vtr.data(),&neig,
	       &one,rbas.data(),&ndim);
   };
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   vector<double> rtr(nres*nres/4);
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = dnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
         transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		   [rii](const double& x){ return x/rii; });
         rii = dnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) continue;
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-V*V^+)*R_rest
         ::dgemm_("T","N",&neig,&N,&ndim,
                  &one,vbas.data(),&ndim,&rbas[(i+1)*ndim],&ndim,
                  &zero,vtr.data(),&neig);
         ::dgemm_("N","N",&ndim,&N,&neig,
                  &mone,vbas.data(),&ndim,vtr.data(),&neig,
                  &one,&rbas[(i+1)*ndim],&ndim);
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         ::dgemm_("T","N",&nindp,&N,&ndim,
                  &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
        	  &zero,rtr.data(),&nindp);
         ::dgemm_("N","N",&ndim,&N,&nindp,
                  &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                  &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}

int dvdsonSolver::gen_ortho_basis(const int ndim,
		    		  const int nres,
		    		  vector<double>& rbas,
		    		  const double crit_indp){
   double one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   vector<double> rtr(nres*nres/4);
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = dnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
         transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		   [rii](const double& x){ return x/rii; });
         rii = dnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) continue;
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         ::dgemm_("T","N",&nindp,&N,&ndim,
                  &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
        	  &zero,rtr.data(),&nindp);
         ::dgemm_("N","N",&ndim,&N,&nindp,
                  &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                  &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}
