#include <iomanip>
#include "dvdson.h"
#include "matrix.h"
#include "linalg.h"
#include "tools.h"
#include "../settings/global.h"

using namespace std;
using namespace linalg;

// check by full diag
void dvdsonSolver::solve_diag(double* es, double* vs){
   cout << "\ndvdsonSolver:solve_diag" << endl;
   matrix id = identity_matrix(ndim);
   matrix H(ndim,ndim);
   HVecs(ndim,H.data(),id.data());
   // check symmetry
   auto sdiff = symmetric_diff(H);
   cout << "|H-H.T|=" << sdiff << endl;
   if(sdiff > 1.e-5){
      H.print("H");
      cout << "error in dvdsonSolver::solve_diag: H is not symmetric!" << endl;
      exit(1);
   }
   // eigenvalue problem
   vector<double> e(ndim);
   matrix V(H);
   eigen_solver(V,e);
   cout << "eigenvalues:\n" << setprecision(12);
   for(int i=0; i<ndim; i++){
      cout << "i=" << i << " e=" << e[i] << endl;
   }
   // copy results
   copy(e.begin(), e.begin()+neig, es);
   copy(V.data(), V.data()+ndim*neig, vs);
}

// Davidson iterative algorithm for Hv=ve
void dvdsonSolver::solve_iter(double* es, double* vs, double* vguess){
   cout << "\ndvdsonSolver::solve_iter" << endl;
   if(neig > ndim){
      cout << "error in dvdson: neig>ndim, neig/ndim=" << neig << "," << ndim << endl; 
      exit(1);
   }
   // clear counter
   nmvp = 0;
   auto t0 = global::get_time();

   // generate initial subspace - vbas
   int nl = min(ndim,neig+nbuff); // maximal subspace size
   vector<double> vbas(ndim*nl), wbas(ndim*nl);
   if(vguess != nullptr){
      copy(vguess, vguess+ndim*neig, vbas.data());
   }else{
      auto index = tools::sort_index(ndim, Diag, 1);
      for(int i=0; i<neig; i++){
	 vbas[i*ndim+index[i]] = 1.0;
      }
   }
   check_orthogonality(ndim, neig, vbas);
   HVecs(neig, wbas.data(), vbas.data());

   // Begin to solve
   vector<double> rbas(ndim*nl), tbas(ndim*nl);
   vector<double> tmpE(nl), tmpV(nl*nl), tnorm(neig);
   vector<bool> rconv(neig);
   matrix eigs(neig,maxcycle+1,1.e3), rnorm(neig,maxcycle+1); // record history
   double damp = damping;
   bool ifconv = false;
   int nsub = neig;
   for(int iter=1; iter<maxcycle+1; iter++){
     
      // solve subspace problem and form full residuals: Res[i]=HX[i]-w[i]*X[i]
      subspace_solver(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
      // check convergence
      for(int i=0; i<neig; i++){
	 auto norm = dnrm2(ndim, &rbas[i*ndim]);
	 eigs(i,iter) = tmpE[i];
	 rnorm(i,iter) = norm;
	 rconv[i] = (norm < crit_v)? true : false;
      }
      auto t1 = global::get_time();
      if(iprt > 0) print_iter(iter,nsub,eigs,rnorm,global::get_duration(t1-t0));
      t0 = global::get_time();
      
      // check convergence and return (e,v) if applied 
      ifconv = (count(rconv.begin(), rconv.end(), true) == neig);
      if(ifconv || iter == maxcycle){
	 copy(tmpE.data(), tmpE.data()+neig, es);
	 copy(vbas.data(), vbas.data()+ndim*neig, vs);
	 break;
      }
      // if not converged, improve the subspace by ri/(abs(D-ei)+damp) 
      int nres = 0;
      for(int i=0; i<neig; i++){
	 if(rconv[i]) continue;
	 transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, Diag, &tbas[nres*ndim],
		   [i,&tmpE,&damp](const double& r, const double& d){
		   return r/(abs(d-tmpE[i])+damp);});
	 tnorm[nres] = dnrm2(ndim,&tbas[nres*ndim]);
	 nres += 1;		
      }
      // *** this part is critical for better performance ***
      // ordering the residual to be added from large to small
      auto index = tools::sort_index(nres, tnorm.data());
      for(int i=0; i<nres; i++){
	 copy(&tbas[index[i]*ndim], &tbas[index[i]*ndim]+ndim, &rbas[i*ndim]); 
      }
      // re-orthogonalization and get nindp
      int nindp = gen_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
      if(nindp == 0){
	 cout << "Convergence failure: unable to generate new direction: nindp=0!" << endl;
	 exit(1);
      }else{
	 // expand V and W
         nindp = min(nindp,nbuff);
	 copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
	 HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
	 nsub = neig+nindp;
	 check_orthogonality(ndim,nsub,vbas);
      }
   } // iter
   if(!ifconv){
      cout << "convergence failure: out of maxcycle =" 
	   << maxcycle << endl;
   }
}
