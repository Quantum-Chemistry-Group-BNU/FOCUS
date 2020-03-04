#include "dvdson.h"
#include "linalg.h"

using namespace std;
using namespace linalg;

// perform H*x for a set of input vectors
void dvdsonSolver::HVecs(const int nstate, double* y, const double* x){
   for(int istate=0; istate<nstate; istate++){
      HVec(y+istate*ndim, x+istate*ndim);
   }   
   nmvp += nstate;
}

void dvdsonSolver::subspace_solver(const int nsub, const int ndim,
	       			   const vector<double>& vbas,
	       			   const vector<double>& wbas,
	       			   vector<double>& tmpH,
	       			   vector<double>& tmpV,
				   vector<double>& tmpE){
   // 1. form H in the subspace
   // tmpH = numpy.dot(vbas.conj(),wbas.T)
   double alpha = 1.0, beta=0.0;
   ::dgemm_("N","T",&nsub,&nsub,&ndim,
            &alpha,vbas.data(),&nsub,wbas.data(),&nsub,
	    &beta,tmpH.data(),&nsub);
   for(auto dat : tmpH){
      cout << dat << endl;
   }
   // 2. check symmetry property
   matrix Hm(nsub,nsub,tmpH.data());
   double diff = symmetric_diff(Hm);
   if(diff > 1.e-10){
      cout << "error in dvdsonSolver::subspace_solver: diff_skewH=" << diff << endl;
      Hm.print("Hm");
      exit(1);
   }
   // 3. solve eigenvalue problem
   eigen_solver(Hm, tmpE);
   //fill(Hm.data(), Hm.data()+neig*nsub, tmpV.data());
   cout << tmpE[0] << endl;
}

// column major matrix - vbas(ndim,mstate) as in Fortran
void dvdsonSolver::check_orthogonality(const int n, const int m, 
				       const double* vbas, 
				       const double thresh){
   matrix V(n,m,vbas);
   matrix Vt = transpose(V);
   matrix dev = mdot(Vt,V) - identity_matrix(m);
   double diff = normF(dev);
   cout << "dvdsonSolver::check_orthogonality diff=" << diff << endl;
   if(diff > thresh){
      cout << "error: deviation from orthonormal basis exceed thresh=",thresh      
      V.print("V");
      dev.print("dev");
   }
}
