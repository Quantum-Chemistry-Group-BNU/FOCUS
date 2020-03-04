#ifndef DVDSON_H
#define DVDSON_H

#include <vector>
#include <functional> // for std::function

namespace linalg{

struct dvdsonSolver{
   public:
      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, double* y, const double* x); 
      // check by full diag
      void full_diag(double* es, double* vs);
      // Davidson iterative algorithm for Hv=ve 
      void solve_iter(double* es, double* vs, double* vguess=nullptr);
      // subspace problem
      void subspace_solver(const int nsub, const int ndim,
      	       		   const std::vector<double>& vbas,
      	       		   const std::vector<double>& wbas,
      	       		   std::vector<double>& tmpH,
      	       		   std::vector<double>& tmpV,
      			   std::vector<double>& tmpE);
      // modified Gram-Schmidt orthogonalization for vbas(n,m)
      void check_orthogonality(const int n, const int m, 
		      	       const double* vbas,
			       const double thresh=1.e-10);
      //void gen_ortho_basis(matrix& rbas, const matrix& vbas, const double cindp); 
   public:
      int iprt = 1;
      int ndim = 0;
      int neig = 0;
      double cnst = 0.0;
      double* Diag;
      std::function<void(double*, const double*)> HVec;
      // parameters
      double crit_e = 1.e-8;
      double crit_v = 1.e-8;
      double crit_indp = 1.e-12;
      double lshift = 1.e-4;
      int maxcycle = 50;
      int nbuff = 3; // additional subspace size
      int nmvp = 0;
};	

}

#endif
