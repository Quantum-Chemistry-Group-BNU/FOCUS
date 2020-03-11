#ifndef DVDSON_H
#define DVDSON_H

#include <vector>
#include <functional> // for std::function
#include "matrix.h"

namespace linalg{

struct dvdsonSolver{
   public:
      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, double* y, const double* x); 
      // check by full diag
      void solve_diag(double* es, double* vs);
      // Davidson iterative algorithm for Hv=ve 
      void solve_iter(double* es, double* vs, double* vguess=nullptr);
      // subspace problem
      void subspace_solver(const int ndim, 
		      	   const int nsub,
		      	   const int nt,
      	       		   std::vector<double>& vbas,
      	       		   std::vector<double>& wbas,
      	       		   std::vector<double>& tmpE,
      	       		   std::vector<double>& tmpV,
			   std::vector<double>& rbas);
      // iteration info
      void print_iter(const int iter,
		      const int nsub,
		      const linalg::matrix& eigs,
		      const linalg::matrix& rnorm,
		      const double t);
      // orthogonality of vbas
      void check_orthogonality(const int n, const int m, 
		      	       const std::vector<double>& vbas,
			       const double thresh=1.e-10);
      // modified Gram-Schmidt orthogonalization 
      int gen_ortho_basis(const int ndim,
		      	  const int neig,
			  const int nres,
			  const std::vector<double>& vbas,
			  std::vector<double>& rbas,
			  const double crit_indp);
      int gen_ortho_basis(const int ndim,
		          const int nres,
		          std::vector<double>& rbas,
		          const double crit_indp);
   public:
      int ndim = 0;
      int neig = 0;
      double* Diag;
      std::function<void(double*, const double*)> HVec;
      // parameters
      int iprt = 1;
      double crit_v = 1.e-6;  // used 
      double crit_e = 1.e-12; // not used
      double crit_indp = 1.e-12;
      double damping = 1.e-1; 
      int maxcycle = 500;
      int nbuff = 3; // maximal additional vectors
      int nmvp = 0;
};

}

#endif
