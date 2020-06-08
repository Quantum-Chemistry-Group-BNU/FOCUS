#ifndef VMC_H
#define VMC_H

#include "../core/onstate.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/integral.h"
#include "../io/input.h"
#include <vector>
#include <string>
#include <cmath>

namespace vmc{

struct vecmat{
   public:
      // constructor
      vecmat(const int n, const int D0, const std::string& s): D(D0){
	 site.resize(n);
	 for(int i=0; i<n; i++){
	    if(s == "random"){
	       site[i] = linalg::random_matrix(D,D);
	    }else if(s == "zero"){
	       site[i] = linalg::zero_matrix(D,D);
	    }
	 }
      }
      // member functions
      int size() const{ return site.size(); }
      double norm() const{
	 double tmp = 0.0;
	 for(int i=0; i<site.size(); i++){
	    tmp += pow(linalg::normF(site[i]),2);
	 }
	 return sqrt(tmp);
      }
      // useful operations
      vecmat& operator +=(const vecmat& tmp){
	 for(int i=0; i<site.size(); i++){
	    site[i] += tmp.site[i];
	 }
	 return *this;
      }
      vecmat& operator -=(const vecmat& tmp){
	 for(int i=0; i<site.size(); i++){
	    site[i] -= tmp.site[i];
	 }
	 return *this;
      }
      vecmat& operator *=(const double fac){
	 for(int i=0; i<site.size(); i++){
	    site[i] *= fac;
	 }
	 return *this;
      }
      // friend
      friend vecmat operator *(const double fac, const vecmat& tmp);
      friend vecmat operator *(const vecmat& tmp, const double fac);
      friend vecmat operator +(const vecmat& tmp1, const vecmat& tmp2);
      friend vecmat operator -(const vecmat& tmp1, const vecmat& tmp2);
   public:
      int D;
      std::vector<linalg::matrix> site;
};

// Ansatz |Psi>=tr(A[p1]*...*A[pn])|p1...pn> ~ O(KD^2)
double get_Ws(const fock::onstate& state,
	      const vecmat& mps);

// local energy E[n] = <n|H|Psi>/<n|Psi> and gradient info <n|Psi_a>/<n|Psi>
void local_egrad(const fock::onstate& state,
	         const std::vector<int>& olst,
	         const std::vector<int>& vlst,
	         const vecmat& mps,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
	         const double ecore,
	         double& wt,
	         double& elocal,
	         vecmat& delta);

// VMC estimate of energy and gradients
void estimate_egrad(const int mcmc,
		    const fock::onstate& seed,
		    const vecmat& mps,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
		    const double ecore,
		    double& emps,
		    vecmat& grad);

// update MPS
void update_mps(const double step,
		const vecmat& grad,
		vecmat& mps);

// optimize
void solver(const input::schedule& schd,
	    const integral::two_body& int2e,
	    const integral::one_body& int1e,
	    const double ecore);

} // vmc

#endif
