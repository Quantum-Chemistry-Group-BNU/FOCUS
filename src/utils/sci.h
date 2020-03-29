#ifndef SCI_H
#define SCI_H

#include "../core/integral.h"
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../io/input.h"
#include "fci.h"
#include <unordered_set>
#include <functional>
#include <vector>
#include <map>

namespace sci{

/*
template <typename T>
class greater_abs{
  public:
    bool operator()(const T& a, const T& b) const { return abs(a) > abs(b); }
};
*/

// t[pq](r,s)=<pq||rs> for p>q, r>s
struct heatbath_table{
public: 
   heatbath_table(const integral::two_body& int2e,
		  const integral::one_body& int1e);
public:
   int sorb;
   // sorted by magnitude Iij=<ij||kl> (i>j,k>l)
   double thresh = 1.e-14; // cut-off value 
   std::vector<std::multimap<float,int,std::greater<float>>> eri4; 
   // Iik[j]=<ij||kj> (i>=k) for singles
   std::vector<std::vector<float>> eri3; 
};

// expand variational subspace
void expand_varSpace(fock::onspace& space, 
		     std::unordered_set<fock::onstate>& varSpace, 
		     const heatbath_table& hbtab, 
		     std::vector<double>& cmax, 
		     const double eps1);

// prepare intial solution
void get_initial(std::vector<double>& e, 
		 linalg::matrix& v,
		 fock::onspace& space,
	         std::unordered_set<fock::onstate>& varSpace,
		 const heatbath_table& hbtab, 
		 const input::schedule& schd, 
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
	         const double ecore);

// sci
void ci_solver(const input::schedule& schd,
	       fci::sparse_hamiltonian& sparseH,
	       std::vector<double>& es,
	       std::vector<std::vector<double>>& vs,
	       fock::onspace& space,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

} // sci

#endif
