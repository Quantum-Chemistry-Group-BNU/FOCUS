#ifndef FCI_H
#define FCI_H

#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"
#include <string>
#include <vector>
#include <map>
#include <set>

namespace fci{

// represent space of dets by direct product structure
struct product_space{
   public:
      // constructor	   
      product_space(const fock::onspace& space);
   public:
      // second int is used for indexing in constructing bsetA, asetB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dimA, dimB;
};

// compute coupling of states:
// basically describe how two states are differ defined by diff_type,
// which partition the cartesian space (I,J) into disjoint subspace!
struct coupling_table{
   public:
      void get_C11(const fock::onspace& space);
   public:
      std::vector<std::set<int>> C11; // differ by single (sorted, binary_search)
      /*
      std::vector<std::set<int>> C22; // differ by double
      std::vector<std::set<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::set<int>> C21, C12; // <I|p^+q^+r|J>, <I|p^+rs|J>
      std::vector<std::set<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
      */
};

// linked list - store each row H[i] as a list
struct sparse_hamiltonian{
   public:
      void get_hamiltonian(const fock::onspace& space,
		           const product_space& pspace,
		           const coupling_table& ctabA,
			   const coupling_table& ctabB,
			   const integral::two_body& int2e,
			   const integral::one_body& int1e,
			   const double ecore);
      void debug(const fock::onspace& space,
	 	 const integral::two_body& int2e,
		 const integral::one_body& int1e);
      // save for analysis
      void save_gephi(const std::string& fname,
		      const fock::onspace& space);
      void save_text(const std::string& fname);
   public:
      int dim;
      std::vector<double> diag; // H[i,i]
      std::vector<std::vector<int>> connect; // i<->j (i<j) connected by H
      std::vector<std::vector<double>> value; // H[i][j] (i<j)
      std::vector<std::vector<long>> diff; // packed orbital difference 
};

// matrix-vector product using stored H
void get_Hx(double* y,
	    const double* x,
	    const sparse_hamiltonian& sparseH);

// initial guess
void get_initial(const fock::onspace& space,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
		 const double ecore,
		 std::vector<double>& Diag,
		 linalg::matrix& v0);

// fci
void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	
	       sparse_hamiltonian& sparseH,
	       const fock::onspace& space,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

// without sparseH as output
void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	
	       const fock::onspace& space,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

} // fci

#endif
