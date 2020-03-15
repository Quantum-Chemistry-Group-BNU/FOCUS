#ifndef FCI_H
#define FCI_H

#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"
#include <vector>
#include <tuple>
#include <map>

namespace fci{

// represent space of dets by direct product structure
struct product_space{
   public:
      // constructor	   
      product_space(const fock::onspace& space);
   public:
      // second int is used for indexing in constructing bsetA, asetB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB;
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      // dpt - a table to store the set of {Det} in direct product space
      //       |  0  1  2 ... dimB
      // --------------------------
      //   0   |  -1 -1  1      3    rowA = (colIndex,val) per row for val != -1
      //   1   |  -1  4 -1      6    colB = (rowIndex,val) per col for val != -1
      //   2   |   0  5  9     10
      //   .   | 
      //  dimA |   2  7  8     11  
      int dimA, dimB;
      std::vector<std::vector<int>> dpt;
};

// compute coupling of states:
// basically describe how two states are differ defined by diff_type,
// which partition the cartesian space (I,J) into disjoint subspace!
struct coupling_table{
   public:
      // constructor	   
      coupling_table(const std::map<fock::onstate,int>& umap);
   public:
      int dim;
      std::vector<std::vector<int>> C11; // differ by single
      std::vector<std::vector<int>> C22; // differ by double
      /*
      std::vector<std::vector<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::vector<int>> C21, C12; // <I|p^+q^+r|J>, <I|p^+rs|J>
      std::vector<std::vector<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
      */
};

// linked list - store each row H[i] as a list
struct sparse_hamiltonian{
   public:
      sparse_hamiltonian(const fock::onspace& space,
		         const product_space& pspace,
		         const coupling_table& ctabA,
			 const coupling_table& ctabB,
			 const integral::two_body& int2e,
			 const integral::one_body& int1e,
			 const double ecore);
      void debug(const fock::onspace& space,
	 	 const integral::two_body& int2e,
		 const integral::one_body& int1e);
   public:
      int dim;
      std::vector<double> diag; // H[i,i]
      std::vector<std::vector<std::tuple<int,double,size_t>>> connect; // H[i][j] (i<j) 
};

// fci
void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	
	       const fock::onspace& space,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

// matrix-vector product using stored H
void get_Hx(double* y,
	    const double* x,
	    const sparse_hamiltonian& sparseH);

// initial guess
void get_initial(const fock::onspace& space,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
		 const double ecore,
		 vector<double>& Diag,
		 linalg::matrix& v0);

} // fci

#endif
