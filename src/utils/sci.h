#ifndef SCI_H
#define SCI_H

#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"
#include <vector>
#include <map>

namespace sci{

// represent space of dets by direct product structure
struct product_space{
   public:
      // constructor	   
      product_space(const fock::onspace& space);
   public:
      // second int is used for indexing in constructing bsetA, asetB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace usetA, usetB; 
      std::vector<std::vector<int>> dsetA, bsetA, dsetB, asetB; 
      std::vector<int> nnzA, nnzB;
      // dpt - a table to store the set of {Det} in direct product space
      //       |  0  1  2 ... dimB
      // --------------------------
      //   0   |  -1 -1  1      3    nnzA,bsetA,dsetA = (nnz,col,val) per row
      //   1   |  -1  4 -1      6    nnzB,asetB,dsetB = (nnz,row,val) per col
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
      coupling_table(const fock::onspace& uset);
   public:
      int dim;
      std::vector<std::vector<int>> C11;      // <I|p^+q|J> 
      std::vector<std::vector<int>> C22;      // <I|p^+p^+rs|J> 
      /*
      std::vector<std::vector<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::vector<int>> C21, C12; // <I|p^+q^+r|J>, <I|p^+rs|J>
      std::vector<std::vector<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
      */
};

// linked list - store each row H[o] as a list
struct sparse_hamiltonian{
   public:
      sparse_hamiltonian(const fock::onspace& space,
		         const product_space& pspace,
		         const coupling_table& ctableA,
			 const coupling_table& ctableB,
			 const integral::two_body& int2e,
			 const integral::one_body& int1e,
			 const double ecore);
      void debug(const fock::onspace& space,
	 	 const integral::two_body& int2e,
		 const integral::one_body& int1e);
      void get_diagonal(const fock::onspace& space,
	 	        const integral::two_body& int2e,
		        const integral::one_body& int1e,
		        const double ecore);
      void get_localA(const fock::onspace& space,
		      const product_space& pspace,
		      const coupling_table& ctabA,
		      const integral::two_body& int2e,
		      const integral::one_body& int1e);
      void get_localB(const fock::onspace& space,
		      const product_space& pspace,
		      const coupling_table& ctabB,
		      const integral::two_body& int2e,
		      const integral::one_body& int1e);
      void get_int_11_11(const fock::onspace& space,
		         const product_space& pspace,
		         const coupling_table& ctabA,
		         const coupling_table& ctabB,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e);
   public:
      int dim;
      std::vector<std::vector<int>> connect;  // TODOs:
      std::vector<std::vector<double>> value; // this could be merged into a pair!
      std::vector<int> nnz;
};

// sci
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

} // sci

#endif
