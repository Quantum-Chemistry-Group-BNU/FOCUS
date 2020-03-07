#ifndef SCI_H
#define SCI_H

#include <vector>
#include <map>
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"

namespace sci{

// represent space of dets by direct product structure
struct product_space{
   public:
      // constructor	   
      product_space(const fock::onspace& space);
      // debug
      void print();
   public:
      // second int is used for indexing in constructing BsetA, AsetB 
      std::map<fock::onstate,int> UsetA, UsetB; 
      std::vector<std::vector<int>> DsetA, BsetA;
      std::vector<std::vector<int>> DsetB, AsetB;
};

// compute coupling of states:
// basically describe how two states are differ defined by diff_type,
// which partition the cartesian space (I,J) into disjoint subspace!
struct coupling_table{
   public:
      // constructor	   
      coupling_table(const std::map<fock::onstate,int>& Uset);
   public:
      int dim;
      std::vector<std::vector<int>> C11;      // <I|p^+q|J> 
      std::vector<std::vector<int>> C22;      // <I|p^+p^+rs|J> 
      std::vector<std::vector<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::vector<int>> C21, C12; // <I|p^+q^+r|J>, <I|p^+rs|J>
      std::vector<std::vector<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
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
   public:
      size_t dim;
      vector<vector<int>> connect;
      vector<vector<double>> value;
      vector<int> nnz;
};

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

} // sci

#endif
