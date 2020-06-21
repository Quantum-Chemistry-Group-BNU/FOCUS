#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "fci_util.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;

// constructor
void product_space::get_pspace(const onspace& space,
		 	       const int istart){
   dimA0 = spaceA.size();
   dimB0 = spaceB.size();
   int udxA = dimA0, udxB = dimB0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   int dim = space.size();
   for(int i=istart; i<dim; i++){
      // even bits 
      onstate strA = space[i].get_even();
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
	 spaceA.push_back(strA);
         auto pr = umapA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 rowA.resize(udxA); // reserve additional space for the new row
      };
      // odd bits
      onstate strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
	 spaceB.push_back(strB);
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 colB.resize(udxB);
      }
      rowA[itA->second].emplace_back(itB->second,i);
      colB[itB->second].emplace_back(itA->second,i);
   }
   assert(udxA == spaceA.size());
   assert(udxB == spaceB.size());
   dimA = udxA;
   dimB = udxB;
}

// coupling_table
void coupling_table::get_Cmn(const onspace& space,
			     const bool Htype,
			     const int istart){
   auto t0 = tools::get_time();
   int avg = 0;
   int dim = space.size();
   // C11
   C11.resize(dim);
   for(int i=0; i<dim; i++){
      if(i<istart){
         for(int j=istart; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == make_pair(1,1)) C11[i].insert(j);
         }
      }else{
         for(int j=0; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == make_pair(1,1)) C11[i].insert(j);
         }
      }
      avg += C11[i].size();
   }
   // relativistic Hamiltonian
   if(Htype){
      
/*  
      C11.resize(dim);
      for(int i=0; i<dim; i++){
         if(i<istart){
            for(int j=istart; j<dim; j++){
               auto pr = space[i].diff_type(space[j]);
               if(pr == make_pair(1,1)) C11[i].insert(j);
            }
         }else{
            for(int j=0; j<dim; j++){
               auto pr = space[i].diff_type(space[j]);
               if(pr == make_pair(1,1)) C11[i].insert(j);
            }
         }
         avg += C11[i].size();
      }
*/

   } // Htype
   auto t1 = tools::get_time();
   cout << "coupling_table::get_Cmn : dim=" << dim
	<< " avg=" << avg/dim
	<< " timing : " << tools::get_duration(t1-t0) << " s" << endl; 
}
