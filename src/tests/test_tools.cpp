#include "../core/tools.h"
#include "tests.h"

using namespace std;

int tests::test_tools(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_tools" << endl;
   cout << tools::line_separator << endl;
  
   const int nmax = 5;

   cout << "\ncanonical_pair0" << endl;
   for(int i=0; i<nmax; i++){
      for(int j=0; j<nmax; j++){
	 if(i == j) continue;     
	 size_t ij = tools::canonical_pair0(i,j);
         auto p = tools::inverse_pair0(ij);
	 size_t ii = p.first, jj = p.second;
	 cout << "(" << i << "," << j << ")=>" << ij
	      << "=>" << ii << "," << jj << endl;	 
      }
   }

   cout << "\ncanonical_pair" << endl;
   for(int i=0; i<nmax; i++){
      for(int j=0; j<nmax; j++){
	 size_t ij = tools::canonical_pair(i,j);
         auto p = tools::inverse_pair(ij);
	 size_t ii = p.first, jj = p.second;
	 cout << "(" << i << "," << j << ")=>" << ij
	      << "=>" << ii << "," << jj << endl;	 
      }
   }

   return 0;
}
