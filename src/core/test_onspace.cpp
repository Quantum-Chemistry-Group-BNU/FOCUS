#include "onspace.h"
#include "tools.h"
#include "tests_core.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_onspace(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_onspace" << endl;
   cout << tools::line_separator << endl;	
 
   onspace space1 = get_fci_space(2,1);
   check_space(space1);

   onspace space12;
   for(const auto& state1 : space1){
     for(const auto& state2 : space1){
	space12.push_back(state1.join(state2));
     }
   }
   check_space(space12);

   auto bmat0 = get_Bcouple<double>(space1[0],space1,space12);
   bmat0.print("bmat0");
   auto bmat1 = get_Bcouple<double>(space1[1],space1,space12);
   bmat1.print("bmat1");

   return 0; 
} 
