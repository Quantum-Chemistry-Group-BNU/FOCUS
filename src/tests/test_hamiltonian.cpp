#include <iostream>
#include "../utils/onstate.h"
#include "../utils/onspace.h"
#include "../utils/hamiltonian.h"
#include "../settings/global.h"

using namespace std;
using namespace fock;

int test_hamiltonian(){
   cout << global::line_separator << endl;	
   cout << "test_hamiltonian" << endl;
   cout << global::line_separator << endl;	

   onspace space1 = fci_space(4,2);
   onspace space2 = fci_space(6,1,2);

   return 0;
}
