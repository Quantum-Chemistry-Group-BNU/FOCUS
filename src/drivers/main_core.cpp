#include "../core/tests_core.h"

using namespace std;

int main(){
   
   // -- core ---
   tests::test_tools();
   tests::test_matrix();
   tests::test_linalg();
   tests::test_onstate();
   tests::test_onspace();
   tests::test_dvdson();
   tests::test_integral();
   tests::test_hamiltonian();
   tests::test_simpleci();

   return 0;   
}
