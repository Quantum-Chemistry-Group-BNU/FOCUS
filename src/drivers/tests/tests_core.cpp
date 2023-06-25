#include "core/tests_core.h"

int main(){
   
   // -- core ---
   tests::test_tools();
   tests::test_matrix();
   tests::test_onstate();
   tests::test_onspace();
   tests::test_dvdson();
   tests::test_integral();
   tests::test_hamiltonian();
   tests::test_simpleci();
   tests::test_special();

   return 0;   
}
