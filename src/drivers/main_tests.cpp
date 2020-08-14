#include "../tests/tests.h"

using namespace std;

int main(){
   
   // -- core ---
   tests::test_tools();
   tests::test_matrix();
   tests::test_linalg();
   tests::test_onstate();
   tests::test_onspace();
   tests::test_integral();
   tests::test_hamiltonian();
   tests::test_dvdson();
   tests::test_simpleci();

   // --- sci ---
   tests::test_fci();
   tests::test_sci();

   //// --- comb --- 
   //tests::test_tns();
   //tests::test_ctns();

   return 0;   
}
