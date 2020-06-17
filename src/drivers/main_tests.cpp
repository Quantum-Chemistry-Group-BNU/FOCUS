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

/*   
   tests::test_dvdson();
   
   tests::test_fci();
   
   tests::test_rdm();
   
   tests::test_sci();

   tests::test_vmc();
   
   tests::test_pt2();
   
   tests::test_ordering();

   tests::test_proj();  
   
   tests::test_ras();

   tests::test_comb();
*/
   return 0;   
}
