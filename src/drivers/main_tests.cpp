#include "../tests/tests.h"

using namespace std;

int main(){
   
   tests::test_sci();
   
   tests::test_rdm();
   
   tests::test_tools();
   
   tests::test_onstate();

   tests::test_hamiltonian();

   tests::test_dvdson();
   
   tests::test_fci();
   
   return 0;   
}
