#include "../tests/tests.h"

using namespace std;

int main(){
   
   tests::test_fci();
   
   tests::test_tools();
   
   tests::test_onstate();

   tests::test_hamiltonian();

   tests::test_dvdson();
   
   return 0;   
}
