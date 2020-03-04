#include "../tests/tests.h"

using namespace std;

int main(){
   
   tests::test_dvdson();
   
   tests::test_tools();
   
   tests::test_onstate();

   tests::test_hamiltonian();

   return 0;   
}
