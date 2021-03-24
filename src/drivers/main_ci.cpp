#include "../ci/tests_ci.h"

using namespace std;

int main(){
   
   // --- sci ---
   tests::test_fci();
   tests::test_sci();

   return 0;   
}
