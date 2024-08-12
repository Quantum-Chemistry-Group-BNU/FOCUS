#include "integral.h"
#include "integral_io.h"
#include "tools.h"
#include "tests_core.h"

using namespace std;

int tests::test_integral(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_integral" << endl;
   cout << tools::line_separator << endl;	
   
   // read integral
   integral::two_body<double> int2e;
   integral::one_body<double> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, "./rmole.info");
   int2e.print();

   // read integral
   integral::two_body<complex<double>> cint2e;
   integral::one_body<complex<double>> cint1e;
   double cecore;
   integral::load(cint2e, cint1e, cecore, "./cmole.info");
   cint2e.print();
   
   integral::load(cint2e, cint1e, cecore, "./rmole.info");
   cint2e.print();

   return 0;
}
