#include <iostream>
#include "tests_core.h"
#include "special.h"
#include "tools.h"

using namespace std;
using namespace special;

int tests::test_special(){
   cout << std::setprecision(20);
   
   cout << gamma(3) << endl;
   cout << gamma(3.1) << endl;

   cout << "d(1,1,1,1)=" << smalld(1,1,1,1) << endl;
   cout << "d(2,2,-1,1.0)=" << smalld(2,2,-1,1.0) << endl;
   cout << "d(0.5,0.5,0.5,1.0)=" << smalld(0.5,0.5,0.5,1.0) << endl;
   cout << "d(0.5,0.5,0.5,M_PI)=" << smalld(0.5,0.5,0.5,M_PI) << endl;
   cout << "d(1.5,1.5,0.5,0.1)=" << smalld(1.5,1.5,0.5,0.1) << endl;
   cout << "d(1.5,1.5,0.5,1.0)=" << smalld(1.5,1.5,0.5,1.0) << endl;
   cout << "d(1,0,0,1)=" << smalld(1,0,0,1) << endl;
   cout << "d(2,0,0,1)=" << smalld(2,0,0,1) << endl;
   cout << "d(2,0.5,0,1)=" << smalld(2,0.5,0,1) << endl;
   cout << "d(0,0.5,0,1)=" << smalld(0,0.5,0,1) << endl;
   cout << "d(0,0,0,1)=" << smalld(0,0,0,1) << endl;
  
   for(int n=1; n<=5; n++){
      std::vector<double> x(n), w(n); 
      gen_glquad(n, x, w); 
      tools::print_vector(x,"xk");
      tools::print_vector(w,"wk");
   }

   std::vector<double> xts, wts;
   gen_s2quad(4, 3, 0.5, 0.5, xts, wts);
   int npt = xts.size();
   std::cout << "npt=" << npt << std::endl;
   for(int i=0; i<npt; i++){
      std::cout << " x,w=" << xts[i] << "," << wts[i] << std::endl;
   }
}
