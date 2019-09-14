#include "global.h"
#include "integral.h"
#include <vector>
#include <string>
#include <fstream>

using namespace std;

void readIntegral(string fname,
		  twoInt& I2,
		  oneInt& I1,
		  double& coreE){
   pout << line_separator << endl;
   pout << "Integral fname = " << fname << endl;
   pout << line_separator << endl;
  
   ifstream istrm(fname);
   if(!istrm){
      pout << "failed to open " << fname << '\n';
      exit(1);
   }

}
