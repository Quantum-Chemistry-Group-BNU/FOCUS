#include <iostream>
#include "global.h"
#include "input.h"
#include "integral.h"

using namespace std;

void license() {
   pout << "\n"+line_separator <<endl;
   pout << "Matrix++: a matrix-driven ab initio DMRG code" << endl;
   pout << "Copyright (c) 2019 Zhendong Li" << endl;
   pout << "Author: Zhendong Li <zhendongli2008@gmail.com>" << endl;
   pout << "This program is distributed in the hope that it will be useful," << endl;
   pout << "but WITHOUT ANY WARRANTY; without even the implied warranty of" << endl;
   pout << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the" << endl;
   pout << "GNU Lesser General Public License for more details." << endl;
   pout << line_separator+"\n" <<endl;
}

int main(int argc, char* argv[]){

   license();

   // set up parallel env
   
   // input
   string fname = "input.dat";
   if(argc > 1) fname = string(argv[1]);
   schedule schd;
   schd.readInput(fname);

   // integral
   twoInt I2;
   oneInt I1;
   double coreE;
   readIntegral(schd.integralFile, I2, I1, coreE);

   // heat-bath setup

   // sci

   // pt

   // final

}
