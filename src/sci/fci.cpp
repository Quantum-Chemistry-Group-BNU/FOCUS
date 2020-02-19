#include <iostream>
#include "../settings/global.h"
#include "../io/input.h"
#include "../io/integral.h"

using namespace std;

int main(int argc, char* argv[]){
	
   global::license();

   // read input
   string fname = "input.dat";
   if(argc > 1) fname = string(argv[1]);
   input::schedule schd;
   input::read_input(schd,fname);

   // read integral
   double ecore;
   integral::one_body int1e;
   integral::two_body int2e;
   integral::read_integral(schd.integral_file, int2e, int1e, ecore);

   // heat-bath setup
/*
   // sci
   vector<MType> ci(schd.nroots, MType::Zero(schd.nseed,1));
   vector<MType> vdVector(schd.nroots);
*/
   //vector<double> E0 = SHCIbasics::DoVariational(ci, Dets, schd, I2, I2HBSHM, irrep, I1, coreE, nelec, schd.DoRDM);
   // print most important determinants
/*
   //print the 5 most important determinants and their weights
   cout << "Printing most important determinants"<<endl;
   cout << format("%4s %10s  ") %("Det") %("weight"); cout << "Determinant string"<<endl;
   for (int root=0; root<schd.nroots; root++) {
     cout << "State :"<<root<<endl;
     MatrixXx prevci = 1.*ci[root];
     int num = max(6, schd.printBestDeterminants);
     for (int i=0; i<min(num, static_cast<int>(DetsSize)); i++) {
       compAbs comp;
       int m = distance(&prevci(0,0), max_element(&prevci(0,0), &prevci(0,0)+prevci.rows(), comp));
 #ifdef Complex
       cout << format("%4i %18.8e  ") %(i) %(abs(prevci(m,0))); cout << SHMDets[m]<<endl;
 #else
       cout << format("%4i %18.8e  ") %(i) %(prevci(m,0)); cout << SHMDets[m]<<endl;
 #endif
       //cout <<"#"<< i<<"  "<<prevci(m,0)<<"  "<<abs(prevci(m,0))<<"  "<<Dets[m]<<endl;
       prevci(m,0) = 0.0;
     }
   }
*/

   // pt

   // final
   return 0;
}
