#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include "../settings/global.h"
#include "integral.h"

using namespace global;
using namespace std;
using namespace boost;

void readIntegral(string fname,
		  twoInt& I2,
		  oneInt& I1,
		  double& coreE){
   cout << line_separator << endl;
   cout << "Integral fname = " << fname << endl;
   cout << line_separator << endl;
  
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }

   // parse FCIDUMP file
   int icounter=0;
   int norbs=0;
   int i,j,k,l;
   double eri;
   vector<string> v;
   string line;
   while(!istrm.eof() && icounter < 4){
      line.clear();	    
      std::getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      icounter++;
      //cout << line << endl;
      if(icounter == 1){
         trim_left(line); // in case there is a space in FCIDUMP
         split(v,line,is_any_of(" ,="),token_compress_on);
         norbs = stoi(v[2]);
         cout << "norbs_spatial = " << norbs << endl;
         cout << "norbs_spinorb = " << 2*norbs << endl;
      }
   }

   //
   // Currently, suppose spin-orbital eri (TO IMPROVE STORAGE LATER!)
   // 
   I1.norbs = 2*norbs; 
   I1.initSpace();
   cout << "size of I1 (MB) = " << I1.memSpace() << endl;
   I2.norbs = 2*norbs; 
   I2.initSpace();
   cout << "size of I2 (MB) = " << I2.memSpace() << endl;

   while(!istrm.eof()){
      line.clear();	    
      std::getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      trim_left(line);
      split(v,line,is_any_of(" "),token_compress_on);
      eri = stod(v[0]); 
      i = stoi(v[1]); 
      j = stoi(v[2]); 
      k = stoi(v[3]);
      l = stoi(v[4]);
      if(i*j == 0 && k*l == 0){
         coreE = eri;
      }else if(i*j != 0 && k*l == 0){
         I1(2*i-2,2*j-2) = eri; // AA
         I1(2*i-1,2*j-1) = eri; // BB
      }else if(i*j != 0 && k*l != 0){
         I2(2*i-2,2*j-2,2*k-2,2*l-2) = eri;     
      }
   }
   istrm.close();

}
