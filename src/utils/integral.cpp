#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "integral.h"

using namespace std;

void integral::read_integral(integral::two_body& int2e,
		  	     integral::one_body& int1e,
		  	     double& ecore,
			     string fname){
   cout << "\nintegral::read_integral" << endl;
   cout << "fname = " << fname << endl;
  
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }

   // parse FCIDUMP file
   int icounter = 0;
   int norb = 0;
   vector<string> v;
   string line;
   while(!istrm.eof() && icounter < 4){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      icounter++;
      if(icounter == 1){
	 boost::trim_left(line); // in case there is a space in FCIDUMP
	 boost::split(v,line,boost::is_any_of(" ,="),boost::token_compress_on);
         norb = stoi(v[2]);
         cout << "norb(spatial) = " << norb << endl;
         cout << "norb(spinorb) = " << 2*norb << endl;
      }
   }

   // load integrals
   int1e.sorb = 2*norb; 
   int1e.init_space();
   cout << "size of int1e (MB) = " << int1e.get_mem_space() << endl;
   int2e.sorb = 2*norb; 
   int2e.init_space();
   cout << "size of int2e (MB) = " << int2e.get_mem_space() << endl;

   int i,j,k,l;
   double eri;
   while(!istrm.eof()){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      boost::trim_left(line);
      boost::split(v,line,boost::is_any_of(" "),boost::token_compress_on);
      eri = stod(v[0]); 
      i = stoi(v[1]); 
      j = stoi(v[2]); 
      k = stoi(v[3]);
      l = stoi(v[4]);
      if(i*j == 0 && k*l == 0){
         ecore = eri;
      }else if(i*j != 0 && k*l == 0){
         int1e(2*i-2,2*j-2) = eri; // AA
      }else if(i*j != 0 && k*l != 0){
         int2e(2*i-2,2*j-2,2*k-2,2*l-2) = eri; // AAAA 
      }
   }
   istrm.close();

}
