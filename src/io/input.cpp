#include <iostream>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <vector>
#include "../settings/global.h"
#include "input.h"

using namespace global;
using namespace std;

void schedule::readInput(string fname){
   cout << line_separator << endl;
   cout << "Input fname = " << fname << endl;
   cout << line_separator << endl;

   ifstream istrm(fname);
   if(!istrm) {
      cout << "failed to open " << fname << '\n';
      exit(1);
   }

   this->nelec = 0;
   this->norb = 0;
   this->nseed = 0;
   this->nroots = 1;
   this->integralFile = "FCIDUMP";
   	 
   string line;
   while(!istrm.eof()){
      line.clear();	   
      std::getline(istrm,line);
      //cout << line << endl;
      if(line.empty() || line[0]=='#'){
	 continue; // skip empty and comments    
      }else if(line.substr(0,5)=="nelec"){
	 this->nelec = stoi(line.substr(5));
      }else if(line.substr(0,4)=="norb"){
	 this->norb = stoi(line.substr(4));
      }else if(line.substr(0,6)=="nroots"){
	 this->nroots = stoi(line.substr(6));
      }else if(line.substr(0,4)=="dets"){
	 while(true){
            line.clear();	   
            std::getline(istrm,line);
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    // read occ from string 
	    istringstream is(line);
	    string s;
	    vector<int> det;
	    while(is>>s){
	       det.push_back(stoi(s));	    
	    }
	    this->detSeeds.insert(det);
	 }
	 this->nseed = this->detSeeds.size();
      }else if(line.substr(0,8)=="orbitals"){
         istringstream is(line.substr(8));
	 is >> this->integralFile;
      }else{
         cout << "error: no matching key! line=" << line << endl;
	 exit(1);
      }
   }
   istrm.close();

   // check
   cout << "no. of unique seeds = " << this->nseed << endl;
   int idet = 0;
   for(auto det : detSeeds){
      cout << "det" << idet << " = "; 
      for(auto k : det)
         cout << k << " ";
      cout << endl;
      idet += 1;
   }
   cout << "integral file = " << this->integralFile << endl;

}
