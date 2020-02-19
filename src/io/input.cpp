#include <iostream>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <vector>
#include <cassert>
#include "input.h"

using namespace std;

void input::read_input(input::schedule& schd, string fname){
   cout << "\ninput::read_input" << endl;
   cout << "fname = " << fname << endl;

   ifstream istrm(fname);
   if(!istrm) {
      cout << "failed to open " << fname << '\n';
      exit(1);
   }

   schd.nelec = 0;
   schd.norb = 0;
   schd.nseed = 0;
   schd.nroots = 1;
   schd.integral_file = "FCIDUMP";
   	 
   string line;
   while(!istrm.eof()){
      line.clear();	   
      getline(istrm,line);
      if(line.empty() || line[0]=='#'){
	 continue; // skip empty and comments    
      }else if(line.substr(0,5)=="nelec"){
	 schd.nelec = stoi(line.substr(5));
      }else if(line.substr(0,4)=="norb"){
	 schd.norb = stoi(line.substr(4));
      }else if(line.substr(0,6)=="nroots"){
	 schd.nroots = stoi(line.substr(6));
      }else if(line.substr(0,4)=="dets"){
	 int ndet = 0;
	 while(true){
            line.clear();	   
            getline(istrm,line);
	    if(line.empty() || line[0]=='#') continue;
	    if(line.substr(0,3)=="end") break;
	    ndet += 1;
	    // read occ from string 
	    istringstream is(line);
	    string s;
	    vector<int> det;
	    while(is>>s){
	       det.push_back(stoi(s));	    
	    }
	    schd.det_seeds.insert(det);
	    assert(ndet == schd.det_seeds.size());
	 }
	 schd.nseed = schd.det_seeds.size();
      }else if(line.substr(0,8)=="orbitals"){
         istringstream is(line.substr(8));
	 is >> schd.integral_file;
      }else{
         cout << "error: no matching key! line=" << line << endl;
	 exit(1);
      }
   }
   istrm.close();

   // check
   cout << "nelec = " << schd.nelec << endl;
   assert(schd.nelec > 0);
   cout << "norb = " << schd.norb << endl;
   assert(schd.norb > 0);
   cout << "no. of unique seeds = " << schd.nseed << endl;
   int ndet = 0;
   for(auto& det : schd.det_seeds){
      cout << ndet << "-th det: ";
      int nelec = 0; 
      for(auto k : det){
         cout << k << " ";
	 assert(k < 2*schd.norb);
	 nelec += 1;
      }
      assert(nelec == schd.nelec);
      cout << endl;
      ndet += 1;
   }
   cout << "integral file = " << schd.integral_file << endl;

}
