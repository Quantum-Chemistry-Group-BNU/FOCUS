#include <iomanip>
#include <fstream>
#include <sstream> // istringstream
#include <string>
#include <cassert>
#include "input.h"
#include "../core/tools.h"

using namespace std;
using namespace input;

//
// SCI
//
void params_ci::read(ifstream& istrm){
   if(debug_input) cout << "params_ci::read" << endl; 
   run = true;
   // helpers 
   vector<int> sweep_iter; 
   vector<double> sweep_eps;
   // read 
   string line;
   while(true){
      line.clear();	   
      getline(istrm,line);
      std::cout << line << std::endl;
      if(line.empty() || line[0]=='#'){
         continue; // skip empty and comments    
      }else if(line.substr(0,4)=="$end"){
         break;
      }else if(line.substr(0,6)=="nroots"){
         nroots = stoi(line.substr(6));
      }else if(line.substr(0,4)=="flip"){
         flip = true;
      }else if(line.substr(0,4)=="eps0"){
         eps0 = stod(line.substr(4)); 
      }else if(line.substr(0,7)=="maxiter"){
         maxiter = stoi(line.substr(7));
      }else if(line.substr(0,6)=="deltaE"){
         deltaE = stod(line.substr(6));
      }else if(line.substr(0,7)=="checkms"){
         checkms = true; 
      }else if(line.substr(0,3)=="pt2"){
         ifpt2 = true;
      }else if(line.substr(0,4)=="eps2"){
         eps2 = stod(line.substr(4)); 
      }else if(line.substr(0,5)=="iroot"){
         iroot = stoi(line.substr(5));
      }else if(line.substr(0,4)=="load"){
         load = true;
      }else if(line.substr(0,4)=="dets"){
         int ndet = 0;
         while(true){
            line.clear();	   
            getline(istrm,line);
            std::cout << line << std::endl;
            if(line.empty() || line[0]=='#') continue;
            if(line.substr(0,3)=="end") break;
            ndet += 1;
            // read occ from string 
            istringstream is(line);
            string s;
            set<int> det;
            while(is>>s){
               det.insert( stoi(s) );	    
            }
            det_seeds.insert(det);
            if(ndet != det_seeds.size()) tools::exit("error: det is redundant!");
         }
         nseeds = det_seeds.size();
      }else if(line.substr(0,8)=="cisolver"){
         cisolver = stoi(line.substr(8)); 
      }else if(line.substr(0,8)=="maxcycle"){
         maxcycle = stoi(line.substr(8)); 
      }else if(line.substr(0,6)=="crit_v"){
         crit_v = stod(line.substr(6));
      }else if(line.substr(0,8)=="schedule"){
         while(true){
            line.clear();
            getline(istrm,line);
            std::cout << line << std::endl;
            if(line.empty() || line[0]=='#') continue;
            if(line.substr(0,3)=="end") break;
            istringstream is(line);
            string iter, eps;
            is >> iter >> eps;
            sweep_iter.push_back( stoi(iter) );
            sweep_eps.push_back( stod(eps) );
         }
      }else if(line.substr(0,7)=="ci_file"){
         istringstream is(line.substr(7));
         is >> ci_file;
      }else if(line.substr(0,5)=="cthrd"){
         cthrd = stod(line.substr(5));
      }else if(line.substr(0,8)=="analysis"){
         ifanalysis = true;
      }else if(line.substr(0,3)=="rdm"){
         rdm = true;
         istringstream is(line.substr(3));
         is >> iroot >> jroot;
      }else if(line.substr(0,4)=="init"){
         istringstream is(line.substr(4));
         is >> init;
      }else{
         tools::exit("error: no matching key! line = "+line);
      }
   }
   // set miniter & maxiter
   int size = sweep_iter.size();
   if(size == 0){
      tools::exit("error: schedule is not specified!");
   }else{
      // put eps1 into array
      eps1.resize(maxiter);
      for(int i=1; i<size; i++){     
         for(int j=sweep_iter[i-1]; j<sweep_iter[i]; j++){
            if(j < maxiter) eps1[j] = sweep_eps[i-1];
         }
         if(sweep_iter[i-1] < maxiter) miniter = sweep_iter[i-1];
      }
      for(int j=sweep_iter[size-1]; j<maxiter; j++){
         eps1[j] = sweep_eps[size-1];
      }
      if(sweep_iter[size-1] < maxiter) miniter = sweep_iter[size-1];
   } // size
}

void params_ci::print() const{
   cout << "\n===== params_ci::print =====" << endl;
   cout << "nroots = " << nroots << endl;
   cout << "no. of unique seeds = " << nseeds << endl;
   // print det_seeds = {|det>}
   int ndet = 0;
   for(const auto& det : det_seeds){
      cout << ndet << "-th det: ";
      for(auto k : det){
         cout << k << " ";
      }
      cout << endl;
      ndet += 1;
   } // det
   cout << "flip = " << flip << endl;
   cout << "eps0 = " << eps0 << endl;
   cout << "miniter = " << miniter << endl;
   cout << "maxiter = " << maxiter << endl;
   cout << "schedule: iter eps1" << endl;
   for(int i=0; i<maxiter; i++){
      cout << i << " " << eps1[i] << endl;
   }
   cout << "convergence parameters" << endl;
   cout << "deltaE = " << deltaE << endl;
   cout << "checkms = " << checkms << endl;
   cout << "cisolver = " << cisolver << endl;
   cout << "maxcycle = " << maxcycle << endl;
   cout << "crit_v = " << crit_v << endl;
   // pt2
   cout << "ifpt2 = " << ifpt2 << endl;
   cout << "eps2 = " << eps2 << endl;
   cout << "iroot = " << iroot << endl;
   cout << "jroot = " << jroot << endl;
   // io
   cout << "load = " << load << endl; 
   cout << "ci_file = " << ci_file << endl;
   // cthrd
   cout << "cthrd = " << cthrd << endl;
   cout << "ifanalysis = " << ifanalysis << endl;
   // rdm
   cout << "rdm = " << rdm << endl;
   // init
   cout << "init = " << init << endl;
}
