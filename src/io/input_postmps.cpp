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
// POSTMPS
//
void params_postmps::read(ifstream& istrm){
   if(debug_input) cout << "params_postmps::read" << endl; 
   run = true;
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
      }else if(line.substr(0,5)=="qkind"){
         istringstream is(line.substr(5));
         is >> qkind;
      }else if(line.substr(0,13)=="topology_file"){
         istringstream is(line.substr(13));
         is >> topology_file;
      }else if(line.substr(0,7)=="verbose"){
         verbose = stoi(line.substr(7));
      }else if(line.substr(0,9)=="task_ovlp"){
         task_ovlp = true;
      }else if(line.substr(0,12)=="task_cicoeff"){
         task_cicoeff = true;
      }else if(line.substr(0,10)=="task_sdiag"){
         task_sdiag = true;
      }else if(line.substr(0,3)=="bra"){
         line.clear();	   
         getline(istrm,line);
         std::cout << line << std::endl;
         istringstream is(line);
         string s;
         while(is>>s){
            bra.push_back( stoi(s) );	    
         }
      }else if(line.substr(0,3)=="ket"){
         line.clear();	   
         getline(istrm,line);
         std::cout << line << std::endl;
         istringstream is(line);
         string s;
         while(is>>s){
            ket.push_back( stoi(s) );	    
         }
      }else if(line.substr(0,5)=="iroot"){
         iroot = stoi(line.substr(5));
      }else if(line.substr(0,7)=="nsample"){
         nsample = stoi(line.substr(7));
      }else if(line.substr(0,7)=="ndetprt"){
         ndetprt = stoi(line.substr(7));
      }else{
         tools::exit("error: no matching key! line = "+line);
      }
   }
}

void params_postmps::print() const{
   cout << "\n===== params_postmps::print =====" << endl;
   cout << "qkind = " << qkind << endl;
   cout << "topology_file = " << topology_file << endl;
   // debug level
   cout << "verbose = " << verbose << endl;
   // tasks
   cout << "task_ovlp = " << task_ovlp << endl;
   cout << "task_cicoeff = " << task_cicoeff << endl;
   cout << "task_sdiag = " << task_sdiag << endl;
   tools::print_vector(bra, "bra");
   tools::print_vector(ket, "ket");
   cout << "iroot = " << iroot << endl;
   cout << "nsample = " << nsample << endl;
   cout << "ndetprt = " << ndetprt << endl;
}
