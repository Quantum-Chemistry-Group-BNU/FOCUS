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
// VMC
//
void params_vmc::read(ifstream& istrm){
   if(debug_input) cout << "params_vmc::read" << endl; 
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
      }else if(line.substr(0,6)=="ansatz"){
         istringstream is(line.substr(6));
         is >> ansatz;
      }else if(line.substr(0,6)=="nhiden"){
         nhiden = stoi(line.substr(6));
      }else if(line.substr(0,6)=="iscale"){
         iscale = stod(line.substr(6));
      }else if(line.substr(0,8)=="exactopt"){
         exactopt = true;
      }else if(line.substr(0,7)=="nsample"){
         nsample = stoi(line.substr(7));
      }else if(line.substr(0,7)=="maxiter"){
         maxiter = stoi(line.substr(7));
      }else if(line.substr(0,9)=="optimizer"){
         istringstream is(line.substr(9));
         is >> optimizer;
      }else if(line.substr(0,2)=="lr"){
         lr = stod(line.substr(2));
      }else if(line.substr(0,7)=="history"){
         istringstream is(line.substr(7));
         is >> history;
      }else if(line.substr(0,7)=="wf_load"){
         wf_load = true;
      }else if(line.substr(0,7)=="wf_file"){
         istringstream is(line.substr(7));
         is >> wf_file;
      }else{
         tools::exit("error: no matching key! line = "+line);
      }
   }
}

void params_vmc::print() const{
   cout << "\n===== params_vmc::print =====" << endl;
   cout << "ansatz = " << ansatz << endl;
   cout << "nhiden = " << nhiden << endl;
   cout << "iscale = " << iscale << endl;
   cout << "exactopt = " << exactopt << endl;
   cout << "nsample = " << nsample << endl;
   cout << "optimizer = " << optimizer << endl;
   cout << "maxiter = " << maxiter << endl;
   cout << "history = " << history << endl;
   cout << "wf_load = " << wf_load << endl;
   cout << "wf_file = " << wf_file << endl;
}
