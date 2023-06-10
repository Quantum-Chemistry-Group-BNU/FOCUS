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
// schedule
//
void schedule::print() const{
   cout << "\n===== schedule::print =====" << endl;
   cout << "scratch = " << scratch << endl;
   cout << "dtype = " << dtype << endl;
   cout << "nelec = " << nelec << endl;
   cout << "twoms = " << twoms << endl;
   cout << "integral_file = " << integral_file << endl;
   if(sci.run) sci.print();
   if(ctns.run) ctns.print();
   if(post.run) post.print();
   if(vmc.run) vmc.print();
}

void schedule::read(string fname){
   cout << "\nschedule::read fname = " << fname << endl;
   ifstream istrm(fname);
   if(!istrm) {
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
   std::cout << tools::line_separator << std::endl;
   string line;
   while(!istrm.eof()){
      line.clear();	   
      getline(istrm,line);
      std::cout << line << std::endl;
      if(line.empty() || line[0]=='#'){
         continue; // skip empty and comments    
      }else if(line.substr(0,7)=="scratch"){
         istringstream is(line.substr(7));
         is >> scratch;
      }else if(line.substr(0,5)=="dtype"){
         dtype = stoi(line.substr(5));
      }else if(line.substr(0,5)=="nelec"){
         nelec = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,5)=="twoms"){
         twoms = stoi(line.substr(5)); // [5,end)
      }else if(line.substr(0,13)=="integral_file"){
         istringstream is(line.substr(13));
         is >> integral_file;
      }else if(line.substr(0,4)=="$sci"){
         sci.read(istrm);
      }else if(line.substr(0,5)=="$ctns"){
         ctns.read(istrm);
      }else if(line.substr(0,5)=="$post"){
         post.read(istrm);
      }else if(line.substr(0,4)=="$vmc"){
         vmc.read(istrm);
      }else{
         tools::exit("error: no matching key! line = "+line);
      }
   }
   std::cout << tools::line_separator << std::endl;
   istrm.close();
   // consistency check
   if(scratch == ".") tools::exit("error: scratch directory must be defined!");
   print();
}
