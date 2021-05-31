#include <iostream>
#include <boost/filesystem.hpp>
#include "input.h"

using namespace std;
using namespace input;
namespace fs = boost::filesystem;

void schedule::create_scratch(const bool debug){
   if(debug) cout << "\nschedule::create_scratch scratch=" << scratch << endl;
   fs::path dir(scratch);
   if(fs::exists(dir)){
      if(debug) cout << "already exists scratch=" << scratch << endl;
   }else{	 
      if(fs::create_directory(dir)){
         if(debug) cout << "successfully created " << scratch << endl;
      }else{
	 cout << "failed to create " << scratch << endl;
	 exit(1);
      }
   }
}

void schedule::remove_scratch(const bool debug){
   if(debug) cout << "\nschedule::remove_scratch scratch=" << scratch << endl;
   fs::path dir(scratch);
   if(fs::remove_all(dir)){
      if(debug) cout << "successfully removed " << scratch << endl;
   }else{
      cout << "failed in removing " << scratch << endl;
      exit(1);
   }
}
