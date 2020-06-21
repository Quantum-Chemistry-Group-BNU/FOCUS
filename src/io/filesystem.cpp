#include <iostream>
#include <boost/filesystem.hpp>
#include "input.h"

using namespace std;
using namespace input;
namespace fs = boost::filesystem;

void schedule::create_scratch(){
   cout << "\nschedule::create_scratch" << endl;
   if(scratch == ".") return;
   fs::path dir(scratch);
   if(fs::exists(dir)){
      cout << "already exists scratch=" << scratch << endl;
   }else{	 
      if(fs::create_directory(dir)){
         cout << "successfully created " << scratch << endl;
      }else{
	 cout << "failed to create " << scratch << endl;
	 exit(1);
      }
   }
}

void schedule::remove_scratch(){
   cout << "\nschedule::remove_scratch" << endl;
   if(scratch == ".") return;
   fs::path dir(scratch);
   if(fs::remove_all(dir)){
      cout << "successfully removed " << scratch << endl;
   }else{
      cout << "failed in removing " << scratch << endl;
      exit(1);
   }
}
