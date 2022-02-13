#include <iostream>
#include <boost/filesystem.hpp>
#include "input.h"

using namespace std;
using namespace input;

//
// https://www.boost.org/doc/libs/1_68_0/libs/filesystem/doc/reference.html
//
namespace fs = boost::filesystem;

void schedule::create_scratch(const std::string sdir,
			      const bool debug) const{
   if(debug) cout << "\nschedule::create_scratch scratch=" << sdir << endl;
   fs::path dir(sdir);
   if(!fs::exists(dir)){ // check whether the directory exist first
      if(fs::create_directory(dir)){
         if(debug) cout << "successfully created " << sdir << endl;
      }else{
	 cout << "failed to create " << sdir << endl;
	 exit(1);
      }
   }
}

void schedule::remove_scratch(const std::string sdir,
			      const bool debug) const{
   if(debug) cout << "\nschedule::remove_scratch scratch=" << sdir << endl;
   fs::path dir(sdir);
   if(fs::exists(dir)){ // check whether the directory exist first
      if(fs::remove_all(dir)){
         if(debug) cout << "successfully removed " << sdir << endl;
      }else{
         cout << "failed in removing " << sdir << endl;
         exit(1);
      }
   }
}

void schedule::copy_scratch(const std::string sfrom,
	 	            const std::string sto,
		            const bool debug) const{
   if(debug) cout << "\nschedule::copy from " << sfrom
                  << " to " << sto << endl;
   fs::path pfrom(sfrom);
   fs::path pto(sto);
   fs::copy(pfrom, pto);
}
