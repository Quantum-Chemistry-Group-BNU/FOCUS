#include "io.h"
namespace fs = std::filesystem;

// boost fs::copy does not work properly on mac
// #include <boost/filesystem.hpp>
// // https://www.boost.org/doc/libs/1_68_0/libs/filesystem/doc/reference.html
// namespace fs = boost::filesystem;

using namespace std;

void io::create_scratch(const std::string sdir,
      const bool debug){
   if(debug) cout << "\nio::create_scratch scratch=" << sdir << endl;
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

void io::copy_scratch(const std::string sfrom,
      const std::string sto,
      const bool debug){
   if(debug) cout << "\nio::copy from " << sfrom << " to " << sto << endl;
   fs::path pfrom(sfrom);
   fs::path pto(sto);
   fs::copy(pfrom, pto);
}

void io::remove_scratch(const std::string sdir,
      const bool debug){
   if(debug) cout << "\nio::remove_scratch scratch=" << sdir << endl;
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

void io::remove_file(const std::string fname,
      const bool debug){
   if(debug) cout << "io::remove_file fname=" << fname << endl;
   fs::remove(fname);
}

double io::directory_size(const fs::path& directory){
   double size = 0.0;
   try{
      for(const auto& entry : fs::recursive_directory_iterator(directory)){
         if(entry.is_regular_file() && !entry.is_symlink()){
            size += entry.file_size();
         }
      }
   }catch(fs::filesystem_error e){
      /* In multithread case, if an operator is removed, then it will cause a filesystem error: 
         cannot get file size: No such file or directory [./scratch/sweep/rop(2,0).op] */
      std::cout << "exception information:" << std::endl;
      std::cout << "e.code() =" << e.code() << std::endl;
      std::cout << "e.what() =" << e.what() << std::endl;
   }
   return size;
}

double io::available_disk(){
   std::error_code ec;
   const fs::space_info si = fs::space(".", ec);
   return si.available;
}
