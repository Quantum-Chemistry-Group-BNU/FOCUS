#ifndef OPER_IO_H
#define OPER_IO_H

#include <filesystem>
#include "../core/serialization.h"
#include "oper_dict.h"
#include "ctns_comb.h"

//#include "h5pp/h5pp.h"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
using namespace HighFive;

namespace ctns{ 

const bool debug_oper_io = true;
extern const bool debug_oper_io;

inline std::string oper_fname(const std::string scratch, 
  	  	       	      const comb_coord& p,
		       	      const std::string kind){
   return scratch + "/" + kind + "op("
        + std::to_string(p.first) + ","
        + std::to_string(p.second) + ")";
}

template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug){ 
      std::cout << "ctns::oper_save fname=" << fname << " size=" 
		<< tools::sizeMB<Tm>(qops._size) << "MB:" 
		<< tools::sizeGB<Tm>(qops._size) << "GB"
		<< std::endl;
   }
   auto t0 = tools::get_time();

   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
   ofs.close();
   auto t1 = tools::get_time();

   std::ofstream ofs2(fname+".op", std::ios::binary);
   ofs2.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
   ofs2.close();

/*
   File file(fname+".h5", File::ReadWrite | File::Create | File::Truncate); //File::Overwrite);
   std::vector<size_t> dims{qops._size};
   DataSet dataset = file.createDataSet<Tm>("data", DataSpace(dims));
   dataset.write_raw(qops._data);

   h5pp::File file(fname+".h5", h5pp::FileAccess::REPLACE);
   file.writeDataset(qops._data, "data", qops._size);
*/

   auto t2 = tools::get_time();
   if(debug_oper_io and debug){
      std::filesystem::path p(fname+".op");
      std::cout << "size of file =" 
                << std::filesystem::file_size(fname+".op")/std::pow(1024.0,3) << "GB" 
                << std::endl; 
      std::cout << "T[ctns::oper_save](info/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tools::get_duration(t2-t0) << " "
		<< "fname=" << fname << " "
		<< "speed=" << tools::sizeMB<Tm>(qops._size)/tools::get_duration(t2-t0) << "MB/s" 
                << std::endl;
   }
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug){
      std::cout << "ctns::oper_load fname=" << fname << " size="
		<< tools::sizeMB<Tm>(qops._size) << "MB:" 
		<< tools::sizeGB<Tm>(qops._size) << "GB"
		<< std::endl;
   }
   auto t0 = tools::get_time();

   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
   ifs.close();
   auto t1 = tools::get_time();

   qops._setup_opdict();
   qops._data = new Tm[qops._size];
   auto t2 = tools::get_time();

   std::ifstream ifs2(fname+".op", std::ios::binary);
   ifs2.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
   ifs2.close();

   std::filesystem::path p(fname+".op");
   std::cout << "size of file =" 
	     << std::filesystem::file_size(fname+".op")/std::pow(1024.0,3) << "GB" 
	     << std::endl; 
/*
   File file(fname+".h5", File::ReadOnly);
   DataSet dataset = file.getDataSet("data");
   dataset.read<Tm>(qops._data);

   h5pp::File file(fname+".h5", h5pp::FileAccess::READWRITE);
   file.readDataset(qops._data, "data", qops._size);
*/
 
   qops._setup_data(qops._data);

   auto t3 = tools::get_time();
   if(debug_oper_io and debug){
      std::cout << "T[ctns::oper_load](info/setup/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tools::get_duration(t3-t2) << "," 
                << tools::get_duration(t3-t0) << " " 
		<< "fname=" << fname << " "
		<< "speed=" << tools::sizeMB<Tm>(qops._size)/tools::get_duration(t2-t0) << "MB/s" 
                << std::endl;
   }
}

} // ctns

#endif
