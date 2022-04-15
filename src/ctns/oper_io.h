#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"
#include "ctns_comb.h"

/*
#include "h5pp/h5pp.h"

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
using namespace HighFive;
*/

#include "H5Cpp.h"
using namespace H5;

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
void oper_save(const int iomode,
	       const std::string fname, 
	       const oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug){ 
      std::cout << "ctns::oper_save"
	        << " iomode=" << iomode
		<< " fname=" << fname << " size=" 
		<< tools::sizeMB<Tm>(qops._size) << "MB:" 
		<< tools::sizeGB<Tm>(qops._size) << "GB"
		<< std::endl;
   }
   auto t0 = tools::get_time();

   std::ofstream ofs(fname+".info", std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
   ofs.close();
   auto t1 = tools::get_time();

   // save data on disk
   if(iomode == 0){
      std::ofstream ofs2(fname+".op", std::ios::binary);
      ofs2.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
      ofs2.close();
   }else if(iomode == 1 || iomode == 11){
      exit(1);
/*
      File file(fname+".h5", File::Overwrite);
      std::vector<size_t> dims{qops._size};

      DataSet dataset = file.createDataSet<Tm>("data", DataSpace(dims));

      dataset.write_raw(qops._data);
*/ 
   }else if(iomode == 2 || iomode == 12){
      exit(1);
/*
      std::cout << "X0" << std::endl;
      h5pp::File file(fname+".h5", h5pp::FileAccess::REPLACE);
      std::cout << "X1" << std::endl;
      // True if your installation of HDF5 has zlib support  
      if(iomode == 12){
	 assert(h5pp::hdf5::isCompressionAvaliable());
	 file.setCompressionLevel(3);
      }
      double* dat = new double[10];
      file.writeDataset(dat, "dset", 10);
      std::cout << "X2a" << std::endl;
      file.writeDataset(qops._data, "data", qops._size);
      //file.writeDataset(qops._data, "data", qops._size, H5D_CHUNKED);
      //file.writeDataset(dat, "dset", H5D_CHUNKED);
      std::cout << "X2" << std::endl;
*/
   }else{
      const H5std_string FILE_NAME(fname+".h5");
      const H5std_string DATASET_NAME("data");
      H5File file(FILE_NAME, H5F_ACC_TRUNC);
      hsize_t dimsf[1] = {qops._size};
      DataSpace dataspace(1, dimsf);
      DataSet dataset = file.createDataSet(DATASET_NAME, 
					   PredType::NATIVE_DOUBLE,
                                           dataspace);
      dataset.write(qops._data, PredType::NATIVE_DOUBLE);
   }

   auto t2 = tools::get_time();
   if(debug_oper_io and debug){
      double tot = tools::get_duration(t2-t0);
      std::cout << "T[save](info/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tot << " "
		<< "fname=" << fname << " "
		<< "size=" << tools::sizeMB<Tm>(qops._size) << "MB " 
		<< "speed=" << tools::sizeMB<Tm>(qops._size)/tot << "MB/s" 
                << std::endl;
   }
}

template <typename Tm>
void oper_load(const int iomode,
	       const std::string fname, 
	       oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug){
      std::cout << "ctns::oper_load"
	        << " iomode=" << iomode
	     	<< " fname=" << fname << " size="
		<< tools::sizeMB<Tm>(qops._size) << "MB:" 
		<< tools::sizeGB<Tm>(qops._size) << "GB"
		<< std::endl;
   }
   auto t0 = tools::get_time();

   std::ifstream ifs(fname+".info", std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
   ifs.close();
   auto t1 = tools::get_time();

   qops._setup_opdict();
   qops._data = new Tm[qops._size];
   auto t2 = tools::get_time();

   if(iomode == 0){
      std::ifstream ifs2(fname+".op", std::ios::binary);
      ifs2.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
      ifs2.close();
   }else if(iomode == 1){
      exit(1);
/*
      File file(fname+".h5", File::ReadOnly);
      DataSet dataset = file.getDataSet("data");
      dataset.read<Tm>(qops._data);
*/
   }else if(iomode == 2){
      exit(1);
/*
      h5pp::File file(fname+".h5", h5pp::FileAccess::READWRITE);
      file.readDataset(qops._data, "data", qops._size);
*/
   }else{
      const H5std_string FILE_NAME(fname+".h5");
      const H5std_string DATASET_NAME("data");
      H5File file(FILE_NAME, H5F_ACC_RDONLY);
      DataSet dataset = file.openDataSet(DATASET_NAME);
      dataset.read(qops._data, PredType::NATIVE_DOUBLE);
   }
   qops._setup_data(qops._data);

   auto t3 = tools::get_time();
   if(debug_oper_io and debug){
      double tot = tools::get_duration(t3-t0);
      std::cout << "T[load](info/setup/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tools::get_duration(t3-t2) << "," 
                << tot << " " 
		<< "fname=" << fname << " "
		<< "size=" << tools::sizeMB<Tm>(qops._size) << "MB " 
		<< "speed=" << tools::sizeMB<Tm>(qops._size)/tot << "MB/s" 
                << std::endl;
   }
}

} // ctns

#endif
