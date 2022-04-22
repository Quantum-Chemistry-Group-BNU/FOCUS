#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "../io/io.h"
#include "oper_dict.h"
#include "ctns_comb.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zstd.hpp>
#include "../io/lz4_filter.h"
namespace ext { namespace bio = ext::boost::iostreams; }

#include "fp_codec.h"


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
   }else if(iomode == 1){
      std::ofstream ofs2(fname+".op", std::ios::binary);
      boost::iostreams::filtering_ostream out;
      out.push( ext::bio::lz4_compressor() );
      out.push(ofs2);
      out.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
      out.reset();
      ofs2.close();
   }else if(iomode == 2){
      std::ofstream ofs2(fname+".op", std::ios::binary);
      boost::iostreams::filtering_ostream out;
      boost::iostreams::zstd_params params(boost::iostreams::zstd::best_speed);
      out.push( boost::iostreams::zstd_compressor(params) );
      out.push(ofs2);
      out.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
      out.reset();
      ofs2.close();
   }else if(iomode == 3){
      std::ofstream ofs2(fname+".op", std::ios::binary);
      FPCodec<double> fp;
      fp.write_array(ofs2, (double*)qops._data, qops._size);
      ofs2.close(); 
   }else{
      std::cout << "error: no such option in oper_save! iomode=" << iomode << std::endl;
      exit(1); 
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
      std::filesystem::path path{fname};
      double usage = io::directory_size(path.parent_path())/std::pow(1024,3);
      double available = io::available_disk()/std::pow(1024,3);
      double fsize = std::filesystem::file_size(fname+".op");
      std::cout << "fsize=" << fsize/std::pow(1024.0,2) << "MB " 
		<< "ratio=" << qops._size*sizeof(Tm)/fsize << " "
 	        << "usage=" << usage << "GB "
                << "available=" << available << "GB" 
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
	     	<< " fname=" << fname
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

   // read data
   if(iomode == 0){
      std::ifstream ifs2(fname+".op", std::ios::binary);
      ifs2.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
      ifs2.close();
   }else if(iomode == 1){
      std::ifstream ifs2(fname+".op", std::ios::binary);
      boost::iostreams::filtering_istream in;
      in.push( ext::bio::lz4_decompressor() );
      in.push(ifs2);
      in.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
      in.reset();
      ifs2.close();
   }else if(iomode == 2){
      std::ifstream ifs2(fname+".op", std::ios::binary);
      boost::iostreams::filtering_istream in;
      boost::iostreams::zstd_params params(boost::iostreams::zstd::best_speed);
      in.push( boost::iostreams::zstd_decompressor(params) );
      in.push(ifs2);
      in.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
      in.reset();
      ifs2.close();
   }else if(iomode == 3){
      std::ifstream ifs2(fname+".op", std::ios::binary);
      FPCodec<double> fp; 
      fp.read_array(ifs2, (double*)qops._data, qops._size);
      ifs2.close();
   }else{
      std::cout << "error: no such option in oper_load! iomode=" << iomode << std::endl;
      exit(1); 
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
