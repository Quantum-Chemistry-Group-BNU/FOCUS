#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"
#include "ctns_comb.h"

#include <sys/mman.h>
#include <fcntl.h>

namespace ctns{ 

const bool debug_oper_io = true;
extern const bool debug_oper_io;

inline std::string oper_fname(const std::string scratch, 
  	  	       	      const comb_coord& p,
		       	      const std::string kind){
   return scratch + "/" + kind + "_"
        + std::to_string(p.first) + "_"
        + std::to_string(p.second) + ".op";
}

template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops,
	       const int rank){
   if(debug_oper_io and rank == 0) std::cout << "ctns::oper_save fname=" << fname << std::endl;
   auto t0 = tools::get_time();
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
   ofs.close();
   auto t1 = tools::get_time();

   //ofs.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
   int fp = open((fname+"data").c_str(),  O_WRONLY);
   char* mmapped_data = (char*)mmap(NULL, qops._size*sizeof(Tm), PROT_WRITE, MAP_PRIVATE, fp, 0);
   if(mmapped_data == MAP_FAILED){
      close(fp);
      std::cout << "ERROR in save!" << std::endl;
      exit(1);
   }
   std::memcpy(mmapped_data, reinterpret_cast<void*>(qops._data), qops._size*sizeof(Tm)); 

   int rc = munmap(mmapped_data, qops._size*sizeof(Tm));

   //ofs.close();
   auto t2 = tools::get_time();
   if(debug_oper_io and rank == 0){
      std::cout << "timing for save:" 
                << " info:" << tools::get_duration(t1-t0) << " " 
                << " data:" << tools::get_duration(t2-t1) << " "
                << " tot:" << tools::get_duration(t2-t0) 
                << std::endl;
   }
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops,
	       const int rank){
   if(debug_oper_io and rank == 0) std::cout << "ctns::oper_load fname=" << fname << std::endl;
   auto t0 = tools::get_time();
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
   ifs.close();
   auto t1 = tools::get_time();

   qops._setup_opdict();
   qops._data = new Tm[qops._size];
   auto t2 = tools::get_time();

   int fp = open((fname+"data").c_str(),  O_RDONLY);
   char* mmapped_data = (char*)mmap(NULL, qops._size*sizeof(Tm), PROT_READ, MAP_PRIVATE, fp, 0);
   if(mmapped_data == MAP_FAILED){
      close(fp);
      std::cout << "ERROR in load!" << std::endl;
      exit(1);
   }
   std::memcpy(reinterpret_cast<void*>(qops._data), mmapped_data, qops._size*sizeof(Tm)); 
   int rc = munmap(mmapped_data, qops._size*sizeof(Tm));
   //std::fread(&qops._data[0], sizeof(Tm), qops._size, fp);
   //std::fclose(fp);
 
  //ifs.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
   //ifs.close();
   qops._setup_data(qops._data);
   auto t3 = tools::get_time();
   if(debug_oper_io and rank == 0){
      std::cout << "TIMING for load:" 
                << " info:" << tools::get_duration(t1-t0) << " " 
                << " setup:" << tools::get_duration(t2-t1) << " "
                << " data:" << tools::get_duration(t3-t2) << " "
                << " tot:" << tools::get_duration(t3-t0) 
                << std::endl;
   }
}

//
// load operators from disk for site p
//
//       cop
//        |
// lop ---*--- rop
//	  p
//
template <typename Km>
void oper_load_qops(const comb<Km>& icomb,
     		    const comb_coord& p,
     		    const std::string scratch,
		    const std::string kind,
		    oper_dict<typename Km::dtype>& qops,
		    const int rank){
   const auto& node = icomb.topo.get_node(p);
   if(kind == "c"){
      if(node.type != 3){
         auto fname0c = oper_fname(scratch, p, "c"); // physical dofs
         oper_load(fname0c, qops, rank);
      }else{
         auto pc = node.center;
         auto fname0c = oper_fname(scratch, pc, "r"); // branching site
         oper_load(fname0c, qops, rank);
      }
   }else if(kind == "r"){
      auto pr = node.right;
      auto fname0r = oper_fname(scratch, pr, "r");
      oper_load(fname0r, qops, rank);
   }else if(kind == "l"){
      auto pl = node.left;
      auto fname0l = oper_fname(scratch, pl, "l");
      oper_load(fname0l, qops, rank);
   }
}

} // ctns

#endif
