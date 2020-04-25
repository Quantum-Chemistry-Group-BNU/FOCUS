#ifndef TNS_QTENSOR_H
#define TNS_QTENSOR_H

#include "../core/serialization.h"
#include "../core/matrix.h"
#include "tns_qsym.h"
#include <vector>
#include <string>
#include <map>

namespace tns{

// --- rank-2 tensor ---
// matrix <in1|o0|out> = [o0](in1,out)
struct qtensor2{
   public:
      qsym qsym0;
      qsym_space qspace1; // row
      qsym_space qspace;  // col
      std::map<std::pair<qsym,qsym>,linalg::matrix> qblocks;
};

// --- rank-3 tensor ---
const std::vector<linalg::matrix> empty_block;
extern const std::vector<linalg::matrix> empty_block;
// <in0,in1|out> = [in0](in1,out)
struct qtensor3{
   private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
         ar & qspace0;
	 ar & qspace1;
	 ar & qspace;
	 ar & qblocks;
      }
   public:
      inline int get_dim0() const{ return qsym_space_dim(qspace0); }
      inline int get_dim1() const{ return qsym_space_dim(qspace1); }
      inline int get_dim() const{ return qsym_space_dim(qspace); }
      void print(const std::string msg, const int level=0);
   public:
      qsym_space qspace0; // central [sym,dim]
      qsym_space qspace1; // in [sym,dim] - row
      qsym_space qspace;  // out [sym,dim] - col
      std::map<std::tuple<qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- tensor linear algebra : contractions ---
//qtensor2 contractCR(const qtensor3 qta, const qtensor3 qtb);

} // tns

#endif
