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
   private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
         ar & msym;
	 ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
      linalg::matrix to_matrix() const;
      void init_qblocks();
      qtensor2 transpose();
      qtensor2 csigned();
   public:
      qsym msym;
      qsym_space qrow;
      qsym_space qcol;
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
         ar & qmid;
         ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      inline int get_dim_mid() const{ return qsym_space_dim(qmid); }
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
   public:
      qsym_space qmid; // [sym,dim] - middle
      qsym_space qrow; // [sym,dim] - row
      qsym_space qcol; // [sym,dim] - col
      std::map<std::tuple<qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- tensor linear algebra : contractions ---
// for right canonical form
qtensor2 contract_qt3_qt3_cr(const qtensor3& qt3a, const qtensor3& qt3b, 
			     const bool sgnc=false);
qtensor3 contract_qt3_qt2_r(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor3 contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b);
// for left canonical form
qtensor2 contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b,
			     const bool sgnl=false);
qtensor3 contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b);

} // tns

#endif
