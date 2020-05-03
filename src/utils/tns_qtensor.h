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
	 ar & index;
         ar & msym;
	 ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      // constructor
      qtensor2(){};
      qtensor2(const qsym& msym1, 
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1,
  	       const int nindex=0); 
      // useful functions
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
      linalg::matrix to_matrix() const;
      // deal with fermionic sign in fermionic direct product
      qtensor2 col_signed(const double fac=1.0) const;
      // algorithmic operations like matrix
      qtensor2 transpose() const;
      qtensor2 operator -() const;
      // pure algorithmic operations 
      qtensor2& operator +=(const qtensor2& qt);
      qtensor2& operator -=(const qtensor2& qt);
      qtensor2& operator *=(const double fac);
      friend qtensor2 operator *(const double fac, const qtensor2& qt);  
      friend qtensor2 operator *(const qtensor2& qt, const double fac); 
      friend qtensor2 operator +(const qtensor2& qta, const qtensor2& qtb);
      friend qtensor2 operator -(const qtensor2& qta, const qtensor2& qtb);
   public:
      std::vector<short> index; // for operators (ZL@20200429); not handled in +/-;
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
      // constructor
      qtensor3(){};
      qtensor3(const qsym_space& qmid1,
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1);
      inline int get_dim_mid() const{ return qsym_space_dim(qmid); }
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
      // deal with fermionic sign in fermionic direct product
      qtensor3 mid_signed(const double fac=1.0) const;
      qtensor3 col_signed(const double fac=1.0) const;
   public:
      qsym_space qmid; // [sym,dim] - middle
      qsym_space qrow; // [sym,dim] - row
      qsym_space qcol; // [sym,dim] - col
      std::map<std::tuple<qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- tensor linear algebra : contractions ---
// for right canonical form
qtensor2 contract_qt3_qt3_cr(const qtensor3& qt3a, const qtensor3& qt3b, 
			     const qsym& msym=qsym(0,0)); 
qtensor3 contract_qt3_qt2_r(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor3 contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b);
// for left canonical form
qtensor2 contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b,
			     const qsym& msym=qsym(0,0));
qtensor3 contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b);

} // tns

#endif
