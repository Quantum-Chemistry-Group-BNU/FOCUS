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
	 ar & dir;
         ar & sym;
	 ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      // constructor
      qtensor2(){};
      qtensor2(const qsym& sym1, 
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1,
	       const std::vector<bool> dir1={0,1,0});
      // useful functions
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
      linalg::matrix to_matrix() const;
      // deal with fermionic sign in fermionic direct product
      qtensor2 col_signed(const double fac=1.0) const;
      // algorithmic operations like matrix
      qtensor2 T() const;
      qtensor2 operator -() const;
      // pure algorithmic operations 
      qtensor2& operator +=(const qtensor2& qt);
      qtensor2& operator -=(const qtensor2& qt);
      friend qtensor2 operator +(const qtensor2& qta, const qtensor2& qtb);
      friend qtensor2 operator -(const qtensor2& qta, const qtensor2& qtb);
      qtensor2& operator *=(const double fac);
      friend qtensor2 operator *(const double fac, const qtensor2& qt);  
      friend qtensor2 operator *(const qtensor2& qt, const double fac); 
      double normF() const;
      double check_identity(const double thresh_ortho,
		            const bool debug=false) const;
      int get_dim() const;
      // symmetry conservation rule [20200515]
      bool ifconserve(const qsym& qr, const qsym& qc) const{
	 auto qsum = dir[0] ? sym : -sym;
	 qsum += dir[1] ? qr : -qr;
	 qsum += dir[2] ? qc : -qc;
	 return qsum == qsym(0,0); 
      }
   public:
      std::vector<bool> dir = {0,1,0}; // {in,out,in} by usual convention in diagrams
      qsym sym; // <row|op|col>
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
	 ar & dir;     
         ar & sym;
         ar & qmid;
         ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      // constructor
      qtensor3(){};
      qtensor3(const qsym& sym1,
	       const qsym_space& qmid1,
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1,
	       const std::vector<bool> dir1={0,1,1,0});
      inline int get_dim_mid() const{ return qsym_space_dim(qmid); }
      inline int get_dim_row() const{ return qsym_space_dim(qrow); }
      inline int get_dim_col() const{ return qsym_space_dim(qcol); }
      void print(const std::string msg, const int level=0) const;
      // deal with fermionic sign in fermionic direct product
      qtensor3 mid_signed(const double fac=1.0) const;
      qtensor3 col_signed(const double fac=1.0) const;
      // simple algrithmic operations 
      double normF() const;
      qtensor3& operator +=(const qtensor3& qt);
      qtensor3& operator -=(const qtensor3& qt);
      friend qtensor3 operator +(const qtensor3& qta, const qtensor3& qtb);
      friend qtensor3 operator -(const qtensor3& qta, const qtensor3& qtb);
      qtensor3& operator *=(const double fac);
      friend qtensor3 operator *(const double fac, const qtensor3& qt);  
      friend qtensor3 operator *(const qtensor3& qt, const double fac); 
      // for Davidson algorithm
      int get_dim() const;
      void from_array(const double* array);
      void to_array(double* array) const; 
      // symmetry conservation rule [20200510]
      bool ifconserve(const qsym& qm, const qsym& qr, const qsym& qc) const{
	 auto qsum = dir[0] ? sym : -sym;
	 qsum += dir[1] ? qm : -qm;
	 qsum += dir[2] ? qr : -qr;
	 qsum += dir[3] ? qc : -qc;
	 return qsum == qsym(0,0); 
      }
   public:
      std::vector<bool> dir = {0,1,1,0}; // {in,out,out,in}
      qsym sym; // <mid,row|op|col> (tensor A[m](r,c) = <mr|c>) 
      qsym_space qmid; // [sym,dim] - middle
      qsym_space qrow; // [sym,dim] - row
      qsym_space qcol; // [sym,dim] - col
      std::map<std::tuple<qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- rank-4 tensor: only for holding two-dot wavefunction ---
struct qtensor4{
   private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
         ar & sym;
	 ar & qmid;
         ar & qver;
         ar & qrow;
	 ar & qcol;
	 ar & qblocks;
      }
   public:
      // constructor
      qtensor4(){};
      qtensor4(const qsym& sym1,
	       const qsym_space& qmid1,
	       const qsym_space& qver1,
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1);
      void print(const std::string msg, const int level=0) const;
      // for Davidson algorithm
      int get_dim() const;
      void from_array(const double* array);
      void to_array(double* array) const; 
   public:
      //std::vector<bool> dir = {0,1,1,1,1}; // {in,out,out,out,out}
      qsym sym; 
      qsym_space qmid; 
      qsym_space qver;
      qsym_space qrow; 
      qsym_space qcol; 
      std::map<std::tuple<qsym,qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- symmetry operations : merge & expand operations ---

/*
// one-dot wavefunction
qtensor2 merge_qt3_qt2_lc(const qtensor3& qt3,
			  const qsym_space& qlc, 
			  const qsym_dpt& dpt);
qtensor3 split_qt3_qt2_lc(const qtensor2& qt2,
			  const qsym_space& ql,
			  const qsym_space& qc,
			  const qsym_dpt& dpt);
qtensor2 merge_qt3_qt2_cr(const qtensor3& qt3,
			  const qsym_space& qcr, 
			  const qsym_dpt& dpt);
qtensor3 split_qt3_qt2_cr(const qtensor2& qt2,
			  const qsym_space& qc,
			  const qsym_space& qr,
			  const qsym_dpt& dpt);
qtensor2 merge_qt3_qt2_lr(const qtensor3& qt3,
			  const qsym_space& qlr, 
			  const qsym_dpt& dpt);
qtensor3 split_qt3_qt2_lr(const qtensor2& qt2,
			  const qsym_space& ql,
			  const qsym_space& qr,
			  const qsym_dpt& dpt);
*/

// two-dot wavefunction
qtensor2 merge_qt4_qt2_lr_c1c2(const qtensor4& qt4,
			       const qsym_space& qlr,
		               const qsym_space& qc1c2,
			       const qsym_dpt& dpt1,
			       const qsym_dpt& dpt2);	       
qtensor3 merge_qt4_qt3_lc1(const qtensor4& qt4,
			   const qsym_space& qlc1, 
			   const qsym_dpt& dpt);
qtensor4 split_qt4_qt3_lc1(const qtensor3& qt3,
			   const qsym_space& ql,
			   const qsym_space& qc1,
			   const qsym_dpt& dpt);
qtensor3 merge_qt4_qt3_c2r(const qtensor4& qt4,
			   const qsym_space& qc2r, 
			   const qsym_dpt& dpt);
qtensor4 split_qt4_qt3_c2r(const qtensor3& qt3,
			   const qsym_space& qc2,
			   const qsym_space& qr,
			   const qsym_dpt& dpt);

// --- tensor linear algebra : contractions ---
qtensor3 contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor3 contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor2 contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b);

} // tns

#endif
