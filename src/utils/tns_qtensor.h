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
	       const qsym_space& qcol1);
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
   public:
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
      void from_vector(const std::vector<double>& vec);
      std::vector<double> to_vector() const;
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
      // merge
      qtensor3 mergeLC();
      qtensor3 mergeCR();
   public:
      qsym sym; 
      qsym_space qmid; 
      qsym_space qver;
      qsym_space qrow; 
      qsym_space qcol; 
      std::map<std::tuple<qsym,qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- tensor linear algebra : contractions ---
qtensor3 contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor3 contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b);
qtensor2 contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b);

} // tns

#endif
