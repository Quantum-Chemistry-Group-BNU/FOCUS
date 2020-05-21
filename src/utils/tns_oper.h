#ifndef TNS_OPER_H
#define TNS_OPER_H

#include "../core/integral.h"
#include "tns_comb.h"
#include "tns_qtensor.h"
#include <string>
#include <vector>

namespace tns{

// Build 7 types of operators specified by coord and kind 
//
// {C,A,B}:
//    Cp = ap^+
//    Bpq = ap^+aq
//    Apq = ap^+aq^+ (p<q)
// 
// {H,S,Q,P}:
//    Qps = <pq||sr> aq^+ar 
//    Ppq = <pq||sr> aras [r>s] (p<q)
//    Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
//    H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]

inline int oper_pack(const int i, const int j){ 
   const int kpack = 1000;
   return i+j*kpack;
}

inline std::pair<int,int> oper_unpack(const int ij){
   const int kpack = 1000;
   return std::make_pair(ij%kpack,ij/kpack);
}

using oper_dict = std::map<char,std::map<int,qtensor2>>;

inline std::string oper_dict_opnames(const oper_dict& qops){
   std::string s;
   s += (qops.find('A') != qops.end())? "A" : "";
   s += (qops.find('B') != qops.end())? "B" : "";
   s += (qops.find('P') != qops.end())? "P" : "";
   s += (qops.find('Q') != qops.end())? "Q" : "";
   return s;
}

// 0. IO for operators
std::string oper_fname(const std::string scratch, 
  		       const comb_coord& p,
		       const std::string optype);

void oper_save(const std::string fname, const oper_dict& qops);

void oper_load(const std::string fname, oper_dict& qops);

// 1. exact construction at type 0 site {|n>}
void oper_dot_C(const int kp, oper_dict& qops);

void oper_dot_A(const int kp, oper_dict& qops);

void oper_dot_B(const int kp, oper_dict& qops);

void oper_dot_S(const int kp,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const std::vector<int>& psupp,
		oper_dict& qops);

void oper_dot_H(const int kp,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        oper_dict& qops);

// 2. universal blocking code for {|lc>},{|cr>},{|lr>}
// O*psi
qtensor3 oper_kernel_OIwf(const std::string& superblock,
			  const qtensor3& ksite,
 			  const qtensor2& o1,
			  const bool ifdagger=false);

qtensor3 oper_kernel_IOwf(const std::string& superblock,
			  const qtensor3& ksite,
 			  const qtensor2& o2,
			  const bool po2,
			  const bool ifdagger=false);
 
qtensor3 oper_kernel_OOwf(const std::string& superblock,
			  const qtensor3& ksite,
 			  const qtensor2& o1,
 			  const qtensor2& o2,
			  const bool po2);

qtensor2 oper_kernel_renorm(const std::string& superblock,
			    const qtensor3& bsite,
			    const qtensor3& ksite);

// {P,Q,S,H}*wf
qtensor3 oper_kernel_Pwf(const std::string& superblock,
		         const qtensor3& ksite,
		         oper_dict& qops1,
		         oper_dict& qops2,
	                 const integral::two_body& int2e,
	                 const integral::one_body& int1e,
		         const int index,
			 const bool ifdagger=false);

qtensor3 oper_kernel_Qwf(const std::string& superblock,
		         const qtensor3& ksite,
		         oper_dict& qops1,
		         oper_dict& qops2,
	                 const integral::two_body& int2e,
	                 const integral::one_body& int1e,
		         const int index);

qtensor3 oper_kernel_Swf(const std::string& superblock,
		         const qtensor3& ksite,
		         oper_dict& qops1,
		         oper_dict& qops2,
	                 const integral::two_body& int2e,
	                 const integral::one_body& int1e,
		         const int index,
			 const bool ifdagger=false);

qtensor3 oper_kernel_Hwf(const std::string& superblock,
		         const qtensor3& ksite,
		         oper_dict& qops1,
		         oper_dict& qops2,
	                 const integral::two_body& int2e,
	                 const integral::one_body& int1e);

// renorm different types of operators
// normal operators
void oper_renorm_opC(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const bool debug=false);

void oper_renorm_opA(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const bool debug=false);

void oper_renorm_opB(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const bool debug=false);

// complementary operators
void oper_renorm_opP(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const std::vector<int>& supp,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e,
		     const bool debug=false);

void oper_renorm_opQ(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const std::vector<int>& supp,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e,
		     const bool debug=false);

void oper_renorm_opS(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
		     const std::vector<int>& supp,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e,
		     const bool debug=false);

void oper_renorm_opH(const std::string& superblock,
		     const qtensor3& bsite,
		     const qtensor3& ksite,
		     oper_dict& qops1,
		     oper_dict& qops2,
		     oper_dict& qops,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e,
		     const bool debug=false);

// helpers for building/loading environment
oper_dict oper_build_local(const int kp,
		           const integral::two_body& int2e,
		           const integral::one_body& int1e);

void oper_build_boundary(const comb& icomb,
			 const integral::two_body& int2e,
			 const integral::one_body& int1e,
		         const std::string scratch);

oper_dict oper_get_cqops(const comb& icomb,
		         const comb_coord& p,
			 const std::string scratch);

oper_dict oper_get_rqops(const comb& icomb,
		         const comb_coord& p,
			 const std::string scratch);

oper_dict oper_get_lqops(const comb& icomb,
		         const comb_coord& p,
			 const std::string scratch);

// driver for renorm in different directions  
oper_dict oper_renorm_ops(const std::string& superblock,
			  const comb& bra, 
		          const comb& ket,
		          const comb_coord& p,
		          oper_dict& cqops,
		          oper_dict& rqops,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug=false);

void oper_env_right(const comb& bra, 
		    const comb& ket,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
		    const std::string scratch=".");

// generator operators based on rbases from determinants for debugging
// normal operators
void oper_rbases(const comb& bra,
		 const comb& ket,
		 const comb_coord& p,
		 oper_dict& qops, 
		 const char opname);

// complementary operator
void oper_rbases(const comb& bra,
		 const comb& ket,
		 const comb_coord& p, 
		 oper_dict& qops, 
		 const char opname,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e);

} // tns

#endif
