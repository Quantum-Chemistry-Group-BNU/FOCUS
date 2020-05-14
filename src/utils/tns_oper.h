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

// 2. universal blocking code for {|nr>} and {|ln>}

// kernel for right renormalization 
qtensor2 oper_kernel_OcIr(const qtensor3& bsite,
		          const qtensor3& ksite,
		          const qtensor2& cop);

qtensor2 oper_kernel_IcOr(const qtensor3& bsite,
		          const qtensor3& ksite,
		          const qtensor2& rop,
			  const int prop);

qtensor2 oper_kernel_OcOr(const qtensor3& bsite,
		          const qtensor3& ksite,
		          const qtensor2& cop,
		          const qtensor2& rop,
			  const int prop);

// renorm different types of operators
void oper_renorm_ropC(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool debug=false);

void oper_renorm_ropA(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool debug=false);

void oper_renorm_ropB(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool debug=false);

void oper_renorm_ropP(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool& ifAB,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const bool debug=false);
	
void oper_renorm_ropQ(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool& ifAB,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const bool debug=false);
	
void oper_renorm_ropS(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool& ifAB,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const bool debug=false);
	
void oper_renorm_ropH(const comb& bra,
		      const comb& ket,
		      const comb_coord& p, 
		      oper_dict& cqops,
		      oper_dict& rqops,
		      oper_dict& qops,
		      const bool& ifAB,
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
void oper_renorm_rops(const comb& bra, 
		      const comb& ket,
		      const comb_coord& p, 
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const std::string scratch);

void oper_env_right(const comb& bra, 
		    const comb& ket,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
		    const std::string scratch=".");

void oper_renorm_lops(const comb& bra, 
		      const comb& ket,
		      const comb_coord& p, 
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const std::string scratch);

// generator operators based on rbases from determinants for debugging
// normal operators
void oper_rbases(const comb& bra,
		 const comb& ket,
		 const comb_coord& p, 
		 const std::string scratch,
		 const std::string optype);

// complementary operator
void oper_rbases(const comb& bra,
		 const comb& ket,
		 const comb_coord& p, 
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
		 const std::string scratch,
		 const std::string optype);

} // tns

#endif
