#ifndef TNS_OPER_H
#define TNS_OPER_H

#include "tns_comb.h"
#include "tns_qtensor.h"
#include "../core/integral.h"
#include <tuple>
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

// 1. exact construction at type 0 site {|n>}
using qopers = std::vector<qtensor2>;
qopers oper_dot_c(const int k0);
qopers oper_dot_cc(const int k0);
qopers oper_dot_ca(const int k0);
qopers oper_dot_caa(const int k0);
qopers oper_dot_ccaa(const int k0);

void oper_dot_rightS_loc(const int k,
			 const std::vector<int>& lsupp,
			 const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const qopers& cqops_c,
			 qopers& cqops_S);
 
// 2. universal blocking code for {|nr>} and {|ln>}

// --- kernel ---
qtensor2 oper_kernel_right_IcOr(const qtensor3& bsite,
		                const qtensor3& ksite,
		                const qtensor2& rop);

qtensor2 oper_kernel_right_OcIr(const qtensor3& bsite,
		                const qtensor3& ksite,
		                const qtensor2& cop);

qtensor2 oper_kernel_right_OcOr(const qtensor3& bsite,
		                const qtensor3& ksite,
		                const qtensor2& cop,
		                const qtensor2& rop);

qtensor2 oper_kernel_right_OrOc(const qtensor3& bsite,
		                const qtensor3& ksite,
		                const qtensor2& rop,
		                const qtensor2& cop);

// --- renorm ---
void oper_renorm_rightC(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
		        const std::string scratch);

void oper_renorm_rightB(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
		        const std::string scratch);

void oper_renorm_rightA(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
		        const std::string scratch);

void oper_renorm_rightQ(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
	                const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const std::string scratch);

void oper_renorm_rightP(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
	                const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const std::string scratch);

void oper_renorm_rightS(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
	                const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const std::string scratch);

void oper_renorm_rightH(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
	                const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const std::string scratch);

// --- different directions ---
void oper_renorm_right(const comb& bra, 
		       const comb& ket,
		       const comb_coord& p, 
		       const comb_coord& p0,
	               const integral::two_body& int2e,
	               const integral::one_body& int1e,
		       const std::string scratch);

void oper_env_right(const comb& bra, 
		    const comb& ket,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
		    const std::string scratch=".");

// --- io for operators ---
std::string oper_fname(const std::string scratch, 
  		       const comb_coord& p,
		       const std::string optype);
void oper_save(const std::string fname, const qopers& qops);
void oper_load(const std::string fname, qopers& qops);

} // tns

#endif
