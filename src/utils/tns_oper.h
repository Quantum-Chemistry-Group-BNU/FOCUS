#ifndef TNS_OPER_H
#define TNS_OPER_H

#include "tns_comb.h"
#include "tns_qtensor.h"
#include "../core/integral.h"
#include <tuple>
#include <string>
#include <vector>

namespace tns{

// build different types of operators specified by
// - coord (i,j)
// - kind (Symbol)

// 1. exact construction at type 0 site {|n>}
using qopers = std::vector<qtensor2>;
qopers oper_dot_c(const int k0);
qopers oper_dot_cc(const int k0);
qopers oper_dot_ca(const int k0);
qopers oper_dot_caa(const int k0);
qopers oper_dot_ccaa(const int k0);

// 7 kinds of operators:
//
// {C,A,B}
// Cp = ap^+
// Apq = ap^+aq^+
// Bpq = ap^+aq  
//
// {H,S,Q,P}
// H = hpq ap^+aq + 1/4<pq||sr> ap^+aq^+aras
// Sp = 1/2 hpq aq + <pq||sr> aq^+ ar as (r<s)
// Qps = <pq||sr> aq^+ ar
// Ppq = <pq||rs> ar as (r<s)

// 2. universal blocking code to deal with
//    - blocking at type 1,2 site (L/R) {|nr>}
//    - blocking at type 3 site (L/R) {|ur>}

/*
void oper_renorm_A(const comb& bra, const comb& ket, comb_coord& coord);
void oper_renorm_B(const comb& bra, const comb& ket, comb_coord& coord);
void oper_renorm_H(const comb& bra, const comb& ket, comb_coord& coord);
void oper_renorm_S(const comb& bra, const comb& ket, comb_coord& coord);
void oper_renorm_Q(const comb& bra, const comb& ket, comb_coord& coord);
void oper_renorm_P(const comb& bra, const comb& ket, comb_coord& coord);
*/

void oper_renorm_rightC_kernel(const qtensor3& bsite,
		               const qtensor3& ksite,
		               const qopers& cqops,
		               const qopers& rqops,
		               qopers& qops);

void oper_renorm_rightC(const comb& bra, 
		        const comb& ket,
		        const comb_coord& p, 
		        const comb_coord& p0,
		        const int ifload,
		        const std::string scratch);
/*
void oper_renorm_B(const comb& bra, 
		   const comb& ket,
		   const comb_coord& p, 
		   const comb_coord& p0,
		   const int ifload,
		   const std::string scratch);
*/

void oper_renorm_right(const comb& bra, 
		       const comb& ket,
		       const comb_coord& p, 
		       const comb_coord& p0,
		       const std::string scratch);

void oper_env_right(const comb& bra, 
		    const comb& ket,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
		    const std::string scratch=".");

// io for operators
std::string oper_fname(const std::string scratch, 
  		       const comb_coord& p,
		       const std::string optype);
void oper_save(const std::string fname, const qopers& qops);
void oper_load(const std::string fname, qopers& qops);

} // tns

#endif
