#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

void tns::oper_renorm_ropP(const comb& bra,
			   const comb& ket,
			   const comb_coord& p,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropP" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& lsupp = bra.lsupport.at(p);
   const auto& orbord = bra.orbord;
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   qtensor2 Paa(qsym(-2,-2), bsite.qrow, ksite.qrow);
   qtensor2 Pbb(qsym(-2, 0), bsite.qrow, ksite.qrow);
   qtensor2 Pab(qsym(-2,-1), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_q : lsupp){
	 int qa = 2*korb_q, qb = qa+1;
	 if(orbord[pa] < orbord[qa]) qops['P'][oper_pack(pa,qa)] = Paa;
	 if(orbord[pb] < orbord[qb]) qops['P'][oper_pack(pb,qb)] = Pbb;
	 if(orbord[pa] < orbord[qb]) qops['P'][oper_pack(pa,qb)] = Pab;
	 if(orbord[pb] < orbord[qa]) qops['P'][oper_pack(pb,qa)] = Pab;
      }
   }
   // kernel for computing renormalized P_pLqL^{CR} [3 terms] 
   for(auto& qop : qops['P']){
      auto pq = oper_unpack(qop.first);
      int pL = pq.first;
      int qL = pq.second;
      // 1. CC: Pc*Ir with Pc = <pLqL||sCrC> rCsC [=(sCrC)^+, s<r] 
      qtensor2 cop(-qop.second.sym, bsite.qmid, ksite.qmid); 
      for(const auto& copA : cqops['A']){
	 auto sr = oper_unpack(copA.first);
         int sC = sr.first;
         int rC = sr.second;
         if(cop.sym != copA.second.sym) continue;
         cop += int2e.getAnti(pL,qL,sC,rC)*copA.second;
      }
      qop.second += oper_kernel_OcIr(bsite,ksite,cop.T());
      // 3. RC: sC * <pLqL||sCrR>(-rR)
      for(const auto& copC : cqops['C']){
         int sC = copC.first;
         // rop = sum_rR <pLqL||sCrR>(-rR)
         qsym rsym = qop.second.sym + copC.second.sym;
         qtensor2 rop(-rsym, bsite.qcol, ksite.qcol);
         for(const auto& ropC : rqops['C']){
            if(rop.sym != ropC.second.sym) continue;
            int rR = ropC.first;
            rop -= int2e.getAnti(pL,qL,sC,rR)*ropC.second;
         }
         qop.second += oper_kernel_OcOr(bsite,ksite,copC.second.T(),rop.T(),1);
      }
      // 2. RR: Ic*Pr
      if(ifAB){
	 // P_pLqL^R = sum_rRsR <pLqL||sRrR> rRsR
         qtensor2 rop(-qop.second.sym, bsite.qcol, ksite.qcol);
         for(const auto& ropA : rqops['A']){
            if(rop.sym != ropA.second.sym) continue;
	    auto sr = oper_unpack(ropA.first);
            int sR = sr.first; 
            int rR = sr.second;
            rop += int2e.getAnti(pL,qL,sR,rR)*ropA.second;
         }
         qop.second += oper_kernel_IcOr(bsite,ksite,rop.T(),0);
      }else{
         qop.second += oper_kernel_IcOr(bsite,ksite,rqops['P'][qop.first],0);
      }
   }
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropP ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_ropQ(const comb& bra,
			   const comb& ket,
			   const comb_coord& p,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropQ" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& lsupp = bra.lsupport.at(p);
   const auto& orbord = bra.orbord;
   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb, Qab ~ b^+a, Qba ~ a^+b
   qtensor2 Qss(qsym(0, 0), bsite.qrow, ksite.qrow);
   qtensor2 Qab(qsym(0,-1), bsite.qrow, ksite.qrow);
   qtensor2 Qba(qsym(0, 1), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_s : lsupp){
	 int sa = 2*korb_s, sb = sa+1;
	 qops['Q'][oper_pack(pa,sa)] = Qss;
	 qops['Q'][oper_pack(pb,sb)] = Qss;
	 qops['Q'][oper_pack(pa,sb)] = Qab;
	 qops['Q'][oper_pack(pb,sa)] = Qba;
      }
   }
   // kernel for computing renormalized Q_pLsL^{CR} [4 terms]
   for(auto& qop : qops['Q']){
      auto ps = oper_unpack(qop.first);
      int pL = ps.first;
      int sL = ps.second;
      // 1. CC: Qc*Ir with Qc = <pLqC||sLrC> qC^+rC
      qtensor2 cop(qop.second.sym, bsite.qmid, ksite.qmid);
      for(const auto& copB : cqops['B']){
	 auto qr = oper_unpack(copB.first);
	 int qC = qr.first;
	 int rC = qr.second;
	 if(cop.sym != copB.second.sym) continue;
	 cop += int2e.getAnti(pL,qC,sL,rC)*copB.second;
      }
      qop.second += oper_kernel_OcIr(bsite,ksite,cop);
      // 3. CR: qC^+ * (<pLqC||sLrR> rR)
      for(const auto& copC : cqops['C']){
	 int qC = copC.first;
	 // rop = sum_rR <pLqC||sLrR> rR
	 qsym rsym = qop.second.sym - copC.second.sym;
	 qtensor2 rop(-rsym, bsite.qcol, ksite.qcol);
	 for(const auto& ropC : rqops['C']){
	    if(rop.sym != ropC.second.sym) continue;
	    int rR = ropC.first;
	    rop += int2e.getAnti(pL,qC,sL,rR)*ropC.second;
	 }
	 qop.second += oper_kernel_OcOr(bsite,ksite,copC.second,rop.T(),1);
      }
      // 4. RC: rC * (<pLqR||sLrC> (-qR^+))
      for(const auto& copC : cqops['C']){
	 int rC = copC.first;
	 // rop = sum_rR <pLqR||sLrC> (-qR^+)
	 qsym rsym = qop.second.sym + copC.second.sym;
	 qtensor2 rop(rsym, bsite.qcol, ksite.qcol);
	 for(const auto& ropC : rqops['C']){
            if(rop.sym != ropC.second.sym) continue;
	    int qR = ropC.first;
	    rop -= int2e.getAnti(pL,qR,sL,rC)*ropC.second;
	 }
	 qop.second += oper_kernel_OcOr(bsite,ksite,copC.second.T(),rop,1);
      }
      // 2. RR: Ic*Qr
      if(ifAB){
	 // Q_pLsL^R = sum_qRrR <pLqR||sLrR> qR^+rR
         qtensor2 rop(qop.second.sym, bsite.qcol, ksite.qcol);
         for(const auto& ropB : rqops['B']){
            if(rop.sym != ropB.second.sym) continue;
	    auto qr = oper_unpack(ropB.first);
            int qR = qr.first;
            int rR = qr.second;
            rop += int2e.getAnti(pL,qR,sL,rR)*ropB.second;
         }
         qop.second += oper_kernel_IcOr(bsite,ksite,rop,0);
      }else{
         qop.second += oper_kernel_IcOr(bsite,ksite,rqops['Q'][qop.first],0);
      }
   }
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropQ ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}


void tns::oper_renorm_ropS(const comb& bra,
			   const comb& ket,
			   const comb_coord& p,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropS" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& lsupp = bra.lsupport.at(p);
   const auto& orbord = bra.orbord;
   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   qtensor2 Sa(qsym(-1,-1), bsite.qrow, ksite.qrow);
   qtensor2 Sb(qsym(-1, 0), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      qops['S'][pa] = Sa;
      qops['S'][pb] = Sb;
   }
   // kernel for computing renormalized Sp^{CR} [6 terms]
   for(auto& qop : qops['S']){
      int pL = qop.first;
      // 1. qCrCsC: Sc*Ir
      qop.second += oper_kernel_OcIr(bsite,ksite,cqops['S'][pL]);
      // 2. qRrRsR: Ic*Sr
      qop.second += oper_kernel_IcOr(bsite,ksite,rqops['S'][pL],1);
      // 3. qCrRsC: B[qC,sC]*[<pLqC||sCrR>(-rR)]
      for(const auto& copB : cqops['B']){
	 auto qs = oper_unpack(copB.first);
         int qC = qs.first;
	 int sC = qs.second;
	 // rop = sum_rR <pLqC||sCrR>(-rR)
	 qsym rsym = qop.second.sym - copB.second.sym;
	 qtensor2 rop(-rsym, bsite.qcol, ksite.qcol);
	 for(const auto& ropC : rqops['C']){
	    if(rop.sym != ropC.second.sym) continue;
	    int rR = ropC.first;
	    rop -= int2e.getAnti(pL,qC,sC,rR)*ropC.second;
	 }
	 qop.second += oper_kernel_OcOr(bsite,ksite,copB.second,rop.T(),1);
      }
      // 4. qRrCsC: A[sC,rC]^+*[<pLqR||sCrC>qR^+] (s<r)
      for(const auto& copA : cqops['A']){
	 auto sr = oper_unpack(copA.first);
         int sC = sr.first;
	 int rC = sr.second;
	 // rop = sum_qR <pLqR||sCrC>qR^+
	 qsym rsym = qop.second.sym + copA.second.sym;
	 qtensor2 rop(rsym, bsite.qcol, ksite.qcol);
	 for(const auto& ropC : rqops['C']){
	    if(rsym != ropC.second.sym) continue;
	    int qR = ropC.first;
            rop += int2e.getAnti(pL,qR,sC,rC)*ropC.second;
	 }
	 qop.second += oper_kernel_OcOr(bsite,ksite,copA.second.T(),rop,1);
      }
      if(ifAB){
	 // 5. qCrRsR: A: sum_qC qC^+ * sum_rRsR <pLqC||sRrR> rRsR (r>s)
         for(const auto& copC : cqops['C']){
	    int qC = copC.first;
	    // rop = sum_rRsR <pLqC||sRrR> rRsR (r>s)
	    qsym rsym = qop.second.sym - copC.second.sym;
	    qtensor2 rop(-rsym, bsite.qcol, ksite.qcol);
	    for(const auto& ropA : rqops['A']){
	       if(rop.sym != ropA.second.sym) continue;
	       auto sr = oper_unpack(ropA.first);
	       int sR = sr.first;
	       int rR = sr.second;
	       rop += int2e.getAnti(pL,qC,sR,rR)*ropA.second;
	    }
	    qop.second += oper_kernel_OcOr(bsite,ksite,copC.second,rop.T(),0);
         }
	 // 6. qRrRsC: B: sum_sC sC * sum_qRrR <pLqR||sCrR> qR^+rR
	 for(const auto& copC : cqops['C']){
	    int sC = copC.first;
	    // rop = sum_qRrR <pLqR||sCrR> qR^+rR
	    qsym rsym = qop.second.sym + copC.second.sym;
	    qtensor2 rop(rsym, bsite.qcol, ksite.qcol);
	    for(const auto& ropB : rqops['B']){
	       if(rsym != ropB.second.sym) continue;
	       auto qr = oper_unpack(ropB.first);
	       int qR = qr.first;
	       int rR = qr.second;
	       rop += int2e.getAnti(pL,qR,sC,rR)*ropB.second;
	    }
	    qop.second += oper_kernel_OcOr(bsite,ksite,copC.second.T(),rop,0);
	 }
      }else{
	 // 5. qCrRsR: P: sum_qC qC^+ * P_pLqC^R
  	 for(const auto& copC : cqops['C']){
	    int qC = copC.first;
	    const auto& rop = rqops['P'][oper_pack(pL,qC)];
	    qsym rsym = qop.second.sym - copC.second.sym;
 	    assert(rsym == rop.sym);
	    qop.second += oper_kernel_OcOr(bsite,ksite,copC.second,rop,0);
	 }
	 // 6. qRrRsC: Q: sum_sC sC * Q_pLsC^R
	 for(const auto& copC : cqops['C']){
	    int sC = copC.first;
	    const auto& rop = rqops['Q'][oper_pack(pL,sC)];
	    qsym rsym = qop.second.sym + copC.second.sym;
	    assert(rsym == rop.sym);
	    qop.second += oper_kernel_OcOr(bsite,ksite,copC.second.T(),rop,0);
	 }
      }
   } // pL
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropS ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_ropH(const comb& bra,
			   const comb& ket,
			   const comb_coord& p,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropH" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& lsupp = bra.lsupport.at(p);
   const auto& orbord = bra.orbord;
   // kernel for H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   qtensor2 H(qsym(0,0), bsite.qrow, ksite.qrow);
   // 1. local term: Hc*Ir
   H += oper_kernel_OcIr(bsite,ksite,cqops['H'][0]);
   // 2. local term: Ic*Hr
   H += oper_kernel_IcOr(bsite,ksite,rqops['H'][0],0);
   // 3. pC^+ S_pC^R + h.c. 
   for(const auto& copC : cqops['C']){
      int pC = copC.first;
      const auto& cop = copC.second;
      const auto& rop = rqops['S'][pC];
      H += oper_kernel_OcOr(bsite,ksite,cop,rop,1);
      H -= oper_kernel_OcOr(bsite,ksite,cop.T(),rop.T(),1);
   }
   // 4. qR^+ S_qR^C + h.c. 
   for(const auto& ropC : rqops['C']){
      int qR = ropC.first;
      const auto& rop = ropC.second;
      const auto& cop = cqops['S'][qR];
      H -= oper_kernel_OcOr(bsite,ksite,cop,rop,1);
      H += oper_kernel_OcOr(bsite,ksite,cop.T(),rop.T(),1);
   }
   // 5. A: A_pCqC^C*P_pCqC^R + h.c.
   for(const auto& copA : cqops['A']){
      const auto& cop = copA.second;
      auto pq = oper_unpack(copA.first);
      int pC = pq.first;
      int qC = pq.second;
      if(ifAB){
         // rop = P_pCqC^R = sum_rRsR <pCqC||sRrR> rRsR (r>s)
         qsym rsym = H.sym - cop.sym;
         qtensor2 rop(-rsym, bsite.qcol, ksite.qcol);
         for(const auto& ropA : rqops['A']){
            if(rop.sym != ropA.second.sym) continue;
            auto sr = oper_unpack(ropA.first);
            int sR = sr.first;
            int rR = sr.second;
            rop += int2e.getAnti(pC,qC,sR,rR)*ropA.second;
         }
         rop = rop.T(); // as P constructed from A
         H += oper_kernel_OcOr(bsite,ksite,cop,rop,0);
         H += oper_kernel_OcOr(bsite,ksite,cop.T(),rop.T(),0);
      }else{
         const auto& rop = rqops['P'][copA.first];
         H += oper_kernel_OcOr(bsite,ksite,cop,rop,0);
         H += oper_kernel_OcOr(bsite,ksite,cop.T(),rop.T(),0);
      }
   }
   // 6. B: Q: B_pCsC^C*Q_pCsC^R
   for(const auto& copB : cqops['B']){
      const auto& cop = copB.second;
      auto ps = oper_unpack(copB.first);
      int pC = ps.first;
      int sC = ps.second;
      if(ifAB){
         // rop = Q_pCsC^R = sum_qRrR <pCqR||sCrR> qR^+rR
         qsym rsym = H.sym - cop.sym;
         qtensor2 rop(rsym, bsite.qcol, ksite.qcol);
         for(const auto& ropB : rqops['B']){
            if(rsym != ropB.second.sym) continue;
            auto qr = oper_unpack(ropB.first);
            int qR = qr.first;
            int rR = qr.second;
            rop += int2e.getAnti(pC,qR,sC,rR)*ropB.second;
         }
         H += oper_kernel_OcOr(bsite,ksite,cop,rop,0);
      }else{
         const auto& rop = rqops['Q'][copB.first];
         H += oper_kernel_OcOr(bsite,ksite,cop,rop,0);
      }
   }
   qops['H'][0] = H;   
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropH ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}
