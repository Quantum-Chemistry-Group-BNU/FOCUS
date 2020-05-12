#include "../settings/global.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_ropP(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropP" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops;
   qopers cqops_c, cqops_cc;
   qopers rqops_c, rqops_cc, rqops_P;
   string fname0r, fname0c;
   auto lsupp = bra.lsupport.at(p);
   // C: build / load
   if(bra.ifbuild_c(p)){
      cqops_c  = oper_dot_c(kp);
      cqops_cc = oper_dot_cc(kp);
   }else{
      auto pc = bra.get_c(p);
      fname0c = oper_fname(scratch, pc, "ropC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "ropA");
      oper_load(fname0c, cqops_cc);
   }
   // R: load
   assert(!bra.ifbuild_r(p));
   auto pr = bra.get_r(p);
   fname0r = oper_fname(scratch, pr, "ropC");
   oper_load(fname0r, rqops_c);

   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   qtensor2 Paa(qsym(-2,-2), bsite.qcol, ksite.qcol, 2);
   qtensor2 Pbb(qsym(-2, 0), bsite.qcol, ksite.qcol, 2);
   qtensor2 Pab(qsym(-2,-1), bsite.qcol, ksite.qcol, 2);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_q : lsupp){
	 int qa = 2*korb_q, qb = qa+1;
         Paa.index[0] = pa;
         Paa.index[1] = qa;
	 if(bra.orbord[pa] < bra.orbord[qa]) qops.push_back(Paa);
         Pbb.index[0] = pb;
         Pbb.index[1] = qb;
	 if(bra.orbord[pb] < bra.orbord[qb]) qops.push_back(Pbb);
         Pab.index[0] = pa;
         Pab.index[1] = qb;
	 if(bra.orbord[pa] < bra.orbord[qb]) qops.push_back(Pab);
         // for ababab orbital ordering, Pba may also be needed.
	 Pab.index[0] = pb;
         Pab.index[1] = qa;
	 if(bra.orbord[pb] < bra.orbord[qa]) qops.push_back(Pab);
      }
   }

   // kernel for computing renormalized P_pLqL^{CR} [3 terms] 
   for(int i=0; i<qops.size(); i++){
      int pL = qops[i].index[0];
      int qL = qops[i].index[1];
      assert(bra.orbord[pL] < bra.orbord[qL]);
      // 1. CC: Pc*Ir 
      qtensor2 cop(-qops[i].sym, bsite.qmid, ksite.qmid, 2); 
      for(const auto& cop_cc : cqops_cc){
         int sC = cop_cc.index[0];
         int rC = cop_cc.index[1];
         // Pc = <pLqL||sCrC> rCsC [=(sCrC)^+, s<r]
         assert(bra.orbord[sC] < bra.orbord[rC]);
         if(cop.sym != cop_cc.sym) continue;
         cop += int2e.getAnti(pL,qL,sC,rC)*cop_cc;
      }
      qops[i] += oper_kernel_OcIr(bsite,ksite,cop.T());
      // 3. RC: sC * <pLqL||sCrR>(-rR)
      for(const auto& cop_c : cqops_c){
         int sC = cop_c.index[0];
         // rop = sum_rR <pLqL||sCrR>(-rR)
         qsym rsym = qops[i].sym + cop_c.sym;
         qtensor2 rop(-rsym, bsite.qrow, ksite.qrow, 1);
         for(const auto& rop_c : rqops_c){
            if(rop.sym != rop_c.sym) continue;
            int rR = rop_c.index[0];
            rop -= int2e.getAnti(pL,qL,sC,rR)*rop_c;
         }
         qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop.T());
      }
   }
   // 2. RR: Ic*Pr
   if(ifAB){
      fname0r = oper_fname(scratch, pr, "ropA");
      oper_load(fname0r, rqops_cc);
      // overlap switch point constructed from Asr=as^+ar^+
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
         int qL = qops[i].index[1];
         assert(bra.orbord[pL] < bra.orbord[qL]);
	 // P_pLqL^R = sum_rRsR <pLqL||sRrR> rRsR
         qtensor2 rop(-qops[i].sym, bsite.qrow, ksite.qrow, 2);
         for(const auto& rop_cc : rqops_cc){
            if(rop.sym != rop_cc.sym) continue;
            int sR = rop_cc.index[0];
            int rR = rop_cc.index[1];
            rop += int2e.getAnti(pL,qL,sR,rR)*rop_cc;
         }
         qops[i] += oper_kernel_IcOr(bsite,ksite,rop.T());
      }
   }else{
      fname0r = oper_fname(scratch, pr, "ropP");
      oper_load(fname0r, rqops_P);
      map<pair<int,int>,int> Pr_pLqL2pos;
      for(int idx=0; idx<rqops_P.size(); idx++){
         int pL = rqops_P[idx].index[0];
         int qL = rqops_P[idx].index[1];
         Pr_pLqL2pos[make_pair(pL,qL)] = idx;
      }
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
         int qL = qops[i].index[1];
         assert(bra.orbord[pL] < bra.orbord[qL]);
         // RR: Ic*Pr
         int pos_r = Pr_pLqL2pos.at(make_pair(pL,qL));
         qops[i] += oper_kernel_IcOr(bsite,ksite,rqops_P[pos_r]);
      }
   }
   string fname = oper_fname(scratch, p, "ropP"); 
   oper_save(fname, qops);
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropP ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,int2e,int1e,scratch,"P");
   }
}

void tns::oper_renorm_ropQ(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropQ" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops;
   qopers cqops_c, cqops_ca;
   qopers rqops_c, rqops_ca, rqops_Q;
   string fname0r, fname0c;
   auto lsupp = bra.lsupport.at(p);
   // C: build / load
   if(bra.ifbuild_c(p)){
      cqops_c  = oper_dot_c(kp);
      cqops_ca = oper_dot_ca(kp);
   }else{
      auto pc = bra.get_c(p);
      fname0c = oper_fname(scratch, pc, "ropC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "ropB");
      oper_load(fname0c, cqops_ca);
   }
   // R: load
   assert(!bra.ifbuild_r(p));
   auto pr = bra.get_r(p);
   fname0r = oper_fname(scratch, pr, "ropC");
   oper_load(fname0r, rqops_c);

   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb, Qab ~ b^+a, Qba ~ a^+b
   qtensor2 Qss(qsym(0, 0), bsite.qcol, ksite.qcol, 2);
   qtensor2 Qab(qsym(0,-1), bsite.qcol, ksite.qcol, 2);
   qtensor2 Qba(qsym(0, 1), bsite.qcol, ksite.qcol, 2);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_s : lsupp){
	 int sa = 2*korb_s, sb = sa+1;
         Qss.index[0] = pa;
         Qss.index[1] = sa;
	 qops.push_back(Qss);
         Qss.index[0] = pb;
         Qss.index[1] = sb;
	 qops.push_back(Qss);
         Qab.index[0] = pa;
         Qab.index[1] = sb;
	 qops.push_back(Qab);
         Qba.index[0] = pb;
         Qba.index[1] = sa;
	 qops.push_back(Qba);
      }
   }

   // kernel for computing renormalized Q_pLsL^{CR} [4 terms]
   for(int i=0; i<qops.size(); i++){
      int pL = qops[i].index[0];
      int sL = qops[i].index[1];
      // 1. CC: Qc*Ir
      qtensor2 cop(qops[i].sym, bsite.qmid, ksite.qmid, 2);
      for(const auto& cop_ca : cqops_ca){
	 int qC = cop_ca.index[0];
	 int rC = cop_ca.index[1];
	 // Qc = <pLqC||sLrC> qC^+rC
	 if(cop.sym != cop_ca.sym) continue;
	 cop += int2e.getAnti(pL,qC,sL,rC)*cop_ca;
      }
      qops[i] += oper_kernel_OcIr(bsite,ksite,cop);
      // 3. CR: qC^+ * (<pLqC||sLrR> rR)
      for(const auto& cop_c : cqops_c){
	 int qC = cop_c.index[0];
	 // rop = sum_rR <pLqC||sLrR> rR
	 qsym rsym = qops[i].sym - cop_c.sym;
	 qtensor2 rop(-rsym, bsite.qrow, ksite.qrow, 1);
	 for(const auto& rop_c : rqops_c){
	    if(rop.sym != rop_c.sym) continue;
	    int rR = rop_c.index[0];
	    rop += int2e.getAnti(pL,qC,sL,rR)*rop_c;
	 }
	 qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c,rop.T()); 
      }
      // 4. RC: rC * (<pLqR||sLrC> (-qR^+))
      for(const auto& cop_c : cqops_c){
	 int rC = cop_c.index[0];
	 // rop = sum_rR <pLqR||sLrC> (-qR^+)
	 qsym rsym = qops[i].sym + cop_c.sym;
	 qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 1);
	 for(const auto& rop_c : rqops_c){
            if(rop.sym != rop_c.sym) continue;
	    int qR = rop_c.index[0];
	    rop -= int2e.getAnti(pL,qR,sL,rC)*rop_c;
	 }
	 qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop);
      }
   }
   // 2. RR: Ic*Qr
   if(ifAB){
      fname0r = oper_fname(scratch, pr, "ropB");
      oper_load(fname0r, rqops_ca);
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
         int sL = qops[i].index[1];
	 // Q_pLsL^R = sum_qRrR <pLqR||sLrR> qR^+rR
         qtensor2 rop(qops[i].sym, bsite.qrow, ksite.qrow, 2);
         for(const auto& rop_ca : rqops_ca){
            if(rop.sym != rop_ca.sym) continue;
            int qR = rop_ca.index[0];
            int rR = rop_ca.index[1];
            rop += int2e.getAnti(pL,qR,sL,rR)*rop_ca;
         }
         qops[i] += oper_kernel_IcOr(bsite,ksite,rop);
      }
   }else{
      fname0r = oper_fname(scratch, pr, "ropQ");
      oper_load(fname0r, rqops_Q);
      map<pair<int,int>,int> Qr_pLsL2pos;
      for(int idx=0; idx<rqops_Q.size(); idx++){
         int pL = rqops_Q[idx].index[0];
         int sL = rqops_Q[idx].index[1];
         Qr_pLsL2pos[make_pair(pL,sL)] = idx;
      }
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
         int sL = qops[i].index[1];
         // RR: Ic*Qr
         int pos_r = Qr_pLsL2pos.at(make_pair(pL,sL));
         qops[i] += oper_kernel_IcOr(bsite,ksite,rqops_Q[pos_r]);
      }
   }
   string fname = oper_fname(scratch, p, "ropQ"); 
   oper_save(fname, qops);
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropQ ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,int2e,int1e,scratch,"Q");
   }
}

void tns::oper_renorm_ropS(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropS" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops;
   qopers cqops_c, cqops_cc, cqops_ca, cqops_S;
   qopers rqops_c, rqops_cc, rqops_ca, rqops_S, rqops_P, rqops_Q;
   string fname0r, fname0c;
   auto lsupp = bra.lsupport.at(p);
   // C: build / load
   if(bra.ifbuild_c(p)){
      cqops_c  = oper_dot_c(kp);
      cqops_cc = oper_dot_cc(kp);
      cqops_ca = oper_dot_ca(kp);
      oper_dot_S(kp, lsupp, int2e, int1e, cqops_c, cqops_S);
   }else{
      auto pc = bra.get_c(p);
      fname0c = oper_fname(scratch, pc, "ropC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "ropA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "ropB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "ropS");
      oper_load(fname0c, cqops_S);
   }
   // R: build /load
   auto pr = bra.get_r(p);
   if(bra.ifbuild_r(p)){
      int kpr = bra.get_kp(pr);
      rqops_c  = oper_dot_c(kpr);
      rqops_cc = oper_dot_cc(kpr);
      rqops_ca = oper_dot_ca(kpr);
      oper_dot_S(kpr, lsupp, int2e, int1e, rqops_c, rqops_S);
   }else{
      fname0r = oper_fname(scratch, pr, "ropC");
      oper_load(fname0r, rqops_c);
      if(ifAB){	      
	 // branch & rop half of the backbone [type-AB]
         fname0r = oper_fname(scratch, pr, "ropA");
         oper_load(fname0r, rqops_cc);
         fname0r = oper_fname(scratch, pr, "ropB");
         oper_load(fname0r, rqops_ca);
      }else{
	 // left half of the backbone [type-PQ]
         fname0r = oper_fname(scratch, pr, "ropP");
         oper_load(fname0r, rqops_P);
         fname0r = oper_fname(scratch, pr, "ropQ");
         oper_load(fname0r, rqops_Q);
      }
      fname0r = oper_fname(scratch, pr, "ropS");
      oper_load(fname0r, rqops_S);
   }

   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   qtensor2 Sa(qsym(-1,-1), bsite.qcol, ksite.qcol, 1);
   qtensor2 Sb(qsym(-1, 0), bsite.qcol, ksite.qcol, 1);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      Sa.index[0] = pa;
      qops.push_back(Sa);
      Sb.index[0] = pb;
      qops.push_back(Sb);
   }

   // kernel for computing renormalized Sp^{CR} [6 terms]
   // resolving the index matching issue
   map<int,int> Sc_pL2pos;
   for(int idx=0; idx<cqops_S.size(); idx++){
      int pL = cqops_S[idx].index[0];
      Sc_pL2pos[pL] = idx;
   }
   map<int,int> Sr_pL2pos;
   for(int idx=0; idx<rqops_S.size(); idx++){
      int pL = rqops_S[idx].index[0];
      Sr_pL2pos[pL] = idx;
   }
   // SpL = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   for(int i=0; i<qops.size(); i++){
      int pL = qops[i].index[0];
      // 1. qCrCsC: Sc*Ir
      int pos_c = Sc_pL2pos.at(pL);
      qops[i] += oper_kernel_OcIr(bsite,ksite,cqops_S[pos_c]);
      // 2. qRrRsR: Ic*Sr
      int pos_r = Sr_pL2pos.at(pL);
      qops[i] += oper_kernel_IcOr(bsite,ksite,rqops_S[pos_r]);
      // 3. qCrRsC: B[qC,sC]*[<pLqC||sCrR>(-rR)]
      for(const auto& cop_ca : cqops_ca){
         int qC = cop_ca.index[0];
	 int sC = cop_ca.index[1];
	 // rop = sum_rR <pLqC||sCrR>(-rR)
	 qsym rsym = qops[i].sym - cop_ca.sym;
	 qtensor2 rop(-rsym, bsite.qrow, ksite.qrow, 1);
	 for(const auto& rop_c : rqops_c){
	    if(rop.sym != rop_c.sym) continue;
	    int rR = rop_c.index[0];
	    rop -= int2e.getAnti(pL,qC,sC,rR)*rop_c;
	 }
	 qops[i] += oper_kernel_OcOr(bsite,ksite,cop_ca,rop.T());
      }
      // 4. qRrCsC: A[sC,rC]^+*[<pLqR||sCrC>qR^+] (s<r)
      for(const auto& cop_cc : cqops_cc){
         int sC = cop_cc.index[0];
	 int rC = cop_cc.index[1];
	 assert(bra.orbord[sC] < bra.orbord[rC]);
	 // rop = sum_qR <pLqR||sCrC>qR^+
	 qsym rsym = qops[i].sym + cop_cc.sym;
	 qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 1);
	 for(const auto& rop_c : rqops_c){
	    if(rsym != rop_c.sym) continue;
	    int qR = rop_c.index[0];
            rop += int2e.getAnti(pL,qR,sC,rC)*rop_c; 
	 }
	 qops[i] += oper_kernel_OcOr(bsite,ksite,cop_cc.T(),rop);
      }
   } // pL
   if(ifAB){
      // type-AB decomposition
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
	 // 5. qCrRsR: A: sum_qC qC^+ * sum_rRsR <pLqC||sRrR> rRsR (r>s)
         for(const auto& cop_c : cqops_c){
	    int qC = cop_c.index[0];
	    // rop = sum_rRsR <pLqC||sRrR> rRsR (r>s)
	    qsym rsym = qops[i].sym - cop_c.sym;
	    qtensor2 rop(-rsym, bsite.qrow, ksite.qrow, 2);
	    for(const auto& rop_cc : rqops_cc){
	       if(rop.sym != rop_cc.sym) continue;
	       int rR = rop_cc.index[1];
	       int sR = rop_cc.index[0];
	       rop += int2e.getAnti(pL,qC,sR,rR)*rop_cc;
	    }
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c,rop.T());
         }
	 // 6. qRrRsC: B: sum_sC sC * sum_qRrR <pLqR||sCrR> qR^+rR
	 for(const auto& cop_c : cqops_c){
	    int sC = cop_c.index[0];
	    // rop = sum_qRrR <pLqR||sCrR> qR^+rR
	    qsym rsym = qops[i].sym + cop_c.sym;
	    qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
	    for(const auto& rop_ca : rqops_ca){
	       if(rsym != rop_ca.sym) continue;
	       int qR = rop_ca.index[0];
	       int rR = rop_ca.index[1];
	       rop += int2e.getAnti(pL,qR,sC,rR)*rop_ca;
	    }
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop);
	 }
      } // pL
   }else{
      // type-PQ decomposition
      map<pair<int,int>,int> Pr_pLqL2pos;
      for(int idx=0; idx<rqops_P.size(); idx++){
         int pL = rqops_P[idx].index[0];
	 int qL = rqops_P[idx].index[1];
         Pr_pLqL2pos[make_pair(pL,qL)] = idx;
      }
      map<pair<int,int>,int> Qr_pLsL2pos;
      for(int idx=0; idx<rqops_Q.size(); idx++){
         int pL = rqops_Q[idx].index[0];
         int sL = rqops_Q[idx].index[1];
         Qr_pLsL2pos[make_pair(pL,sL)] = idx;
      }
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
	 // 5. qCrRsR: P: sum_qC qC^+ * P_pLqC^R
  	 for(const auto& cop_c : cqops_c){
	    int qC = cop_c.index[0];
	    assert(bra.orbord[pL] < bra.orbord[qC]);
	    int pos = Pr_pLqL2pos.at(make_pair(pL,qC));
	    auto& rop = rqops_P[pos];
	    qsym rsym = qops[i].sym - cop_c.sym;
 	    assert(rsym == rop.sym);
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c,rop);
	 } 
	 // 6. qRrRsC: Q: sum_sC sC * Q_pLsC^R
	 for(const auto& cop_c : cqops_c){
	    int sC = cop_c.index[0];
	    int pos = Qr_pLsL2pos.at(make_pair(pL,sC));
	    auto& rop = rqops_Q[pos];
	    qsym rsym = qops[i].sym + cop_c.sym;
	    assert(rsym == rop.sym);
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop);
	 }
      }	// pL 
   } // ifAB
   string fname = oper_fname(scratch, p, "ropS"); 
   oper_save(fname, qops);
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropS ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,int2e,int1e,scratch,"S");
   }
}

void tns::oper_renorm_ropH(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const bool& ifAB,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropH" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops;
   qopers cqops_c, cqops_cc, cqops_ca, cqops_S, cqops_H;
   qopers rqops_c, rqops_cc, rqops_ca, rqops_S, rqops_H, rqops_P, rqops_Q;
   string fname0r, fname0c;
   // C: build / load
   if(bra.ifbuild_c(p)){
      cqops_c  = oper_dot_c(kp);
      cqops_cc = oper_dot_cc(kp);
      cqops_ca = oper_dot_ca(kp);
      auto rsupp = bra.rsupport.at(p);
      oper_dot_S(kp, rsupp, int2e, int1e, cqops_c, cqops_S);
      oper_dot_H(kp, int2e, int1e, cqops_ca, cqops_H);
   }else{
      auto pc = bra.get_c(p);
      fname0c = oper_fname(scratch, pc, "ropC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "ropA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "ropB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "ropS");
      oper_load(fname0c, cqops_S);
      fname0c = oper_fname(scratch, pc, "ropH");
      oper_load(fname0c, cqops_H);
   }
   // R: build /load
   auto pr = bra.get_r(p);
   if(bra.ifbuild_r(p)){
      int kpr = bra.get_kp(pr);
      rqops_c  = oper_dot_c(kpr);
      rqops_cc = oper_dot_cc(kpr);
      rqops_ca = oper_dot_ca(kpr);
      // support for Hc
      vector<int> csupp;
      if(bra.ifbuild_c(p)){
         csupp.push_back({kp});
      }else{
         auto pc = bra.get_c(p);
	 csupp = bra.rsupport.at(pc);
      }
      oper_dot_S(kpr, csupp, int2e, int1e, rqops_c, rqops_S);
      oper_dot_H(kpr, int2e, int1e, rqops_ca, rqops_H);
   }else{
      fname0r = oper_fname(scratch, pr, "ropC");
      oper_load(fname0r, rqops_c);
      if(ifAB){
         // branch & rop half of the backbone [type-AB]
         fname0r = oper_fname(scratch, pr, "ropA");
         oper_load(fname0r, rqops_cc);
         fname0r = oper_fname(scratch, pr, "ropB");
         oper_load(fname0r, rqops_ca);
      }else{
	 // left half of the backbone [type-PQ]
         fname0r = oper_fname(scratch, pr, "ropP");
         oper_load(fname0r, rqops_P);
         fname0r = oper_fname(scratch, pr, "ropQ");
         oper_load(fname0r, rqops_Q);
      }
      fname0r = oper_fname(scratch, pr, "ropS");
      oper_load(fname0r, rqops_S);
      fname0r = oper_fname(scratch, pr, "ropH");
      oper_load(fname0r, rqops_H);
   }

   // kernel for H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   qtensor2 H(qsym(0,0), bsite.qcol, ksite.qcol, 0);
   // 1. local term: Hc*Ir
   H += oper_kernel_OcIr(bsite,ksite,cqops_H[0]);
   // 2. local term: Ic*Hr
   H += oper_kernel_IcOr(bsite,ksite,rqops_H[0]);
   // 3. pC^+ S_pC^R + h.c. 
   map<int,int> Sr_pL2pos;
   for(int idx=0; idx<rqops_S.size(); idx++){
      int pL = rqops_S[idx].index[0];
      Sr_pL2pos[pL] = idx;
   }
   for(const auto& cop_c : cqops_c){
      int pC = cop_c.index[0];
      int pos = Sr_pL2pos.at(pC);
      auto& rop = rqops_S[pos];
      qsym rsym = H.sym - cop_c.sym;
      assert(rsym == rop.sym);
      H += oper_kernel_OcOr(bsite,ksite,cop_c,rop);
      H += oper_kernel_OrOc(bsite,ksite,rop.T(),cop_c.T());
   }
   // 4. qR^+ S_qR^C + h.c. 
   map<int,int> Sc_pL2pos;
   for(int idx=0; idx<cqops_S.size(); idx++){
      int pL = cqops_S[idx].index[0];
      Sc_pL2pos[pL] = idx;
   }   
   for(const auto& rop_c : rqops_c){
      int qR = rop_c.index[0];
      int pos = Sc_pL2pos.at(qR);
      auto& cop = cqops_S[pos];
      qsym csym = H.sym - rop_c.sym;
      assert(csym == cop.sym);
      H += oper_kernel_OrOc(bsite,ksite,rop_c,cop);
      H += oper_kernel_OcOr(bsite,ksite,cop.T(),rop_c.T());
   }
   if(ifAB){
      // 5. A: A_pCqC^C*P_pCqC^R + h.c.
      for(const auto& cop_cc : cqops_cc){
         int pC = cop_cc.index[0];
	 int qC = cop_cc.index[1];
	 assert(bra.orbord[pC] < bra.orbord[qC]);
         // rop = P_pCqC^R = sum_rRsR <pCqC||sRrR> rRsR (r>s)
         qsym rsym = H.sym - cop_cc.sym;
         qtensor2 rop(-rsym, bsite.qrow, ksite.qrow, 2);
         for(const auto& rop_cc : rqops_cc){
            if(rop.sym != rop_cc.sym) continue;
            int rR = rop_cc.index[1];
            int sR = rop_cc.index[0];
            rop += int2e.getAnti(pC,qC,sR,rR)*rop_cc;
         }
	 H += oper_kernel_OcOr(bsite,ksite,cop_cc,rop.T());
	 H += oper_kernel_OrOc(bsite,ksite,rop,cop_cc.T());
      }
      // 6. B: Q: B_pCsC^C*Q_pCsC^R
      for(const auto& cop_ca : cqops_ca){
         int pC = cop_ca.index[0];
	 int sC = cop_ca.index[1];
         // rop = Q_pCsC^R = sum_qRrR <pCqR||sCrR> qR^+rR
         qsym rsym = H.sym - cop_ca.sym;
         qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
         for(const auto& rop_ca : rqops_ca){
            if(rsym != rop_ca.sym) continue;
            int qR = rop_ca.index[0];
            int rR = rop_ca.index[1];
            rop += int2e.getAnti(pC,qR,sC,rR)*rop_ca;
         }
         H += oper_kernel_OcOr(bsite,ksite,cop_ca,rop);
      }
   }else{
      // 5. P: A_pCqC^C*P_pCqC^R + h.c.
      map<pair<int,int>,int> Pr_pLqL2pos;
      for(int idx=0; idx<rqops_P.size(); idx++){
         int pL = rqops_P[idx].index[0];
	 int qL = rqops_P[idx].index[1];
         Pr_pLqL2pos[make_pair(pL,qL)] = idx;
      }
      for(const auto& cop_cc : cqops_cc){
	 int pC = cop_cc.index[0];
	 int qC = cop_cc.index[1];
	 assert(bra.orbord[pC] < bra.orbord[qC]);
	 int pos = Pr_pLqL2pos.at(make_pair(pC,qC));
	 auto& rop = rqops_P[pos];
         qsym rsym = H.sym - cop_cc.sym;
	 assert(rsym == rop.sym);
	 H += oper_kernel_OcOr(bsite,ksite,cop_cc,rop);
	 H += oper_kernel_OrOc(bsite,ksite,rop.T(),cop_cc.T());
      }
      // 6. Q: B_pCsC^C*Q_pCsC^R
      map<pair<int,int>,int> Qr_pLsL2pos;
      for(int idx=0; idx<rqops_Q.size(); idx++){
         int pL = rqops_Q[idx].index[0];
         int sL = rqops_Q[idx].index[1];
         Qr_pLsL2pos[make_pair(pL,sL)] = idx;
      }
      for(const auto& cop_ca : cqops_ca){
	 int pC = cop_ca.index[0];
	 int sC = cop_ca.index[1];
	 int pos = Qr_pLsL2pos.at(make_pair(pC,sC));
	 auto& rop = rqops_Q[pos];
	 qsym rsym = H.sym - cop_ca.sym;
	 assert(rsym == rop.sym);
	 H += oper_kernel_OcOr(bsite,ksite,cop_ca,rop);
      }
   } // ifAB
   qops.push_back(H);
   string fname = oper_fname(scratch, p, "ropH"); 
   oper_save(fname, qops);
   /*
   // debug against fci::get_Hij
   if(ip==0){
      H.print("H",2);
      auto Hmat = H.to_matrix();
      matrix Hm;
      Hm.load("fci_Hmat");
      auto diffH = Hmat-Hm;
      diffH.print("diffH");
      cout << normF(diffH) << endl;
   }
   */
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropH ifAB=" << ifAB
	   << " : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,int2e,int1e,scratch,"H");
   }
}
