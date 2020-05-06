#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_rightQ(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
	                     const integral::two_body& int2e,
	                     const integral::one_body& int1e,
			     const string scratch){
   cout << "tns::oper_renorm_rightQ ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.topo[ip][jp];
   qopers qops, qops_ca;
   string fname0 = oper_fname(scratch, p, "rightB");
   oper_load(fname0, qops_ca);
   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb, Qab ~ b^+a, Qba ~ a^+b
   qtensor2 Qss(qsym(0, 0), bsite.qcol, ksite.qcol, 2);
   qtensor2 Qab(qsym(0,-1), bsite.qcol, ksite.qcol, 2);
   qtensor2 Qba(qsym(0, 1), bsite.qcol, ksite.qcol, 2);
   for(int korb_p : bra.lsupport.at(p)){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_s : bra.lsupport.at(p)){
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
   // Qps = <pq||sr> aq^+ar
   for(auto& qop : qops){
      int orb_p = qop.index[0];
      int orb_s = qop.index[1];
      for(const auto& op_ca : qops_ca){
         if(qop.sym != op_ca.sym) continue;
         int orb_q = op_ca.index[0];
         int orb_r = op_ca.index[1];
         qop += int2e.getAnti(orb_p,orb_q,orb_s,orb_r) * op_ca;
      } // ps
   } // qr
   string fname = oper_fname(scratch, p, "rightQ"); 
   oper_save(fname, qops);
}

void tns::oper_renorm_rightP(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
	                     const integral::two_body& int2e,
	                     const integral::one_body& int1e,
			     const string scratch){
   cout << "tns::oper_renorm_rightP ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.topo[ip][jp];
   qopers qops, qops_cc;
   string fname0 = oper_fname(scratch, p, "rightA");
   oper_load(fname0, qops_cc);
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   qtensor2 Paa(qsym(-2,-2), bsite.qcol, ksite.qcol, 2);
   qtensor2 Pbb(qsym(-2, 0), bsite.qcol, ksite.qcol, 2);
   qtensor2 Pab(qsym(-2,-1), bsite.qcol, ksite.qcol, 2);
   for(int korb_p : bra.lsupport.at(p)){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_q : bra.lsupport.at(p)){
	 int qa = 2*korb_q, qb = qa+1;
         Paa.index[0] = pa;
         Paa.index[1] = qa;
	 if(pa<qa) qops.push_back(Paa);
         Pbb.index[0] = pb;
         Pbb.index[1] = qb;
	 if(pb<qb) qops.push_back(Pbb);
         Pab.index[0] = pa;
         Pab.index[1] = qb;
	 if(pa<qb) qops.push_back(Pab);
         // for ababab orbital ordering, Pba is also needed.
	 Pab.index[0] = pb;
         Pab.index[1] = qa;
	 if(pb<qa) qops.push_back(Pab);
      }
   }
   // Ppq = <pq||sr> aras [r>s] (p<q)
   for(auto& qop : qops){
      int orb_p = qop.index[0];
      int orb_q = qop.index[1];
      // (as^+ar^+)^+ (s<r) => aras
      for(const auto& op_cc : qops_cc){
         if(qop.sym != -op_cc.sym) continue;
         int orb_s = op_cc.index[0];
         int orb_r = op_cc.index[1];
         qop += int2e.getAnti(orb_p,orb_q,orb_s,orb_r) * op_cc.T();
      } // ps
   } // qr
   string fname = oper_fname(scratch, p, "rightP"); 
   oper_save(fname, qops);
}

void tns::oper_renorm_rightS(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
	                     const integral::two_body& int2e,
	                     const integral::one_body& int1e,
			     const string scratch){
   cout << "tns::oper_renorm_rightS ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.topo[ip][jp];
   qopers qops;
   qopers cqops_c, cqops_cc, cqops_ca, cqops_S;
   qopers rqops_c, rqops_cc, rqops_ca, rqops_S, rqops_P, rqops_Q;
   string fname0r, fname0c;
   auto lsupp = bra.lsupport.at(p);
   // C: build / load
   if(ifload/2 == 0){
      // type AB decomposition (build,build)
      cqops_c  = oper_dot_c(kp);
      cqops_cc = oper_dot_cc(kp);
      cqops_ca = oper_dot_ca(kp);
      oper_dot_S(kp, lsupp, int2e, int1e, cqops_c, cqops_S);
   }else if(ifload/2 == 1){
      // type PQ decomposition (load,build)
      assert(jp == 0);
      auto pc = make_pair(ip,1);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "rightA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "rightB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "rightS");
      oper_load(fname0c, cqops_S);
   }
   // R: build /load
   int rtype = 0; // =0, AB; =1, PQ;
   if(ifload%2 == 0){
      rtype = 0;
      int kp0 = bra.rsupport.at(p0)[0];
      rqops_c  = oper_dot_c(kp0);
      rqops_cc = oper_dot_cc(kp0);
      rqops_ca = oper_dot_ca(kp0);
      oper_dot_S(kp0, lsupp, int2e, int1e, rqops_c, rqops_S);
   }else if(ifload%2 == 1){
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops_c);
      if(jp == 0 && ip <= bra.iswitch){ 
	 // left half of the backbone [type-PQ]
         rtype = 1;
         fname0r = oper_fname(scratch, p0, "rightP");
         oper_load(fname0r, rqops_P);
         fname0r = oper_fname(scratch, p0, "rightQ");
         oper_load(fname0r, rqops_Q);
      }else{
         // branch & left half of the backbone [type-AB]
	 rtype = 0;
         fname0r = oper_fname(scratch, p0, "rightA");
         oper_load(fname0r, rqops_cc);
         fname0r = oper_fname(scratch, p0, "rightB");
         oper_load(fname0r, rqops_ca);
      }
      fname0r = oper_fname(scratch, p0, "rightS");
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
   // kernel for computing renormalized Sp^{CR} [6 terms]
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
	 qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 1);
	 bool ifcal = false;
	 for(const auto& rop_c : rqops_c){
	    if(rsym != -rop_c.sym) continue;
	    ifcal = true;
	    int rR = rop_c.index[0];
	    rop -= int2e.getAnti(pL,qC,sC,rR)*rop_c.T();
	 }
	 if(ifcal) qops[i] += oper_kernel_OcOr(bsite,ksite,cop_ca,rop);
      }
      // 4. qRrCsC: A[sC,rC]^+*[<pLqR||sCrC>qR^+] (s<r)
      for(const auto& cop_cc : cqops_cc){
         int sC = cop_cc.index[0];
	 int rC = cop_cc.index[1];
	 // rop = sum_qR <pLqR||sCrC>qR^+
	 qsym rsym = qops[i].sym + cop_cc.sym;
	 qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 1);
	 bool ifcal = false;
	 for(const auto& rop_c : rqops_c){
	    if(rsym != rop_c.sym) continue;
	    ifcal = true;
	    int qR = rop_c.index[0];
            rop += int2e.getAnti(pL,qR,sC,rC)*rop_c; 
	 }
	 if(ifcal) qops[i] += oper_kernel_OcOr(bsite,ksite,cop_cc.T(),rop);
      }
   } // pL
   // type-AB/PQ decomposition for SpL^CR
   if(rtype == 0){
      cout << "decomposition type-AB" << endl;
      // type-AB decomposition
      for(int i=0; i<qops.size(); i++){
         int pL = qops[i].index[0];
	 // 5. qCrRsR: A: sum_qC qC^+ * sum_rRsR <pLqC||sRrR> rRsR (r>s)
         for(const auto& cop_c : cqops_c){
	    int qC = cop_c.index[0];
	    // rop = sum_rRsR <pLqC||sRrR> rRsR (r>s)
	    qsym rsym = qops[i].sym - cop_c.sym;
	    qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
	    bool ifcal = false; 
	    for(const auto& rop_cc : rqops_cc){
	       if(rsym != -rop_cc.sym) continue;
	       ifcal = true;
	       int rR = rop_cc.index[1];
	       int sR = rop_cc.index[0];
	       rop += int2e.getAnti(pL,qC,sR,rR)*rop_cc.T();
	    }
	    if(ifcal) qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c,rop);
         }
	 // 6. qRrRsC: B: sum_sC sC * sum_qRrR <pLqR||sCrR> qR^+rR
	 for(const auto& cop_c : cqops_c){
	    int sC = cop_c.index[0];
	    // rop = sum_qRrR <pLqR||sCrR> qR^+rR
	    qsym rsym = qops[i].sym + cop_c.sym;
	    qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
	    bool ifcal = false;
	    for(const auto& rop_ca : rqops_ca){
	       if(rsym != rop_ca.sym) continue;
	       ifcal = true;
	       int qR = rop_ca.index[0];
	       int rR = rop_ca.index[1];
	       rop += int2e.getAnti(pL,qR,sC,rR)*rop_ca;
	    }
	    if(ifcal) qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop);
	 }
      } // pL
   }else{
      cout << "decomposition type-PQ" << endl;
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
	 // 5. qCrRsR: P: sum_qC qC^+ * P_pLqC
  	 for(const auto& cop_c : cqops_c){
	    int qC = cop_c.index[0];
	    int pos = Pr_pLqL2pos.at(make_pair(pL,qC));
	    auto& rop = rqops_P[pos];
	    qsym rsym = qops[i].sym - cop_c.sym;
 	    assert(rsym == rop.sym);
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c,rop);
	 } 
	 // 6. qRrRsC: Q: sum_sC sC * Q_pLsC
	 for(const auto& cop_c : cqops_c){
	    int sC = cop_c.index[0];
	    int pos = Qr_pLsL2pos.at(make_pair(pL,sC));
	    auto& rop = rqops_Q[pos];
	    qsym rsym = qops[i].sym + cop_c.sym;
	    assert(rsym == rop.sym);
	    qops[i] += oper_kernel_OcOr(bsite,ksite,cop_c.T(),rop);
	 }
      }	// pL 
   } // rtype
   string fname = oper_fname(scratch, p, "rightS"); 
   oper_save(fname, qops);
}

void tns::oper_renorm_rightH(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
	                     const integral::two_body& int2e,
	                     const integral::one_body& int1e,
			     const string scratch){
   cout << "tns::oper_renorm_rightH ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.topo[ip][jp];
   qopers qops;
   qopers cqops_c, cqops_cc, cqops_ca, cqops_S, cqops_H;
   qopers rqops_c, rqops_cc, rqops_ca, rqops_S, rqops_H, rqops_P, rqops_Q;
   string fname0r, fname0c;
   // C: build / load
   if(ifload/2 == 0){
      // type AB decomposition (build,build)
      cqops_c  = oper_dot_c(kp);
      cqops_cc = oper_dot_cc(kp);
      cqops_ca = oper_dot_ca(kp);
      auto rsupp = bra.rsupport.at(p);
      oper_dot_S(kp, rsupp, int2e, int1e, cqops_c, cqops_S);
      oper_dot_H(kp, int2e, int1e, cqops_ca, cqops_H);
   }else if(ifload/2 == 1){
      // type PQ decomposition (load,build)
      assert(jp == 0);
      auto pc = make_pair(ip,1);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops_c);
      fname0c = oper_fname(scratch, pc, "rightA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "rightB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "rightS");
      oper_load(fname0c, cqops_S);
      fname0c = oper_fname(scratch, pc, "rightH");
      oper_load(fname0c, cqops_H);
   }
   // R: build /load
   int rtype = 0; // =0, AB; =1, PQ;
   if(ifload%2 == 0){
      rtype = 0;
      int kp0 = bra.rsupport.at(p0)[0];
      rqops_c  = oper_dot_c(kp0);
      rqops_cc = oper_dot_cc(kp0);
      rqops_ca = oper_dot_ca(kp0);
      vector<int> csupp({kp});
      oper_dot_S(kp0, csupp, int2e, int1e, rqops_c, rqops_S);
      oper_dot_H(kp0, int2e, int1e, rqops_ca, rqops_H);
   }else if(ifload%2 == 1){
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops_c);
      if(jp == 0 && ip <= bra.iswitch){ 
	 // left half of the backbone [type-PQ]
         rtype = 1;
         fname0r = oper_fname(scratch, p0, "rightP");
         oper_load(fname0r, rqops_P);
         fname0r = oper_fname(scratch, p0, "rightQ");
         oper_load(fname0r, rqops_Q);
      }else{
         // branch & left half of the backbone [type-AB]
	 rtype = 0;
         fname0r = oper_fname(scratch, p0, "rightA");
         oper_load(fname0r, rqops_cc);
         fname0r = oper_fname(scratch, p0, "rightB");
         oper_load(fname0r, rqops_ca);
      }
      fname0r = oper_fname(scratch, p0, "rightS");
      oper_load(fname0r, rqops_S);
      fname0r = oper_fname(scratch, p0, "rightH");
      oper_load(fname0r, rqops_H);
   }

   // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
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
   // type-AB/PQ decomposition for H 
   if(rtype == 0){
      cout << "decomposition type-AB" << endl;
      // type-AB decomposition
      // 5. A: A_pCqC^C*P_pCqC^R + h.c.
      for(const auto& cop_cc : cqops_cc){
         int pC = cop_cc.index[0];
	 int qC = cop_cc.index[1];
         // rop = P_pCqC^R = sum_rRsR <pCqC||sRrR> rRsR (r>s)
         qsym rsym = H.sym - cop_cc.sym;
         qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
         bool ifcal = false; 
         for(const auto& rop_cc : rqops_cc){
            if(rsym != -rop_cc.sym) continue;
            ifcal = true;
            int rR = rop_cc.index[1];
            int sR = rop_cc.index[0];
            rop += int2e.getAnti(pC,qC,sR,rR)*rop_cc.T();
         }
         if(ifcal){
	    H += oper_kernel_OcOr(bsite,ksite,cop_cc,rop);
	    H += oper_kernel_OrOc(bsite,ksite,rop.T(),cop_cc.T());
	 }
      }
      // 6. B: Q: B_pCsC^C*Q_pCsC^R
      for(const auto& cop_ca : cqops_ca){
         int pC = cop_ca.index[0];
	 int sC = cop_ca.index[1];
         // rop = Q_pCsC^R = sum_qRrR <pCqR||sCrR> qR^+rR
         qsym rsym = H.sym - cop_ca.sym;
         qtensor2 rop(rsym, bsite.qrow, ksite.qrow, 2);
         bool ifcal = false;
         for(const auto& rop_ca : rqops_ca){
            if(rsym != rop_ca.sym) continue;
            ifcal = true;
            int qR = rop_ca.index[0];
            int rR = rop_ca.index[1];
            rop += int2e.getAnti(pC,qR,sC,rR)*rop_ca;
         }
         if(ifcal) H += oper_kernel_OcOr(bsite,ksite,cop_ca,rop);
      }
   }else{
      cout << "decomposition type-PQ" << endl;
      // type-PQ decomposition
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
   } // rtype
   qops.push_back(H);
   string fname = oper_fname(scratch, p, "rightH"); 
   oper_save(fname, qops);

   if(ip==0){
      H.print("H",2);
      auto Hmat = H.to_matrix();
      matrix Hm;
      Hm.load("fci_Hmat");
      auto diffH = Hmat-Hm;
      diffH.print("diffH");
      cout << normF(diffH) << endl;
   }

}
