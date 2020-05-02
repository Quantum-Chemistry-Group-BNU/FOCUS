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
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops, qops_ca;
   string fname0 = oper_fname(scratch, p, "rightB");
   oper_load(fname0, qops_ca);
   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb, Qab ~ b^+a, Qba ~ a^+b
   qtensor2 Qss(qsym(0, 0),bsite.qrow,ksite.qrow,2);
   qtensor2 Qab(qsym(0,-1),bsite.qrow,ksite.qrow,2);
   qtensor2 Qba(qsym(0, 1),bsite.qrow,ksite.qrow,2);
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
         if(qop.msym != op_ca.msym) continue;
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
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops, qops_cc;
   string fname0 = oper_fname(scratch, p, "rightA");
   oper_load(fname0, qops_cc);
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   qtensor2 Paa(qsym(-2,-2),bsite.qrow,ksite.qrow,2);
   qtensor2 Pbb(qsym(-2, 0),bsite.qrow,ksite.qrow,2);
   qtensor2 Pab(qsym(-2,-1),bsite.qrow,ksite.qrow,2);
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
         if(qop.msym != -op_cc.msym) continue;
         int orb_s = op_cc.index[0];
         int orb_r = op_cc.index[1];
         qop += int2e.getAnti(orb_p,orb_q,orb_s,orb_r) * op_cc.transpose();
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
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops;
   qopers cqops_caa, cqops_ca, cqops_cc, cqops_c, cqops_S;
   qopers rqops_caa, rqops_ca, rqops_cc, rqops_c, rqops_S, rqops_Q, rqops_P;
   string fname0r, fname0c;
   // C: build / load
   if(ifload/2 == 0){
      // type AB decomposition (build,build)
      cqops_caa = tns::oper_dot_caa(k);
      cqops_ca  = tns::oper_dot_ca(k);
      cqops_cc  = tns::oper_dot_cc(k);
      cqops_c   = tns::oper_dot_c(k);
      oper_dot_rightS_loc(k,bra.lsupport.at(p),cqops_c,cqops_caa,int2e,int1e,cqops_S);
   }else if(ifload/2 == 1){
      // type PQ decomposition (load,build)
      assert(j == 0);
      auto pc = make_pair(i,1);
      fname0c = oper_fname(scratch, pc, "rightS");
      oper_load(fname0c, cqops_S);
      fname0c = oper_fname(scratch, pc, "rightB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "rightA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops_c);
   }
   // R: build /load
   if(ifload%2 == 0){
      int k0 = bra.rsupport.at(p0)[0];
      rqops_caa = tns::oper_dot_caa(k0);
      rqops_ca  = tns::oper_dot_ca(k0);
      rqops_cc  = tns::oper_dot_cc(k0);
      rqops_c   = tns::oper_dot_c(k0);
      oper_dot_rightS_loc(k0,bra.lsupport.at(p),rqops_c,rqops_caa,int2e,int1e,rqops_S);
   }else if(ifload%2 == 1){
      // type PQ decomposition (load,build)
      fname0r = oper_fname(scratch, p0, "rightS");
      oper_load(fname0r, rqops_S);
      fname0r = oper_fname(scratch, p0, "rightQ");
      oper_load(fname0r, rqops_Q);
      fname0r = oper_fname(scratch, p0, "rightP");
      oper_load(fname0r, rqops_P);
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops_c);
   }
   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   qtensor2 Sa(qsym(-1,-1),bsite.qrow,ksite.qrow,1);
   qtensor2 Sb(qsym(-1, 0),bsite.qrow,ksite.qrow,1);
   for(int korb_p : bra.lsupport.at(p)){
      int pa = 2*korb_p, pb = pa+1;
      Sa.index[0] = pa;
      qops.push_back(Sa);
      Sb.index[0] = pb;
      qops.push_back(Sb);
   }
   // kernel for computing renormalized Sp^{CR} [6 terms]
   qtensor2 qt2;
   for(int i=0; i<qops.size(); i++){
      // local term: Sc*Ir
      qt2 = oper_kernel_right_OcIr(bsite,ksite,cqops_S[i]);
      qops[i] += qt2;
   //   // local term: Ic*Sr
   //   auto& rop_S = rqops_S[i];
   //   qt3 = contract_qt3_qt2_r(ksite,rop_S);
   //   qt2 = contract_qt3_qt3_cr(bsite,qt3,true);
   //   qops[i] += qt2;
   } // pL
   exit(1);

//   if(ifload%2 == 0){
//cqops_S, cqops_ca, cqops_cc, cqops_c,
//rqops_S, rqops_ca, rqops_cc, rqops_c,
//   }else if(ifload%2 == 1{
//cqops_S, cqops_ca, cqops_cc, cqops_c,
//rqops_S, rqops_Q, rqops_P, rqops_c,
//   }
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
   cout << "tns::oper_renorm_rightP ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   qopers qops, rqops_cc, rqops_c, cqops_cc, cqops_c;
   string fname = oper_fname(scratch, p, "rightH"); 
   string fname0r, fname0c;
   int i = p.first, j = p.second, k = bra.topo[i][j];
   // C: build / load
   if(ifload/2 == 0){
      cqops_cc = tns::oper_dot_cc(k);
      cqops_c  = tns::oper_dot_c(k);
   }else if(ifload/2 == 1){
      assert(j == 0);
      auto pc = make_pair(i,1);
      fname0c = oper_fname(scratch, pc, "rightA");
      oper_load(fname0c, cqops_cc);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops_c);
   }
   // R: build /load
   if(ifload%2 == 0){
      int k0 = bra.rsupport.at(p0)[0];
      rqops_cc = tns::oper_dot_cc(k0);
      rqops_c  = tns::oper_dot_c(k0);
   }else if(ifload%2 == 1){
      fname0r = oper_fname(scratch, p0, "rightA");
      oper_load(fname0r, rqops_cc);
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops_c);
   }
   // kernel for computing renormalized ap^+aq
   oper_kernel_rightA(bsite,ksite,
	   	      cqops_cc,cqops_c,
	   	      rqops_cc,rqops_c,
	   	      qops);
   oper_save(fname, qops);
}
