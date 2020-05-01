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
   string fname = oper_fname(scratch, p, "rightQ"); 
   string fname0 = oper_fname(scratch, p, "rightB");
   oper_load(fname0, qops_ca);
   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb
   qtensor2 Qss;
   Qss.msym = qsym(0,0);
   Qss.qrow = bsite.qrow;
   Qss.qcol = ksite.qrow;
   Qss.init_qblocks();
   // Qab ~ b^+a
   qtensor2 Qab;
   Qab.msym = qsym(0,-1);
   Qab.qrow = bsite.qrow;
   Qab.qcol = ksite.qrow;
   Qab.init_qblocks();
   // Qba ~ a^+b
   qtensor2 Qba;
   Qba.msym = qsym(0,1);
   Qba.qrow = bsite.qrow;
   Qba.qcol = ksite.qrow;
   Qba.init_qblocks();
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
         // <pq||sr> = [ps|qr] - [pr|qs]
         double vpqsr = int2e.get(orb_p,orb_s,orb_q,orb_r)
   	              - int2e.get(orb_p,orb_r,orb_q,orb_s);
         qop += vpqsr * op_ca;
      } // ps
   } // qr
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
   string fname = oper_fname(scratch, p, "rightP"); 
   string fname0 = oper_fname(scratch, p, "rightA");
   oper_load(fname0, qops_cc);
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   // Paa
   qtensor2 Paa;
   Paa.msym = qsym(-2,-2);
   Paa.qrow = bsite.qrow;
   Paa.qcol = ksite.qrow;
   Paa.init_qblocks();
   // Pbb
   qtensor2 Pbb;
   Pbb.msym = qsym(-2,0);
   Pbb.qrow = bsite.qrow;
   Pbb.qcol = ksite.qrow;
   Pbb.init_qblocks();
   // Pab
   qtensor2 Pab;
   Pab.msym = qsym(-2,-1);
   Pab.qrow = bsite.qrow;
   Pab.qcol = ksite.qrow;
   Pab.init_qblocks();
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
         // <pq||sr> = [ps|qr] - [pr|qs]
         double vpqsr = int2e.get(orb_p,orb_s,orb_q,orb_r)
   	              - int2e.get(orb_p,orb_r,orb_q,orb_s);
         qop += vpqsr * op_cc.transpose();
      } // ps
   } // qr
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
   qopers qops, rqops_cc, rqops_c, cqops_cc, cqops_c;
   string fname = oper_fname(scratch, p, "rightP"); 
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
   string fname = oper_fname(scratch, p, "rightP"); 
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
