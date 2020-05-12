#include "../settings/global.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_ropC(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropC" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops, rqops, cqops;
   string fname0r, fname0c;
   // C: build / load
   if(bra.ifbuild_c(p)){
      cqops = oper_dot_c(kp);
   }else{
      auto pc = bra.get_c(p);
      fname0c = oper_fname(scratch, pc, "ropC");
      oper_load(fname0c, cqops);
   }
   // R: build /load
   auto pr = bra.get_r(p);
   if(bra.ifbuild_r(p)){
      int kpr = bra.get_kp(pr);
      rqops = oper_dot_c(kpr);
   }else{
      fname0r = oper_fname(scratch, pr, "ropC");
      oper_load(fname0r, rqops);
   }
   // kernel for computing renormalized ap^+
   // Ic*pR^+ 
   for(const auto& rop : rqops){
      auto qt2 = oper_kernel_IcOr(bsite,ksite,rop);
      qops.push_back(qt2);
   }
   // pC^+*Ir 
   for(const auto& cop : cqops){
      auto qt2 = oper_kernel_OcIr(bsite,ksite,cop);
      qops.push_back(qt2);
   }
   string fname = oper_fname(scratch, p, "ropC");
   oper_save(fname, qops);
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropC : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,scratch,"C");
   }
}

void tns::oper_renorm_ropA(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropA" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops, rqops_cc, rqops_c, cqops_cc, cqops_c;
   string fname0r, fname0c;
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
   // R: build /load
   auto pr = bra.get_r(p);
   if(bra.ifbuild_r(p)){
      int kpr = bra.get_kp(pr);
      rqops_c  = oper_dot_c(kpr);
      rqops_cc = oper_dot_cc(kpr);
   }else{
      fname0r = oper_fname(scratch, pr, "ropC");
      oper_load(fname0r, rqops_c);
      fname0r = oper_fname(scratch, pr, "ropA");
      oper_load(fname0r, rqops_cc);
   }
   // kernel for computing renormalized Apq=ap^+aq^+
   qtensor2 qt2;
   // Ic * pR^+qR^+ (p<q) 
   for(const auto& rop_cc : rqops_cc){
      qt2 = oper_kernel_IcOr(bsite,ksite,rop_cc);
      qops.push_back(qt2);
   }
   // pC^+qC^+ * Ir
   for(const auto& cop_cc : cqops_cc){
      qt2 = oper_kernel_OcIr(bsite,ksite,cop_cc); 
      qops.push_back(qt2);
   }
   // pC^+ * qR^+
   for(const auto& cop_c : cqops_c){
      int pC = cop_c.index[0];
      for(const auto& rop_c : rqops_c){
	 int qR = rop_c.index[0];
	 // only store Apq where node[p]<node[q]
	 assert(bra.orbord[pC] < bra.orbord[qR]);
	 qt2 = oper_kernel_OcOr(bsite,ksite,cop_c,rop_c);
         qops.push_back(qt2);
      }
   }
   string fname = oper_fname(scratch, p, "ropA"); 
   oper_save(fname, qops);
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropA : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,scratch,"A");
   }
}

void tns::oper_renorm_ropB(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
			   const string scratch,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropB" << endl;
   auto t0 = global::get_time();
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int ip = p.first, jp = p.second, kp = bra.get_kp(p);
   qopers qops, rqops_ca, rqops_c, cqops_ca, cqops_c;
   string fname0r, fname0c;
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
   // R: build /load
   auto pr = bra.get_r(p);
   if(bra.ifbuild_r(p)){
      int kpr = bra.get_kp(pr);
      rqops_c  = oper_dot_c(kpr);
      rqops_ca = oper_dot_ca(kpr);
   }else{
      fname0r = oper_fname(scratch, pr, "ropC");
      oper_load(fname0r, rqops_c);
      fname0r = oper_fname(scratch, pr, "ropB");
      oper_load(fname0r, rqops_ca);
   }
   // kernel for computing renormalized ap^+aq
   // Ic * pR^+qR 
   for(const auto& rop_ca : rqops_ca){
      auto qt2 = oper_kernel_IcOr(bsite,ksite,rop_ca);
      qops.push_back(qt2);
   }
   // pC^+qC * Ir
   for(const auto& cop_ca : cqops_ca){
      auto qt2 = oper_kernel_OcIr(bsite,ksite,cop_ca);
      qops.push_back(qt2);
   }
   // pC^+ * qR and pR^+*qC = -qC*pR^+
   for(const auto& cop_c : cqops_c){
      for(const auto& rop_c : rqops_c){
         auto qt2a = oper_kernel_OcOr(bsite,ksite,cop_c,rop_c.T()); 
         qops.push_back(qt2a);
	 auto qt2b = oper_kernel_OrOc(bsite,ksite,rop_c,cop_c.T());
         qops.push_back(qt2b);
      }
   }
   string fname = oper_fname(scratch, p, "ropB"); 
   oper_save(fname, qops);
   /*
   // debug
   if(ip==0){
     int nb = bra.nphysical*2;
     matrix rdmA(nb,nb),rdmB(nb,nb),rdmC(nb,nb);
     for(const auto& op : qops){
        int r = op.index[0];
        int s = op.index[1];
        rdmA(r,s) = op.to_matrix()(0,0);
        rdmB(r,s) = op.to_matrix()(2,0);
        rdmC(r,s) = op.to_matrix()(1,2);
     }
     cout << setprecision(10) << endl;
     cout << "trA=" << rdmA.trace() << " |A|=" << normF(rdmA)
	  << " A-A.t=" << normF(rdmA-rdmA.T()) << endl;
     cout << "trA=" << rdmB.trace() << " |A|=" << normF(rdmB)
	  << " A-A.t=" << normF(rdmB-rdmB.T()) << endl;
     cout << "trA=" << rdmC.trace() << " |A|=" << normF(rdmC)
	  << " A-A.t=" << normF(rdmC-rdmC.T()) << endl;
     cout << qops.size() << endl;

     matrix rdm1a,rdm1b,rdm1c;
     rdm1a.load("fci_rdm1a");
     rdm1b.load("fci_rdm1b");
     rdm1c.load("fci_rdm1c");
     cout << "fci_drm1:" << endl;
     cout << "trA=" << rdm1a.trace() << " |A|=" << normF(rdm1a)
	  << " A-A.t=" << normF(rdm1a-rdm1a.T()) << endl;
     cout << "trA=" << rdm1b.trace() << " |A|=" << normF(rdm1b)
	  << " A-A.t=" << normF(rdm1b-rdm1b.T()) << endl;
     cout << "trA=" << rdm1c.trace() << " |A|=" << normF(rdm1c)
	  << " A-A.t=" << normF(rdm1c-rdm1c.T()) << endl;

     cout << "diff=" << normF(rdmA-rdm1a) << endl;
     cout << "diff=" << normF(rdmB-rdm1b) << endl;
     cout << "diff=" << normF(rdmC-rdm1c) << endl;
   }
   */
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropB : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
      oper_rbases(bra,ket,p,scratch,"B");
   }
}
