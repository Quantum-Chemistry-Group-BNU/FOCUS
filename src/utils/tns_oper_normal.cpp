#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_rightC(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
			     const string scratch){
   cout << "tns::oper_renorm_rightC ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops, rqops, cqops;
   string fname0r, fname0c;
   // C: build / load
   if(ifload/2 == 0){
      cqops = tns::oper_dot_c(k);
   }else if(ifload/2 == 1){
      assert(j == 0);
      auto pc = make_pair(i,1);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops);
   }
   // R: build /load
   if(ifload%2 == 0){
      int k0 = bra.rsupport.at(p0)[0];
      rqops = tns::oper_dot_c(k0);
   }else if(ifload%2 == 1){
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops);
   }
   // kernel for computing renormalized ap^+
   // pR^+ 
   for(const auto& rop : rqops){
      auto qt2 = oper_kernel_right_IcOr(bsite,ksite,rop);
      qops.push_back(qt2);
   }
   // pC^+ 
   for(const auto& cop : cqops){
      auto qt2 = oper_kernel_right_OcIr(bsite,ksite,cop);
      qops.push_back(qt2);
   }
   string fname = oper_fname(scratch, p, "rightC");
   oper_save(fname, qops);
}

void tns::oper_renorm_rightB(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
			     const string scratch){
   cout << "tns::oper_renorm_rightB ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops, rqops_ca, rqops_c, cqops_ca, cqops_c;
   string fname0r, fname0c;
   // C: build / load
   if(ifload/2 == 0){
      cqops_ca = tns::oper_dot_ca(k);
      cqops_c  = tns::oper_dot_c(k);
   }else if(ifload/2 == 1){
      assert(j == 0);
      auto pc = make_pair(i,1);
      fname0c = oper_fname(scratch, pc, "rightB");
      oper_load(fname0c, cqops_ca);
      fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops_c);
   }
   // R: build /load
   if(ifload%2 == 0){
      int k0 = bra.rsupport.at(p0)[0];
      rqops_ca = tns::oper_dot_ca(k0);
      rqops_c  = tns::oper_dot_c(k0);
   }else if(ifload%2 == 1){
      fname0r = oper_fname(scratch, p0, "rightB");
      oper_load(fname0r, rqops_ca);
      fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops_c);
   }
   // kernel for computing renormalized ap^+aq
   qtensor2 qt2;
   // Ic * pR^+qR 
   for(const auto& rop_ca : rqops_ca){
      qt2 = oper_kernel_right_IcOr(bsite,ksite,rop_ca);
      qops.push_back(qt2);
   }
   // pC^+qC * Ir
   for(const auto& cop_ca : cqops_ca){
      qt2 = oper_kernel_right_OcIr(bsite,ksite,cop_ca);
      qops.push_back(qt2);
   }
   // pC^+ * qR and pR^+*qC = -qC*pR^+
   for(const auto& cop_c : cqops_c){
      for(const auto& rop_c : rqops_c){
         qt2 = oper_kernel_right_OcOr(bsite,ksite,cop_c,rop_c.transpose()); 
         qops.push_back(qt2);
	 qt2 = oper_kernel_right_OrOc(bsite,ksite,rop_c,cop_c.transpose());
         qops.push_back(qt2);
      }
   }
   string fname = oper_fname(scratch, p, "rightB"); 
   oper_save(fname, qops);

   // debug
   if(i==0){
     int nb = bra.nphysical*2;
     matrix rdmA(nb,nb),rdmB(nb,nb),rdmC(nb,nb);
     for(const auto& op : qops){
        int r = op.index[0];
        int s = op.index[1];
        rdmA(r,s) = op.to_matrix()(0,0);
        rdmB(r,s) = op.to_matrix()(0,1);
        rdmC(r,s) = op.to_matrix()(1,2);
     }
     cout << "trA=" << rdmA.trace() << " |A|=" << normF(rdmA)
	  << " A-A.t=" << normF(rdmA-rdmA.transpose()) << endl;
     cout << "trA=" << rdmB.trace() << " |A|=" << normF(rdmB)
	  << " A-A.t=" << normF(rdmB-rdmB.transpose()) << endl;
     cout << "trA=" << rdmC.trace() << " |A|=" << normF(rdmC)
	  << " A-A.t=" << normF(rdmC-rdmC.transpose()) << endl;
     cout << qops.size() << endl;

     matrix rdm1a,rdm1b,rdm1c;
     rdm1a.load("fci_rdm1a");
     rdm1b.load("fci_rdm1b");
     rdm1c.load("fci_rdm1c");
     cout << "trA=" << rdm1a.trace() << " |A|=" << normF(rdm1a)
	  << " A-A.t=" << normF(rdm1a-rdm1a.transpose()) << endl;
     cout << "trA=" << rdm1b.trace() << " |A|=" << normF(rdm1b)
	  << " A-A.t=" << normF(rdm1b-rdm1b.transpose()) << endl;
     cout << "trA=" << rdm1c.trace() << " |A|=" << normF(rdm1c)
	  << " A-A.t=" << normF(rdm1c-rdm1c.transpose()) << endl;

     cout << "diff=" << normF(rdmA-rdm1a) << endl;
     cout << "diff=" << normF(rdmB-rdm1b) << endl;
     cout << "diff=" << normF(rdmC-rdm1c) << endl;
   }

}

void tns::oper_renorm_rightA(const comb& bra,
			     const comb& ket,
		             const comb_coord& p,
		             const comb_coord& p0,
			     const int ifload,
			     const string scratch){
   cout << "tns::oper_renorm_rightA ifload=" << ifload << endl;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   int i = p.first, j = p.second, k = bra.topo[i][j];
   qopers qops, rqops_cc, rqops_c, cqops_cc, cqops_c;
   string fname0r, fname0c;
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
   qtensor2 qt2;
   // Ic * pR^+qR^+ (p<q) 
   for(const auto& rop_cc : rqops_cc){
      qt2 = oper_kernel_right_IcOr(bsite,ksite,rop_cc);
      qops.push_back(qt2);
   }
   // pC^+qC^+ * Ir
   for(const auto& cop_cc : cqops_cc){
      qt2 = oper_kernel_right_OcIr(bsite,ksite,cop_cc); 
      qops.push_back(qt2);
   }
   // pC^+ * qR^+ 
   for(const auto& cop_c : cqops_c){
      for(const auto& rop_c : rqops_c){
	 qt2 = oper_kernel_right_OcOr(bsite,ksite,cop_c,rop_c);
         qops.push_back(qt2);
      }
   }
   string fname = oper_fname(scratch, p, "rightA"); 
   oper_save(fname, qops);
}
