#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include "../core/matrix.h"
#include "../core/linalg.h"

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
   qopers qops, rqops, cqops;
   string fname = oper_fname(scratch, p, "rightC");
   string fname0r, fname0c;
   int i = p.first, j = p.second, k = bra.topo[i][j];
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
   oper_renorm_rightC_kernel(bsite,ksite,cqops,rqops,qops);
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
   qopers qops, rqops_ca, rqops_c, cqops_ca, cqops_c;
   string fname = oper_fname(scratch, p, "rightB"); 
   string fname0r, fname0c;
   int i = p.first, j = p.second, k = bra.topo[i][j];
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
   oper_renorm_rightB_kernel(bsite,ksite,
		   	     cqops_ca,cqops_c,
			     rqops_ca,rqops_c,
			     qops);

   // debug
   if(i==0){
      int nb = bra.nphysical*2;
      matrix rdmA(nb,nb),rdmB(nb,nb),rdmC(nb,nb);
      for(const auto& op : qops){
         int r = op.index[0];
         int s = op.index[1];
         rdmA(r,s) = op.to_matrix()(0,0);
         rdmB(r,s) = op.to_matrix()(1,0);
         rdmC(r,s) = op.to_matrix()(0,2);
      }
      cout << rdmA.trace() << endl;
      cout << rdmB.trace() << endl;
      cout << rdmC.trace() << endl;
      cout << qops.size() << endl;

      matrix rdm1a,rdm1b,rdm1c;
      rdm1a.load("fci_rdm1_00");
      rdm1b.load("fci_rdm1_10");
      rdm1c.load("fci_rdm1_02");
      cout << normF(rdmA-rdm1a) << endl;
      cout << normF(rdmB-rdm1b) << endl;
      cout << normF(rdmC-rdm1c) << endl;
   }

   oper_save(fname, qops);
}

void tns::oper_renorm_right(const comb& bra,
			    const comb& ket,
		            const comb_coord& p,
		            const comb_coord& p0,
			    const string scratch){
   cout << "\ntns::oper_renorm_right" << endl;
   int i = p.first, j = p.second;
   int i0 = p0.first, j0 = p0.second; 
   int tp = bra.type.at(p);
   int tp0 = bra.type.at(p0);
   cout << "p=(" << i << "," << j << ")[" << bra.topo[i][j] << "] "
	<< "p0=(" << i0 << "," << j0 << ")[" << bra.topo[i0][j0] << "] " 
	<< "type=[" << tp << "," << tp0 << "]" << endl;
   auto kind = make_pair(tp,tp0);
   int ifload;
   if(kind == make_pair(1,0) || kind == make_pair(2,0)){
      ifload = 0; // (C:false,R:false)
   }else if(kind == make_pair(1,1) || kind == make_pair(1,3) ||
	    kind == make_pair(0,1) || kind == make_pair(0,3) ||
            kind == make_pair(2,2)){
      ifload = 1; // (C:false,R:true)
   }else if(kind == make_pair(3,0)){
      ifload = 2; // (C:true,R:false)
   }else if(kind == make_pair(3,1) || kind == make_pair(3,3)){
      ifload = 3; // (C:true,R:true)
   }else{
      cout << "error: no such case! (tp,tp0)=" << tp << "," << tp0 << endl;
      exit(1);
   }
   oper_renorm_rightC(bra,ket,p,p0,ifload,scratch);
   oper_renorm_rightB(bra,ket,p,p0,ifload,scratch);
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   int nbackbone = bra.nbackbone;
   // loop over internal nodes
   for(int i=nbackbone-2; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
	 auto p0 = make_pair(i+1,0);    
	 oper_renorm_right(bra,ket,p,p0,scratch);
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-2; j>=1; j--){
	    auto pj = make_pair(i,j);
	    auto p0 = make_pair(i,j+1);    
	    oper_renorm_right(bra,ket,pj,p0,scratch);
	 } // j
	 auto p0 = make_pair(i+1,0);
	 oper_renorm_right(bra,ket,p,p0,scratch);
      }else{
	 cout << "error: tp=" << tp << endl;
	 exit(1);
      }
   } // i
}
