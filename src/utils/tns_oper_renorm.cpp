#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include "../core/matrix.h"
#include "../core/linalg.h"

using namespace std;
using namespace linalg;
using namespace tns;

string tns::oper_fname(const string scratch, 
  	 	       const comb_coord& p,
		       const string optype){
   string fname = scratch+"/oper_("
	         +to_string(p.first)+","
	         +to_string(p.second)+")_"
		 +optype;
   return fname;
}
      
void tns::oper_renorm_rightC_kernel(const qtensor3& bsite,
		                    const qtensor3& ksite,
		                    const qopers& cqops,
		                    const qopers& rqops,
		                    qopers& qops){
   qtensor3 qt3;
   qtensor2 qt2;
   // pR^+ 
   for(const auto& rop : rqops){
      qt3 = contract_qt3_qt2_r(ksite,rop);
      qt2 = contract_qt3_qt3_cr(bsite,qt3,true);
      qt2.msym = rop.msym;
      qt2.index[0] = rop.index[0];
      qops.push_back(qt2);
   }
   // pC^+ 
   for(const auto& cop : cqops){
      qt3 = contract_qt3_qt2_c(ksite,cop); 
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = cop.msym;
      qt2.index[0] = cop.index[0];
      qops.push_back(qt2);
   }
}

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
   int i = p.first, j = p.second, k = bra.topo[i][j];
   // C: build / load
   if(ifload/2 == 0){
      cqops = tns::oper_dot_c(k);
   }else if(ifload/2 == 1){
      assert(j == 0);
      auto pc = make_pair(i,1);
      string fname0c = oper_fname(scratch, pc, "rightC");
      oper_load(fname0c, cqops);
   }
   // R: build /load
   if(ifload%2 == 0){
      int k0 = bra.rsupport.at(p0)[0];
      rqops = tns::oper_dot_c(k0);
   }else if(ifload%2 == 1){
      string fname0r = oper_fname(scratch, p0, "rightC");
      oper_load(fname0r, rqops);
   }
   // kernel for computing renormalized ap^+
   oper_renorm_rightC_kernel(bsite,ksite,cqops,rqops,qops);
   oper_save(fname, qops);
}

/*
void tns::oper_renorm_B(const comb& bra,
			const comb& ket,
		        const comb_coord& p,
		        const comb_coord& p0,
			const int iop,
			const string scratch){
   cout << "tns::oper_renorm_B iop=" << iop << endl;
   const bool sgnc = true;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   qtensor3 qt3;
   qtensor2 qt2;
   qopers qops;
   string fname = oper_fname(scratch, p, "right_B"); 
   int i = p.first, j = p.second, k = bra.topo[i][j];
   if(iop == 0){
      // C: build, R: build
      auto cop_ca = tns::oper_dot_ca(); 
      auto cop_c = tns::oper_dot_c(); 
      // pR^+qR 
      int k0 = bra.rsupport.at(p0)[0];
      for(const auto& cop : cop_ca){
         qt3 = contract_qt3_qt2_r(ksite,cop);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop.msym;
	 qt2.index[0] = 2*k0 + cop.index[0];
	 qt2.index[1] = 2*k0 + cop.index[1];
	 qops.push_back(qt2);
      }
      // pC^+qR
      for(int s=0; s<2; s++){
         for(int s0=0; s0<2; s0++){
	    qt3 = contract_qt3_qt2_r(ksite,cop_c[s0].transpose());
	    auto ctmp = cop_c[s].col_signed();
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = cop_c[s].msym - cop_c[s0].msym;
	    qt2.index[0] = 2*k + cop_c[s].index[0];
	    qt2.index[1] = 2*k0 + cop_c[s0].index[0];
	    qops.push_back(qt2);
	 }
      }
      // pR^+qC = -qC*pR^+
      for(int s0=0; s0<2; s0++){
         for(int s=0; s<2; s++){
	    qt3 = contract_qt3_qt2_r(ksite,cop_c[s0]);
	    auto ctmp = cop_c[s].transpose().col_signed(-1.0);
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = -cop_c[s].msym + cop_c[s0].msym;
	    qt2.index[0] = 2*k0 + cop_c[s0].index[0];
	    qt2.index[1] = 2*k + cop_c[s].index[0];
	    qops.push_back(qt2);
	 }
      }
      // pC^+qC
      for(int s=0; s<4; s++){
         qt3 = contract_qt3_qt2_c(ksite,cop_ca[s]); 
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop_ca[s].msym;
	 qt2.index[0] = 2*k + cop_ca[s].index[0];
	 qt2.index[1] = 2*k + cop_ca[s].index[1];
	 qops.push_back(qt2);
      }
   }else if(iop == 1){
      // C: build, R: load
      qopers rqops_ca, rqops_c;
      string fname0r = oper_fname(scratch, p0, "right_B"); 
      oper_load(fname0r, rqops_ca);
      auto cop_ca = tns::oper_dot_ca(); 
      auto cop_c = tns::oper_dot_c(); 
      // pR^+qR 
      int k0 = bra.rsupport.at(p0)[0];
      for(const auto& rop_ca : rqops_ca){
         qt3 = contract_qt3_qt2_r(ksite,rop_ca);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = rop_ca.msym;
	 qt2.index[0] = rop_ca.index[0];
	 qt2.index[1] = rop_ca.index[1];
	 qops.push_back(qt2);
      }
      string fname0c = oper_fname(scratch, p0, "right_C"); 
      oper_load(fname0c, rqops_c);
      // pC^+qR
      for(int s=0; s<2; s++){
         for(const auto& rop_c : rqops_c){
	    qt3 = contract_qt3_qt2_r(ksite,rop_c.transpose());
	    auto ctmp = cop_c[s].col_signed();
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = cop_c[s].msym - rop_c.msym;
	    qt2.index[0] = 2*k + cop_c[s].index[0];
	    qt2.index[1] = rop_c.index[0];
	    qops.push_back(qt2);
	 }
      }
      // pR^+qC = -qC*pR^+
      for(const auto& rop_c : rqops_c){
         for(int s=0; s<2; s++){
	    qt3 = contract_qt3_qt2_r(ksite,rop_c);
	    auto ctmp = cop_c[s].transpose().col_signed(-1.0);
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = -cop_c[s].msym + rop_c.msym;
	    qt2.index[0] = rop_c.index[0];
	    qt2.index[1] = 2*k + cop_c[s].index[0];
	    qops.push_back(qt2);
	 }
      }
      // pC^+qC
      for(int s=0; s<4; s++){
         qt3 = contract_qt3_qt2_c(ksite,cop_ca[s]); 
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop_ca[s].msym;
	 qt2.index[0] = 2*k + cop_ca[s].index[0];
	 qt2.index[1] = 2*k + cop_ca[s].index[1];
	 qops.push_back(qt2);
      }

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
   }else if(iop == 2){
      // C: load, R: build

      qopers cqops_ca, cqops_c;
      string fname0b = oper_fname(scratch, p0, "right_B"); 
      oper_load(fname0b, rqops_ca);
      auto cop_ca = tns::oper_dot_ca(); 
      auto cop_c = tns::oper_dot_c(); 
      // pR^+qR 
      int k0 = bra.rsupport.at(p0)[0];
      for(const auto& rop_ca : rqops_ca){
         qt3 = contract_qt3_qt2_r(ksite,rop_ca);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = rop_ca.msym;
	 qt2.index[0] = rop_ca.index[0];
	 qt2.index[1] = rop_ca.index[1];
	 qops.push_back(qt2);
      }
      string fname0c = oper_fname(scratch, p0, "right_C"); 
      oper_load(fname0c, rqops_c);
      // pC^+qR
      for(int s=0; s<2; s++){
         for(const auto& rop_c : rqops_c){
	    qt3 = contract_qt3_qt2_r(ksite,rop_c.transpose());
	    auto ctmp = cop_c[s].col_signed();
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = cop_c[s].msym - rop_c.msym;
	    qt2.index[0] = 2*k + cop_c[s].index[0];
	    qt2.index[1] = rop_c.index[0];
	    qops.push_back(qt2);
	 }
      }
      // pR^+qC = -qC*pR^+
      for(const auto& rop_c : rqops_c){
         for(int s=0; s<2; s++){
	    qt3 = contract_qt3_qt2_r(ksite,rop_c);
	    auto ctmp = cop_c[s].transpose().col_signed(-1.0);
	    qt3 = contract_qt3_qt2_c(qt3,ctmp);
            qt2 = contract_qt3_qt3_cr(bsite,qt3);
	    qt2.msym = -cop_c[s].msym + rop_c.msym;
	    qt2.index[0] = rop_c.index[0];
	    qt2.index[1] = 2*k + cop_c[s].index[0];
	    qops.push_back(qt2);
	 }
      }
      // pC^+qC
      for(int s=0; s<4; s++){
         qt3 = contract_qt3_qt2_c(ksite,cop_ca[s]); 
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
	 qt2.msym = cop_ca[s].msym;
	 qt2.index[0] = 2*k + cop_ca[s].index[0];
	 qt2.index[1] = 2*k + cop_ca[s].index[1];
	 qops.push_back(qt2);
      }


      exit(1);
   }else if(iop == 3){
      // C: load, R: load
      exit(1);
   }else{ 
      cout << "error: no such option for iop=" << iop << endl;
      exit(1);      
   }
   oper_save(fname, qops);
}
*/

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
   //oper_renorm_rightB(bra,ket,p,p0,ifload,scratch);
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
