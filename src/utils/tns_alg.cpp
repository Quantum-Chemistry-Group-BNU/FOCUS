#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include "tns_alg.h"

using namespace std;
using namespace linalg;
using namespace tns;

// test subroutines for building operators: Smat & Hmat
matrix tns::get_Smat(const comb& bra, 
  		     const comb& ket){
   cout << "\ntns::get_Smat" << endl;
   auto start = make_pair(0,0);
   int m = bra.rsites.at(start).get_dim_row();
   int n = ket.rsites.at(start).get_dim_row();
   matrix Smat(m,n);
   // use tensor contraction to compute Smat
   if(bra.nbackbone != ket.nbackbone){
      cout << "error: bra/ket nbackbone=" << bra.nbackbone 
	   << "," << ket.nbackbone << endl; 
      exit(1); 
   }
   int nbackbone = bra.nbackbone;
   qtensor2 qt2_r, qt2_u;
   // loop over sites on backbone
   for(int i=nbackbone-1; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
	 if(i==nbackbone-1){
	    qt2_r = contract_qt3_qt3_lc(bra.rsites.at(p),ket.rsites.at(p));
	 }else{
	    auto qtmp = contract_qt3_qt2_l(ket.rsites.at(p),qt2_r);
	    qt2_r = contract_qt3_qt3_lc(bra.rsites.at(p),qtmp);
	 }
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-1; j>=1; j--){
	    auto pj = make_pair(i,j);
            if(j==bra.topo[i].size()-1){
	       qt2_u = contract_qt3_qt3_lc(bra.rsites.at(pj),ket.rsites.at(pj));	   
	    }else{
	       auto qtmp = contract_qt3_qt2_l(ket.rsites.at(pj),qt2_u);
	       qt2_u = contract_qt3_qt3_lc(bra.rsites.at(pj),qtmp);
	    }
	 } // j
	 // internal site without physical index
	 auto qtmp = contract_qt3_qt2_l(ket.rsites.at(p),qt2_r);
	 qtmp = contract_qt3_qt2_c(qtmp,qt2_u); // upper branch
	 qt2_r = contract_qt3_qt3_lc(bra.rsites.at(p),qtmp);
      }
   } // i
   // final: qt2_r to normal matrix
   Smat = qt2_r.to_matrix();
   return Smat;
}

matrix tns::get_Hmat(const comb& bra, 
		     const comb& ket,
		     const integral::two_body& int2e,
		     const integral::one_body& int1e,
		     const double ecore,
		     const string scratch){
   cout << "\ntns::get_Hmat" << endl;
   // environement
   oper_env_right(bra, ket, int2e, int1e, scratch);
   // load
   oper_dict qops;
   auto p = make_pair(0,0); 
   string fname = oper_fname(scratch, p, "rop");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   return Hmat;
}
