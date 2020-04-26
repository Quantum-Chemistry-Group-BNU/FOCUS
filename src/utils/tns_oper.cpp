#include "tns_comb.h" 
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

// build different types of operators specified by
// - coord (i,j)
// - kind (ap,Apq,Bpq,Sp,Ppq,Qpq)
// 1. exact construction at type 0 site {|n>}
// 2. universal blocking code to deal with
//    - blocking at type 1,2 site (L/R) {|nr>}
//    - blocking at type 3 site (L/R) {|ur>}



// test subroutines for building operators: Sij & Hij
matrix tns::get_Sij(const comb& bra, 
  		    const comb& ket){
   auto start = make_pair(0,0);
   int m = bra.rsites.at(start).get_dim_row();
   int n = ket.rsites.at(start).get_dim_row();
   matrix Sij(m,n);
   // use tensor contraction to compute Sij
   if(bra.nbackbone != ket.nbackbone){
      cout << "error: shapes of two combs should be the same!" << endl;
      cout << "bra/ket nbackbone=" << bra.nbackbone << "," << ket.nbackbone << endl; 
      exit(1); 
   }
   int nbackbone = bra.nbackbone;
   qtensor2 qt2_r, qt2_u;
   // loop over sites on backbone
   for(int i=nbackbone-1; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
	 if(i==nbackbone-1){
	    //qt2_r = contract33CR(bra.rsites[p],ket.rsites[p]); 
	 }else{
	    //auto qtmp = contractR(ket.rsites[p],qt2_r)
	    //auto qt2_r = contract22CR(qtmp,bra.rsites[p])
	 }
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-1; j>=1; j--){
	    auto pj = make_pair(i,j);
            if(j==bra.topo[i].size()-1){
	       //qt2_u = contract33CR(bra.rsites[pj],ket.rsites[pj]);	   
	    }else{
	       //auto qtmp = contractR(ket.rsites[pj],qt2_u)
	       //auto qt2_u = contract22CR(qtmp,bra.rsites[pj])
	    } 
	 } // j
	 // internal site without physical index
	 // auto qtmp = contract(ket.rsites[p],qt2_u);
	 // qtmp = contract(qtmp,qt2_r);
	 // qt2_r = contract(bra.rsites[p],qtmp);
      }
   } // i
   // final: qt2_r to normal matrix
   Sij = qt2_r.to_matrix();
   return Sij;
}

matrix tns::get_Hij(const comb& bra, 
		    const comb& ket,
		    const integral::two_body& int2e,
		    const integral::one_body& int1e,
		    const double ecore){
   auto start = make_pair(0,0);
   int m = bra.rsites.at(start).get_dim_row();
   int n = ket.rsites.at(start).get_dim_row();
   matrix Hij(m,n);
   // use renormalization of operators to compute Hij
   
   return Hij;
}
