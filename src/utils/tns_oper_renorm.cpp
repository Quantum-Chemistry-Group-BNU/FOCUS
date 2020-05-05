#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_right(const comb& bra,
			    const comb& ket,
		            const comb_coord& p,
		            const comb_coord& p0,
		            const integral::two_body& int2e,
		            const integral::one_body& int1e,
			    const string scratch){
   bool debug = true;
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
   // Branch only have normal operators in right canonical form. 
   oper_renorm_rightC(bra,ket,p,p0,ifload,scratch);
   oper_renorm_rightA(bra,ket,p,p0,ifload,scratch);
   oper_renorm_rightB(bra,ket,p,p0,ifload,scratch);
   if(j == 0){
      oper_renorm_rightP(bra,ket,p,p0,ifload,int2e,int1e,scratch);
      oper_renorm_rightQ(bra,ket,p,p0,ifload,int2e,int1e,scratch);
   }
   oper_renorm_rightS(bra,ket,p,p0,ifload,int2e,int1e,scratch);
   oper_renorm_rightH(bra,ket,p,p0,ifload,int2e,int1e,scratch);
   if(debug) oper_rbases(bra,ket,p,int2e,int1e,scratch);
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   auto t0 = global::get_time();
   int nbackbone = bra.nbackbone;
   // loop over internal nodes
   for(int i=nbackbone-2; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
	 auto p0 = make_pair(i+1,0);
	 oper_renorm_right(bra,ket,p,p0,int2e,int1e,scratch);
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-2; j>=1; j--){
	    auto pj = make_pair(i,j);
	    auto p0 = make_pair(i,j+1);    
	    oper_renorm_right(bra,ket,pj,p0,int2e,int1e,scratch);
	 } // j
	 auto p0 = make_pair(i+1,0);
	 oper_renorm_right(bra,ket,p,p0,int2e,int1e,scratch);
      }else{
	 cout << "error: tp=" << tp << endl;
	 exit(1);
      }
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
