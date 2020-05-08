#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

void tns::oper_renorm_right(const comb& bra,
			    const comb& ket,
		            const comb_coord& p,
		            const comb_coord& p0,
		            const integral::two_body& int2e,
		            const integral::one_body& int1e,
			    const string scratch){
   bool debug = false;
   auto t0 = global::get_time();
   cout << "\ntns::oper_renorm_right switch=" 
	<< (p == make_pair(bra.iswitch,0)) << endl;
   int ip  =  p.first, jp  =  p.second, tp  = bra.type.at(p); 
   int ip0 = p0.first, jp0 = p0.second, tp0 = bra.type.at(p0);
   auto type = make_pair(tp,tp0);
   
   pair<bool,bool> ifbuild;
   if(type == make_pair(1,0) || type == make_pair(2,0)){
      ifbuild = make_pair(1,1); // build or load
   }else if(type == make_pair(1,1) || type == make_pair(1,3) ||
	    type == make_pair(2,2) ||
	    type == make_pair(0,1) || type == make_pair(0,3)){ // start site
      ifbuild = make_pair(1,0);
   }else if(type == make_pair(3,0)){
      ifbuild = make_pair(0,1);
   }else if(type == make_pair(3,1) || type == make_pair(3,3)){
      ifbuild = make_pair(0,0);
   }else{
      cout << "error: no such case! (tp,tp0)=" << tp << "," << tp0 << endl;
      exit(1);
   }

   cout << "p=(" << ip << "," << jp << ")[" << bra.topo[ip][jp] << "]"
	<< " p0=(" << ip0 << "," << jp0 << ")[" << bra.topo[ip0][jp0] << "]" 
	<< " type=(" << tp << "," << tp0 << ")"
	<< " ifbuild(C,R)=(" << ifbuild.first << "," << ifbuild.second <<  ")"
	<< endl;
   
   // three kinds of sites 
   bool left = (jp == 0 && ip < bra.iswitch);
   bool swpt = (jp == 0 && ip == bra.iswitch);
   bool rest = !(left || swpt);
   oper_renorm_rightC(bra,ket,p,p0,ifbuild,scratch,debug);
   if(rest){
      oper_renorm_rightA(bra,ket,p,p0,ifbuild,scratch,debug);
      oper_renorm_rightB(bra,ket,p,p0,ifbuild,scratch,debug);
   }
   if(left || swpt){
      auto ifAB = swpt;	   
      oper_renorm_rightP(bra,ket,p,p0,ifbuild,ifAB,int2e,int1e,scratch,debug);
      oper_renorm_rightQ(bra,ket,p,p0,ifbuild,ifAB,int2e,int1e,scratch,debug);
   }
   auto ifAB = swpt || rest;
   oper_renorm_rightS(bra,ket,p,p0,ifbuild,ifAB,int2e,int1e,scratch,debug);
   oper_renorm_rightH(bra,ket,p,p0,ifbuild,ifAB,int2e,int1e,scratch,debug);
   
   auto t1 = global::get_time();
   cout << "timing for tns::oper_renorm_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   auto t0 = global::get_time();
   cout << "\ntns::oper_env_right" << endl;
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
