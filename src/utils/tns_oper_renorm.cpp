#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

void tns::oper_renorm_rops(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
		           const integral::two_body& int2e,
		           const integral::one_body& int1e,
			   const string scratch){
   bool debug = true;
   auto t0 = global::get_time();
   int ip  =  p.first, jp  =  p.second;
   cout << "\ntns::oper_renorm_rops iswitch=" 
	<< (p == make_pair(bra.iswitch,0)) 
        << " coord=(" << ip << "," << jp << ")"
	<< "[" << bra.topo[ip][jp] << "]" << endl;
   
   // three kinds of sites 
   bool left = (jp == 0 && ip < bra.iswitch);
   bool swpt = (jp == 0 && ip == bra.iswitch);
   bool rest = !(left || swpt);
   oper_renorm_ropC(bra,ket,p,scratch,debug);
   if(rest){
      oper_renorm_ropA(bra,ket,p,scratch,debug);
      oper_renorm_ropB(bra,ket,p,scratch,debug);
   }
   if(left || swpt){
      auto ifAB = swpt;	   
      oper_renorm_ropP(bra,ket,p,ifAB,int2e,int1e,scratch,debug);
      oper_renorm_ropQ(bra,ket,p,ifAB,int2e,int1e,scratch,debug);
   }
   auto ifAB = swpt || rest;
   oper_renorm_ropS(bra,ket,p,ifAB,int2e,int1e,scratch,debug);
   oper_renorm_ropH(bra,ket,p,ifAB,int2e,int1e,scratch,debug);
   
   auto t1 = global::get_time();
   cout << "timing for tns::oper_renorm_rops : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   auto t0 = global::get_time();
   cout << "\ntns::oper_env_right" << endl;
   for(int idx=0; idx<bra.rcoord.size(); idx++){
      auto p = bra.rcoord[idx];
      if(bra.type.at(p) == 0) continue;      
      oper_renorm_rops(bra,ket,p,int2e,int1e,scratch);
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
