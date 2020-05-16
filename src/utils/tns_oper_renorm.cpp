#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// renormalize rops
void tns::oper_renorm_rops(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
		           const integral::two_body& int2e,
		           const integral::one_body& int1e,
			   const string scratch){
   bool debug = false;
   auto t0 = global::get_time();
   int ip  =  p.first, jp  =  p.second;
   cout << "\ntns::oper_renorm_rops iswitch=" 
	<< (p == make_pair(bra.iswitch,0)) 
        << " coord=(" << ip << "," << jp << ")"
	<< "[" << bra.topo[ip][jp] << "]" 
	<< " type=" << bra.type.at(p) << endl;
   oper_dict qops;
   auto cqops = oper_get_cqops(bra, p, scratch);
   auto rqops = oper_get_rqops(bra, p, scratch);
   // three kinds of sites 
   bool left = (jp == 0 && ip < bra.iswitch);
   bool swpt = (jp == 0 && ip == bra.iswitch);
   bool rest = !(left || swpt);
   oper_renorm_ropC(bra,ket,p,cqops,rqops,qops,debug);
   if(debug) oper_rbases(bra,ket,p,qops,'C');
   if(rest){
      oper_renorm_ropA(bra,ket,p,cqops,rqops,qops,debug);
      if(debug) oper_rbases(bra,ket,p,qops,'A');
      oper_renorm_ropB(bra,ket,p,cqops,rqops,qops,debug);
      if(debug) oper_rbases(bra,ket,p,qops,'B');
   }
   if(left || swpt){
      auto ifAB = swpt;
      oper_renorm_ropP(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
      if(debug) oper_rbases(bra,ket,p,int2e,int1e,qops,'P');
      oper_renorm_ropQ(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
      if(debug) oper_rbases(bra,ket,p,int2e,int1e,qops,'Q');
   }
   auto ifAB = swpt || rest;
   oper_renorm_ropS(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
   if(debug) oper_rbases(bra,ket,p,int2e,int1e,qops,'S');
   oper_renorm_ropH(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
   if(debug) oper_rbases(bra,ket,p,int2e,int1e,qops,'H');
   // save
   string fname = oper_fname(scratch, p, "rop");
   oper_save(fname, qops);
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
   oper_build_boundary(bra, int2e, int1e, scratch);
   for(int idx=0; idx<bra.rcoord.size(); idx++){
      auto p = bra.rcoord[idx];
      if(bra.type.at(p) != 0 || p.first == 0){
         oper_renorm_rops(bra, ket, p, int2e, int1e, scratch);
      }
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

// renormalize lops
void tns::oper_renorm_lops(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
		           const integral::two_body& int2e,
		           const integral::one_body& int1e,
			   const string scratch){
   bool debug = false;
   auto t0 = global::get_time();
   int ip  =  p.first, jp  =  p.second;
   cout << "\ntns::oper_renorm_lops iswitch=" 
	<< (p == make_pair(bra.iswitch,0)) 
        << " coord=(" << ip << "," << jp << ")"
	<< "[" << bra.topo[ip][jp] << "]" 
	<< " type=" << bra.type.at(p) << endl;
   oper_dict qops;
   auto cqops = oper_get_cqops(bra, p, scratch);
   auto lqops = oper_get_lqops(bra, p, scratch);
   // three kinds of sites 
   bool left = (jp == 0 && ip < bra.iswitch);
   bool swpt = (jp == 0 && ip == bra.iswitch);
   bool rest = !(left || swpt);
/*   
   oper_renorm_ropC(bra,ket,p,cqops,rqops,qops,debug);
   if(rest){
      oper_renorm_ropA(bra,ket,p,cqops,rqops,qops,debug);
      oper_renorm_ropB(bra,ket,p,cqops,rqops,qops,debug);
   }
   if(left || swpt){
      auto ifAB = swpt;
      oper_renorm_ropP(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
      oper_renorm_ropQ(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
   }
   auto ifAB = swpt || rest;
   oper_renorm_ropS(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
   oper_renorm_ropH(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
*/
   // save
   string fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
   auto t1 = global::get_time();
   cout << "timing for tns::oper_renorm_lops : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;

}
