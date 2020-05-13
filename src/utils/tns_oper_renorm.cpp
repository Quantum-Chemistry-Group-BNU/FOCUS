#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

#include "../core/matrix.h"
#include "../core/linalg.h"

using namespace linalg;
using namespace std;
using namespace tns;

// construct directly for boundary case {C,A,B,S,H}
void tns::oper_build_cops(const int kp,
		          const integral::two_body& int2e,
		          const integral::one_body& int1e,
		          oper_dict& qops){
   vector<int> rsupp;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      rsupp.push_back(k);
   }
   oper_dot_C(kp, qops);
   oper_dot_A(kp, qops);
   oper_dot_B(kp, qops);
   oper_dot_S(kp, int2e, int1e, rsupp, qops);
   oper_dot_H(kp, int2e, int1e, qops);
}

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
   // construct directly for boundary case {C,A,B,S,H}
   if(bra.type.at(p) == 0 && ip != 0){
      int kp = bra.get_kp(p);
      oper_build_cops(kp, int2e, int1e, qops);
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   // construct by renormalization
   }else{
      oper_dict cqops;
      if(bra.ifbuild_c(p)){
         int kp = bra.get_kp(p);
         oper_build_cops(kp, int2e, int1e, cqops);
      }else{
         auto pc = bra.get_c(p);
         string fname0c = oper_fname(scratch, pc, "rop");
         oper_load(fname0c, cqops);
      }
      oper_dict rqops;
      auto pr = bra.get_r(p);
      string fname0r = oper_fname(scratch, pr, "rop");
      oper_load(fname0r, rqops);
      // three kinds of sites 
      bool left = (jp == 0 && ip < bra.iswitch);
      bool swpt = (jp == 0 && ip == bra.iswitch);
      bool rest = !(left || swpt);
      oper_renorm_ropC(bra,ket,p,cqops,rqops,qops,debug);
      if(debug) oper_rbases(bra,ket,p,scratch,"C");
      if(rest){
         oper_renorm_ropA(bra,ket,p,cqops,rqops,qops,debug);
         if(debug) oper_rbases(bra,ket,p,scratch,"A");
         oper_renorm_ropB(bra,ket,p,cqops,rqops,qops,debug);
         if(debug) oper_rbases(bra,ket,p,scratch,"B");
      }
      if(left || swpt){
         auto ifAB = swpt;
         oper_renorm_ropP(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
         if(debug) oper_rbases(bra,ket,p,int2e,int1e,scratch,"P");
         oper_renorm_ropQ(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
         if(debug) oper_rbases(bra,ket,p,int2e,int1e,scratch,"Q");
      }
      auto ifAB = swpt || rest;
      oper_renorm_ropS(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
      if(debug) oper_rbases(bra,ket,p,int2e,int1e,scratch,"S");
      oper_renorm_ropH(bra,ket,p,cqops,rqops,qops,ifAB,int2e,int1e,debug);
      if(debug) oper_rbases(bra,ket,p,int2e,int1e,scratch,"H");
      // save
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
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
      oper_renorm_rops(bra,ket,p,int2e,int1e,scratch);
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
