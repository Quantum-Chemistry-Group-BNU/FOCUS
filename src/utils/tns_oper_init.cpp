#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// construct directly for boundary case {C,A,B,S,H}
oper_dict tns::oper_build_local(const int kp,
		                const integral::two_body& int2e,
		                const integral::one_body& int1e){
   vector<int> rsupp;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      rsupp.push_back(k);
   }
   oper_dict qops;
   oper_dot_C(kp, qops);
   oper_dot_A(kp, qops);
   oper_dot_B(kp, qops);
   oper_dot_S(kp, int2e, int1e, rsupp, qops);
   oper_dot_H(kp, int2e, int1e, qops);
   return qops;
}

// construct directly for boundary case {C,A,B,S,H}
void tns::oper_build_boundary(const comb& icomb,
			      const integral::two_body& int2e,
			      const integral::one_body& int1e,
			      const string scratch){
   // physical
   for(int idx=0; idx<icomb.rcoord.size(); idx++){
      auto p = icomb.rcoord[idx];
      if(icomb.type.at(p) != 3){
         int kp = icomb.get_kp(p);
         auto qops = oper_build_local(kp, int2e, int1e);
         string fname = oper_fname(scratch, p, "cop");
         oper_save(fname, qops);
      }
   } 
   // right boundary (exclude the start point)
   for(int idx=0; idx<icomb.rcoord.size(); idx++){
      auto p = icomb.rcoord[idx];
      if(icomb.type.at(p) == 0 && p.first != 0){
         int kp = icomb.get_kp(p);
         auto qops = oper_build_local(kp, int2e, int1e);
         string fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   } 
   // left boundary
   auto p = make_pair(0,0);
   int kp = icomb.get_kp(p);
   auto qops = oper_build_local(kp, int2e, int1e);
   string fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

oper_dict tns::oper_get_cqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict cqops;
   if(icomb.ifbuild_c(p)){
      string fname0c = oper_fname(scratch, p, "cop");
      oper_load(fname0c, cqops);
   }else{
      auto pc = icomb.get_c(p);
      string fname0c = oper_fname(scratch, pc, "rop");
      oper_load(fname0c, cqops);
   }
   return cqops;
}

oper_dict tns::oper_get_rqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict rqops;
   auto pr = icomb.get_r(p);
   string fname0r = oper_fname(scratch, pr, "rop");
   oper_load(fname0r, rqops);
   return rqops;
}

oper_dict tns::oper_get_lqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict lqops;
   auto pl = icomb.get_l(p);
   string fname0l = oper_fname(scratch, pl, "lop");
   oper_load(fname0l, lqops);
   return lqops;
}
