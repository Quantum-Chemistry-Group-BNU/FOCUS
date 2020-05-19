#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

void tns::oper_renorm_opP(const string& superblock,	
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const vector<int>& lsupp,
			  const vector<int>& orbord,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opP" << endl;
   auto t0 = global::get_time();

   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   qtensor2 Paa(qsym(-2,-2), bsite.qrow, ksite.qrow);
   qtensor2 Pbb(qsym(-2, 0), bsite.qrow, ksite.qrow);
   qtensor2 Pos(qsym(-2,-1), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_q : lsupp){
	 int qa = 2*korb_q, qb = qa+1;
	 if(orbord[pa] < orbord[qa]) qops['P'][oper_pack(pa,qa)] = Paa;
	 if(orbord[pb] < orbord[qb]) qops['P'][oper_pack(pb,qb)] = Pbb;
	 if(orbord[pa] < orbord[qb]) qops['P'][oper_pack(pa,qb)] = Pos;
	 if(orbord[pb] < orbord[qa]) qops['P'][oper_pack(pb,qa)] = Pos;
      }
   }
   for(auto& qop : qops['P']){
      auto pq = qop.first;
      auto Hwf = oper_kernel_Pwf(superblock,ksite,qops1,qops2,int2e,int1e,pq);
      qop.second = oper_kernel_renorm(superblock,bsite,Hwf);
   }

   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::oper_renorm_opP : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_opQ(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const vector<int>& lsupp,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opQ" << endl;
   auto t0 = global::get_time();

   // initialization for Qps = <pq||sr> aq^+ar
   // Qaa,bb, Qab ~ b^+a, Qba ~ a^+b
   qtensor2 Qss(qsym(0, 0), bsite.qrow, ksite.qrow);
   qtensor2 Qab(qsym(0,-1), bsite.qrow, ksite.qrow);
   qtensor2 Qba(qsym(0, 1), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_s : lsupp){
	 int sa = 2*korb_s, sb = sa+1;
	 qops['Q'][oper_pack(pa,sa)] = Qss;
	 qops['Q'][oper_pack(pb,sb)] = Qss;
	 qops['Q'][oper_pack(pa,sb)] = Qab;
	 qops['Q'][oper_pack(pb,sa)] = Qba;
      }
   }
   for(auto& qop : qops['Q']){
      auto ps = qop.first;
      auto Hwf = oper_kernel_Qwf(superblock,ksite,qops1,qops2,int2e,int1e,ps);
      qop.second = oper_kernel_renorm(superblock,bsite,Hwf);
   }

   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::oper_renorm_opQ : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_opS(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const vector<int>& lsupp,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opS" << endl;
   auto t0 = global::get_time();

   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   qtensor2 Sa(qsym(-1,-1), bsite.qrow, ksite.qrow);
   qtensor2 Sb(qsym(-1, 0), bsite.qrow, ksite.qrow);
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      qops['S'][pa] = Sa;
      qops['S'][pb] = Sb;
   }
   for(auto& qop : qops['S']){
      int pL = qop.first;
      auto Hwf = oper_kernel_Swf(superblock,ksite,qops1,qops2,int2e,int1e,pL);
      qop.second = oper_kernel_renorm(superblock,bsite,Hwf);
   }

   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::oper_renorm_opS : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_opH(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opH" << endl;
   auto t0 = global::get_time();

   auto Hwf = oper_kernel_Hwf(superblock,ksite,qops1,qops2,int2e,int1e);
   qops['H'][0] = oper_kernel_renorm(superblock,bsite,Hwf);
   
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::oper_renorm_opH : " << setprecision(2) 
	   << global::get_duration(t1-t0) << " s" << endl;
   }
}
