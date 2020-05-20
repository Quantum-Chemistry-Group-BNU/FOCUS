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
			  const vector<int>& supp,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opP" << endl;
   auto t0 = global::get_time();

   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   vector<int> index;
   for(int korb_p : supp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_q : supp){
	 int qa = 2*korb_q, qb = qa+1;
	 if(pa < qa) index.push_back(oper_pack(pa,qa));
	 if(pb < qb) index.push_back(oper_pack(pb,qb));
	 if(pa < qb) index.push_back(oper_pack(pa,qb));
	 if(pb < qa) index.push_back(oper_pack(pb,qa));
      }
   }
   for(const int pq : index){
      auto Hwf = oper_kernel_Pwf(superblock,ksite,qops1,qops2,int2e,int1e,pq);
      qops['P'][pq] = oper_kernel_renorm(superblock,bsite,Hwf);
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
			  const vector<int>& supp,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opQ" << endl;
   auto t0 = global::get_time();

   // initialization for Qps = <pq||sr> aq^+ar
   vector<int> index;
   for(int korb_p : supp){
      int pa = 2*korb_p, pb = pa+1;
      for(int korb_s : supp){
	 int sa = 2*korb_s, sb = sa+1;
	 index.push_back(oper_pack(pa,sa));
	 index.push_back(oper_pack(pb,sb));
	 index.push_back(oper_pack(pa,sb));
	 index.push_back(oper_pack(pb,sa));
      }
   }
   for(const int ps : index){
      auto Hwf = oper_kernel_Qwf(superblock,ksite,qops1,qops2,int2e,int1e,ps);
      qops['Q'][ps] = oper_kernel_renorm(superblock,bsite,Hwf);
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
			  const vector<int>& supp,
	                  const integral::two_body& int2e,
	                  const integral::one_body& int1e,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opS" << endl;
   auto t0 = global::get_time();

   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   vector<int> index;
   for(int korb_p : supp){
      int pa = 2*korb_p, pb = pa+1;
      index.push_back(pa);
      index.push_back(pb);
   }
   for(const int p : index){
      auto Hwf = oper_kernel_Swf(superblock,ksite,qops1,qops2,int2e,int1e,p);
      qops['S'][p] = oper_kernel_renorm(superblock,bsite,Hwf);
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
