#include "tns_initial.h"

using namespace std;
using namespace tns;

vector<qtensor2> tns::get_cwf0(const qtensor3& rsite0){
   assert(rsite0.qrow.size() == 1); // only same symmetry of wfs
   auto it = rsite0.qrow.begin();
   qsym sym = it->first;
   int neig = it->second;
   vector<qtensor2> cwf0;
   for(int i=0; i<neig; i++){
      qtensor2 qt2(sym,rsite0.qmid,rsite0.qcol,{0,1,1});
      cwf0.push_back(qt2);
   }
   // put data into cwf0[I](n,c) = R0[n](I,c)
   for(const auto& pm : rsite0.qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& pr : rsite0.qrow){
	 auto qr = pr.first;
	 int rdim = pr.second;
	 for(const auto& pc : rsite0.qcol){
	    auto qc = pc.first;
	    int cdim = pc.second;
	    auto key = make_tuple(qm,qr,qc);
	    const auto& blk0 = rsite0.qblocks.at(key);
	    if(blk0.size() > 0){
	       for(int m=0; m<mdim; m++){
	          for(int c=0; c<cdim; c++){
	             for(int r=0; r<rdim; r++){
	                cwf0[r].qblocks[make_pair(qm,qc)](m,c) = blk0[m](r,c);
	             }
	          }
	       }
	    }
	 }
      }
   }
   return cwf0;
}

void tns::initial_onedot(const comb& icomb,
			 const int nsub,
			 const int neig,
			 vector<double>& v0){
   auto& rsite0 = icomb.rsites.at(make_pair(0,0));
   auto& rsite1 = icomb.rsites.at(make_pair(1,0));
   auto cwf0 = get_cwf0(rsite0);
   assert(cwf0.size() == neig);
   for(int i=0; i<neig; i++){
      auto psi0a = contract_qt3_qt2_l(rsite1,cwf0[i]);
      assert(psi0a.get_dim() == nsub);
      psi0a.to_array(&v0[nsub*i]);
   }
}
