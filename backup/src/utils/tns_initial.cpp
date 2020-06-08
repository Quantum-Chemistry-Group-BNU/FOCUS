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
      vector<bool> dir = {1,1};
      qtensor2 qt2(sym,rsite0.qmid,rsite0.qcol,dir);
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

qtensor3 tns::get_rsite0(const vector<qtensor3>& psi){
   int neig = psi.size();
   qsym sym = psi[0].sym, qvac = qsym(0,0);
   qsym_space qstates = {{sym,neig}};
   qtensor3 rsite0(qvac,psi[0].qmid,qstates,psi[0].qcol); 
   // put data into rsite0[n](I,c) = psi[I](n,1,c);
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
	    auto& blk0 = rsite0.qblocks[key];
	    if(blk0.size() > 0){
	       for(int m=0; m<mdim; m++){
	          for(int c=0; c<cdim; c++){
	             for(int r=0; r<rdim; r++){
	                blk0[m](r,c) = psi[r].qblocks.at(make_tuple(qm,qvac,qc))[0](0,c);
	             }
	          }
	       }
	    }
	 }
      }
   }
   //debug
   //auto qt2 = contract_qt3_qt3_cr(rsite0,rsite0);
   //qt2.print("qt2",2);
   return rsite0;
}

void tns::initial_onedot(comb& icomb){
   auto& rsite0 = icomb.rsites.at(make_pair(0,0));
   auto& rsite1 = icomb.rsites.at(make_pair(1,0));
   auto cwf0 = get_cwf0(rsite0);
   for(int i=0; i<cwf0.size(); i++){
      auto psi = contract_qt3_qt2_l(rsite1,cwf0[i]);
      icomb.psi.push_back(psi);
   }
}

void tns::initial_twodot(comb& icomb, 
			 const directed_bond& dbond,
			 qtensor4& wf,
			 const int nsub,
			 const int neig,
		         vector<double>& v0){
   const bool debug = true;
   if(debug) cout << "tns::initial_twodot ";
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   if(forward){
      if(!cturn){
         if(debug) cout << "|lc1>" << endl;
	 for(int i=0; i<neig; i++){
	    auto cwf = icomb.psi[i].merge_lc();
            auto wf3 = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
	    auto wf4 = wf3.split_lc1(wf.qrow, wf.qmid, wf.dpt_lc1().second);
	    assert(wf4.get_dim() == nsub);
	    wf4.to_array(&v0[nsub*i]);
         }
      }else{
	 if(debug) cout << "|lr> (comb)" << endl;
	 for(int i=0; i<neig; i++){
	    auto cwf = icomb.psi[i].merge_lr(); // on backone
	    auto wf2 = cwf.dot(icomb.rsites[p1].merge_cr());
	    auto wf4 = wf2.split_lr_c1c2(wf.qrow, wf.qcol, wf.dpt_lr().second,
			    	         wf.qmid, wf.qver, wf.dpt_c1c2().second);
	    assert(wf4.get_dim() == nsub);
	    wf4.to_array(&v0[nsub*i]);
	 }
      }
   }else{
      if(!cturn){
	 if(debug) cout << "|c2r>" << endl;
	 for(int i=0; i<neig; i++){
	    auto cwf = icomb.psi[i].merge_cr();
	    auto wf3 = contract_qt3_qt2_r0(icomb.lsites[p0],cwf);
	    auto wf4 = wf3.split_c2r(wf.qver, wf.qcol, wf.dpt_c2r().second);
	    assert(wf4.get_dim() == nsub);
	    wf4.to_array(&v0[nsub*i]);
	 }
      }else{
	 if(debug) cout << "|c1c2> (comb)" << endl;
	 for(int i=0; i<neig; i++){
	    auto cwf = icomb.psi[i].merge_cr(); // on branch
	    auto wf2 = icomb.lsites[p0].merge_lr().dot(cwf);
	    auto wf4 = wf2.split_lr_c1c2(wf.qrow, wf.qcol, wf.dpt_lr().second,
			    	         wf.qmid, wf.qver, wf.dpt_c1c2().second);
	    assert(wf4.get_dim() == nsub);
	    wf4 = wf4.perm_signed(); // back to backbone
	    wf4.to_array(&v0[nsub*i]);
	 }
      }
   }
}
