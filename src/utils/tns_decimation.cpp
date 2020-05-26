#include "../core/linalg.h"
#include "../core/tools.h"
#include "tns_decimation.h"

using namespace std;
using namespace tns;
using namespace linalg;

// wf[L,R] = U[L,l]*sl*Vh[l,R]
qtensor2 tns::decimation_row(const vector<qtensor2>& wfs,
			     const int Dcut,
			     double& dwt,
			     const bool trans){
   const double thresh_sig2 = 1.e-16;
   bool debug = false;
   if(debug) cout << "tns::decimation_row Dcut=" << Dcut << endl;
   const auto& wf = wfs[0];
   const auto& qrow = wf.qrow;
   const auto& qcol = wf.qcol;
   // 1. compute reduced basis
   map<int,qsym> idx2qsym; 
   vector<double> sig2all;
   map<qsym,matrix> rbasis;
   int idx = 0, ioff = 0;
   for(const auto& pr : qrow){
      auto qr = pr.first;
      int rdim = pr.second;
      matrix rdm(rdim,rdim);
      // construct RDM = (psi*psi^dagger)[l',l] by averaging over states
      for(int i=0; i<wfs.size(); i++){
         for(const auto& pc : qcol){
            auto qc = pc.first;
            const auto& blk = wfs[i].qblocks.at(make_pair(qr,qc)); 
            if(blk.size() == 0) continue;
            rdm += dgemm("N","N",blk,blk.T());
         }
      }
      rdm *= 1.0/wfs.size(); 
      // compute renormalized basis
      vector<double> sig2(rdim);
      eigen_solver(rdm, sig2, 1);
      // save
      for(int i=0; i<rdim; i++){
         idx2qsym[ioff+i] = qr;
      }
      copy(sig2.begin(), sig2.end(), back_inserter(sig2all));
      rbasis[qr] = rdm;
      ioff += rdim;
      if(debug){
	 cout << "idx=" << idx << " qr=" << qr << " rdim=" << rdim << endl;
	 cout << "    sig2=";
	 for(auto s : sig2) cout << s << " ";
	 cout << endl;
	 idx++;
      }
   }
   assert(wf.get_dim_row() == sig2all.size());
   // 2. select important ones
   auto index = tools::sort_index(sig2all, 1);
   qsym_space qres;
   map<qsym,double> wts;
   double sum = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(i >= Dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      auto q = idx2qsym[idx];
      auto it = qres.find(q);
      if(it == qres.end()){
         qres[q] = 1;
	 wts[q] = sig2all[idx];
      }else{
	 qres[q] += 1;
	 wts[q] += sig2all[idx];
      }
      sum += sig2all[idx];
      if(debug){
	if(i==0) cout << "sorted sig2:" << endl;     
	cout << "i=" << i << " q=" << q << " idx=" << qres[q]-1 
             << " sig2=" << sig2all[idx] 
	     << " accum=" << sum << endl;
      }
   }
   dwt = 1.0-sum;
   cout << " reduction:" << sig2all.size() << "->" << qsym_space_dim(qres) 
        << " dwt=" << dwt << endl; 
   sum = 0.0;
   idx = 0;
   for(const auto& p : qres){
      auto& q = p.first;
      sum += wts[q];
      cout << " idx=" << idx << " qsym=" << q << " dim=" << p.second 
	   << " wt=" << wts[q] << " acc=" << sum << endl;
      idx++;
   }
   // 3. form qt2 assembling blocks
   idx = 0;
   qtensor2 qt2(qsym(0,0), qrow, qres);
   for(auto& p : qt2.qblocks){
      auto q = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         assert(q.first == q.second);
	 auto qd = q.first;
	 auto& rbas = rbasis[qd];
	 copy(rbas.data(), rbas.data()+blk.size(), blk.data());
	 if(debug){
	    if(idx == 0) cout << "reduced basis:" << endl;
	    cout << "idx=" << idx << " qr=" << qd << " shape=(" 
		 << blk.rows() << "," << blk.cols() << ")"
		 << endl;
	    idx++; 
	 }
      }
   }
   // 4. flip direction for later generating right canonical form
   if(trans){
      qt2 = qt2.T();	   
      qt2.dir = {0,0,1};
   }
   return qt2;
}

void tns::decimation_onedot(comb& icomb, 
		            const comb_coord& p, 
		            const bool forward, 
		            const bool cturn, 
		            const int dcut, 
		            const vector<qtensor3>& wfs,
		            double& dwt){
   cout << "tns::decimation_onedot (fw,ct,dcut)=(" 
	<< forward << "," << cturn << "," << dcut << ") "; 
   const auto& wf = wfs[0];
   vector<qtensor2> qt2s;
   if(forward){
      if(!cturn){
         // update lsites & ql
         cout << "renormalize |lc>" << endl;
	 for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_lc());
	 }
	 auto qt2 = decimation_row(qt2s, dcut, dwt);
	 icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }else{
	 // update lsites & qc [special for comb]
         cout << "renormalize |lr>" << endl;
	 for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_lr());
	 }
	 auto qt2 = decimation_row(qt2s, dcut, dwt);
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }
   }else{
      // update rsites (p1) & qr
      cout << "renormalize |cr>" << endl;
      for(int i=0; i<wfs.size(); i++){
	 qt2s.push_back(wfs[i].merge_cr().T()); // transpose here
      }
      auto qt2 = decimation_row(qt2s, dcut, dwt, true);
      icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qcol, wf.dpt_cr().second);
      //-------------------------------------------------------------------	
      assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
      auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
      assert(ovlp.check_identity(1.e-10,false)<1.e-10);
      //-------------------------------------------------------------------	 
   }
}

void tns::decimation_twodot(comb& icomb, 
		            const comb_coord& p, 
		            const bool forward, 
		            const bool cturn, 
		            const int dcut, 
		            const vector<qtensor4>& wfs,
		            double& dwt){
   cout << "tns::decimation_twodot (fw,ct,dcut)=(" 
	<< forward << "," << cturn << "," << dcut << ") "; 
   const auto& wf = wfs[0];
   vector<qtensor2> qt2s;
   if(forward){
      if(!cturn){
         // update lsites & ql
         cout << "renormalize |lc1>" << endl;
	 for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_lc1().merge_cr());
	 }
	 auto qt2 = decimation_row(qt2s, dcut, dwt);
         icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }else{
	 // update lsites & qc [special for comb]
         cout << "renormalize |lr>" << endl;
	 for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_lr_c1c2());
	 }
	 auto qt2 = decimation_row(qt2s, dcut, dwt);
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }
   }else{
      if(!cturn){
         // update rsites (p1) & qr
         cout << "renormalize |c2r>" << endl;
         for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_c2r().merge_lc().T());
	 }
         auto qt2 = decimation_row(qt2s, dcut, dwt, true);
         icomb.rsites[p] = qt2.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identity(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	 
      }else{
	 // update rsites & qr [special for comb]
         cout << "renormalize |c1r2>" << endl;
	 for(int i=0; i<wfs.size(); i++){
	    qt2s.push_back(wfs[i].merge_lr_c1c2().T());
	 }
	 auto qt2 = decimation_row(qt2s, dcut, dwt, true);
	 icomb.rsites[p]= qt2.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identity(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	 
      }
   }
}
