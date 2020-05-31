#include "../core/linalg.h"
#include "../core/tools.h"
#include "tns_decimation.h"
#include "tns_prdm.h"

using namespace std;
using namespace tns;
using namespace linalg;

// wf[L,R] = U[L,l]*sl*Vh[l,R]
qtensor2 tns::decimation_row(const qtensor2& rdm,
			     const int dcut,
			     double& dwt,
			     int& deff,
			     const bool permute){
   const double thresh_sig2 = 1.e-16;
   bool debug = false;
   if(debug) cout << "tns::decimation_row dcut=" << dcut << endl;
   const auto& qrow = rdm.qrow;
   // 0. normalize
   double rfac = 0.0;
   for(const auto& pr : qrow){
      auto qr = pr.first;
      auto key = make_pair(qr,qr);
      rfac += rdm.qblocks.at(key).trace();
   }
   rfac = 1.0/rfac;
   // 1. compute reduced basis
   map<int,qsym> idx2qsym; 
   vector<double> sig2all;
   map<qsym,matrix> rbasis;
   int idx = 0, ioff = 0;
   for(const auto& pr : qrow){
      auto qr = pr.first;
      auto key = make_pair(qr,qr);
      // compute renormalized basis
      int rdim = pr.second;
      vector<double> sig2(rdim);
      matrix rbas(rdm.qblocks.at(key));
      rbas *= rfac; // normalize
      eigen_solver(rbas, sig2, 1);
      // save
      for(int i=0; i<rdim; i++){
         idx2qsym[ioff+i] = qr;
      }
      copy(sig2.begin(), sig2.end(), back_inserter(sig2all));
      rbasis[qr] = rbas;
      ioff += rdim;
      if(debug){
	 cout << "idx=" << idx << " qr=" << qr << " rdim=" << rdim << endl;
	 cout << "    sig2=";
	 for(auto s : sig2) cout << s << " ";
	 cout << endl;
	 idx++;
      }
   }
   assert(rdm.get_dim_row() == sig2all.size());
   // 2. select important ones
   auto index = tools::sort_index(sig2all, 1);
   qsym_space qres;
   map<qsym,double> wts;
   double sum = 0.0;
   double SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(i >= dcut) break; // discard rest
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
      SvN += -sig2all[idx]*log2(sig2all[idx]);
      if(debug){
	if(i==0) cout << "sorted sig2:" << endl;     
	cout << "i=" << i << " q=" << q << " idx=" << qres[q]-1 
             << " sig2=" << sig2all[idx] 
	     << " accum=" << sum << endl;
      }
   }
   deff = qsym_space_dim(qres);
   dwt = 1.0-sum;
   cout << "  reduction:" << sig2all.size() << "->" << deff
        << "  dwt=" << dwt 
	<< "  SvN=" << SvN << endl; 
   idx = 0;
   for(const auto& p : qres){
      auto& q = p.first;
      cout << "  idx=" << idx << "  qsym=" << q << "  dim=" << p.second 
	   << "  wt=" << scientific << setprecision(2) << wts[q] 
	   << "  per=" << defaultfloat << wts[q]*100 << endl;
      idx++;
   }
   // 3. form qt2 assembling blocks
   idx = 0;
   qtensor2 qt2(qsym(0,0), qrow, qres);
   for(auto& p : qt2.qblocks){
      auto q = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         assert(q.first == q.second); // must be block diagonal
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
   // 4. permute two lines for later generating right canonical form
   if(permute) qt2 = qt2.P();
   return qt2;
}
   
void tns::decimation_onedot(comb& icomb, 
		            const directed_bond& dbond,
		            const int dcut,
			    const matrix& vsol,
			    qtensor3& wf,
		            double& dwt,
			    int& deff,
			    const double noise, 
			    oper_dict& cqops,
			    oper_dict& lqops,
			    oper_dict& rqops){
   const double thresh_nz = 1.e-10;
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   cout << "tns::decimation_onedot (fw,ct,dcut,noise)=(" 
	<< forward << "," << cturn << "," << dcut << "," << scientific << noise << ") ";
   qtensor2 rdm;
   if(forward){
      // update lsites & ql
      if(!cturn){
         cout << "renormalize |lc>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.merge_lc().get_rdm_row();
	    }else{
	       rdm += wf.merge_lc().get_rdm_row();
	    }
	    if(noise > thresh_nz) get_prdm_lc(wf, lqops, cqops, noise, rdm);
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
	 // update site tensor
	 icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	
	 // initial guess for next site within the bond
	 icomb.psi0a.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = qt2.T().dot(wf.merge_lc()); // <-W[alpha,r]->
	    auto psi0 = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
	    icomb.psi0a.push_back(psi0);
	 }
      }else{
	 // special for comb
         cout << "renormalize |lr> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.perm_signed().merge_lr().get_rdm_row();
	    }else{
	       rdm += wf.perm_signed().merge_lr().get_rdm_row();
	    }
	    if(noise > thresh_nz) get_prdm_lr(wf, lqops, rqops, noise, rdm);
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
	 // update site tensor
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
	 // initial guess for next site within the bond
	 icomb.psi0a.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = qt2.T().dot(wf.perm_signed().merge_lr()); // <-W[alpha,r]->
	    auto psi0 = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
	    icomb.psi0a.push_back(psi0);
	 }
      }
   }else{
      // update rsites & qr
      cout << "renormalize |cr>" << endl;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         if(i == 0){
            rdm  = wf.merge_cr().get_rdm_col();
         }else{
            rdm += wf.merge_cr().get_rdm_col();
         }
         if(noise > thresh_nz) get_prdm_cr(wf, cqops, rqops, noise, rdm);
      }
      auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
      // update site tensor
      icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qcol, wf.dpt_cr().second);
      //-------------------------------------------------------------------	
      assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
      auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
      assert(ovlp.check_identity(1.e-10,false)<1.e-10);
      //-------------------------------------------------------------------	 
      // initial guess for next site within the bond
      icomb.psi0a.clear();
      for(int i=0; i<vsol.cols(); i++){
	 wf.from_array(vsol.col(i));
	 auto cwf = wf.merge_cr().dot(qt2.T()); // <-W[l,alpha]->
	 if(!cturn){
	    auto psi0 = contract_qt3_qt2_r0(icomb.lsites[p0],cwf);
            icomb.psi0a.push_back(psi0);
	 }else{
	    // special treatment of the propagation downside to backbone
	    auto psi0 = contract_qt3_qt2_c(icomb.lsites[p0],cwf.P());
	    psi0 = psi0.perm_signed(); // |(lr)c> back to |lcr> order on backbone
	    icomb.psi0a.push_back(psi0);
	 }
      }
   }
}

void tns::decimation_twodot(comb& icomb, 
		            const directed_bond& dbond,
		            const int dcut, 
			    const matrix& vsol,
			    qtensor4& wf,
		            double& dwt,
			    int& deff){
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   cout << "tns::decimation_twodot (fw,ct,dcut)=(" 
	<< forward << "," << cturn << "," << dcut << ") "; 
   qtensor2 rdm;
   if(forward){
      // update lsites & ql
      if(!cturn){
         cout << "renormalize |lc1>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
            auto wf3 = wf.merge_c2r();
	    if(i == 0){
	       rdm  = wf3.merge_lc().get_rdm_row();
	    }else{
	       rdm += wf3.merge_lc().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
         icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }else{
         cout << "renormalize |lr> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.perm_signed().merge_lr_c1c2().get_rdm_row();
	    }else{
	       rdm += wf.perm_signed().merge_lr_c1c2().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
      }
   }else{
      // update rsites & qr
      if(!cturn){
         cout << "renormalize |c2r>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_lc1();
            if(i == 0){
               rdm  = wf3.merge_cr().get_rdm_col();
            }else{
               rdm += wf3.merge_cr().get_rdm_col();
            }
         }
         auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
	 icomb.rsites[p] = qt2.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identity(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	 
      }else{
         cout << "renormalize |c1r2> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.perm_signed().merge_lr_c1c2().get_rdm_col();
	    }else{
	       rdm += wf.perm_signed().merge_lr_c1c2().get_rdm_col();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
	 icomb.rsites[p]= qt2.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identity(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	 
      }
   }
}
