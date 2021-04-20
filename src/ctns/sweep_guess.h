#ifndef SWEEP_GUESS_H
#define SWEEP_GUESS_H


namespace ctns{

// generate initial guess for initial sweep optimization 
template <typename Km>
void guess_onedot_psi0(comb<Km>& icomb, const int nstates){
   const auto& rsite0 = icomb.rsites.at(std::make_pair(0,0));
   const auto& rsite1 = icomb.rsites.at(std::make_pair(1,0));
   assert(icomb.rwfuns.qrow.size() == 1); // only same symmetry of wfs
   auto state_sym = icomb.rwfuns.qrow.get_sym(0);
   for(int istate=0; istate<nstates; istate++){
      auto qt2 = icomb.get_istate(istate);
      auto qt3 = contract_qt3_qt2_l(rsite0,qt2);
      // qt3[n0](1,r0) -> cwf(n0,r0)
      qtensor2<typename Km::dtype> cwf(state_sym, rsite0.qmid, rsite0.qcol, {1,1});
      for(int br=0; br<cwf.rows(); br++){
	 for(int bc=0; bc<cwf.cols(); bc++){
	    auto& blk = cwf(br,bc);
	    if(blk.size() == 0) continue;
	    const auto& blk0 = qt3(br,0,bc);
	    int rdim = cwf.qrow.get_dim(br); 
	    int cdim = cwf.qcol.get_dim(bc);
	    for(int r=0; r<rdim; r++){
	       for(int c=0; c<cdim; c++){
	     	  blk(r,c) = blk0[r](0,c); 
	       } // c
	    } // r
	 } // bc
      } // br
      // psi[n1](n0,r1) = cwf(n0,r0)*rsite1[n1](r0,r1)
      auto psi = contract_qt3_qt2_l(rsite1,cwf);
      icomb.psi.push_back(psi);
   } // istate
}

/*
void tns::guess_twodot(comb& icomb, 
			 const directed_bond& dbond,
			 qtensor4& wf,
			 const int nsub,
			 const int neig,
		         vector<double>& v0){
   const bool debug = true;
   if(debug) cout << "tns::guess_twodot ";
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
	    wf4 = wf4.permCR_signed(); // back to backbone
	    wf4.to_array(&v0[nsub*i]);
	 }
      }
   }
}
   
*/

} // ctns

#endif
