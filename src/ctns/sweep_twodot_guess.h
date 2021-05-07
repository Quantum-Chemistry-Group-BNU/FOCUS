#ifndef SWEEP_TWODOT_GUESS_H
#define SWEEP_TWODOT_GUESS_H


namespace ctns{

/*
void twodot_guess(comb& icomb, 
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
