#ifndef SWEEP_TWODOT_GUESS_H
#define SWEEP_TWODOT_GUESS_H

namespace ctns{

/*
template <typename Km>
void twodot_guess(comb<Km>& icomb, 
		  const directed_bond& dbond,
		  const int nsub,
		  const int neig,
		  std::vector<qtensor4<typename Km::dtype>>& wfs){
   const bool debug = true;
   if(debug) cout << "tns::guess_twodot ";
   auto p0 = dbond.p0;
   auto p1 = dbond.p1;
   auto forward = dbond.forward;
   bool cturn = dbond.cturn;
   wfs.clear();
   if(forward){
      if(!cturn){
         if(debug) cout << "|lc1>" << endl;
	 for(int i=0; i<neig; i++){
	    // psi[l,c1,a]->cwf[lc1,a]
	    auto cwf = icomb.psi[i].merge_lc(); 
	    // cwf[lc1,a]*r[a,c2,r]->wf3[lc1,c2,r]
            auto wf3 = contract_qt3_qt2_l(icomb.rsites[p1],cwf); 
	    // wf3[lc1,c2,r]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_lc1(wf.qrow, wf.qmid, wf.dpt_lc1().second);
	    assert(wf4.get_dim() == nsub);
	    wfs.append(wf4);
         }
      }else{
	 if(debug) cout << "|lr>(comb)" << endl;
	 for(int i=0; i<neig; i++){
            // psi[l,a,r]->cwf[lr,a]		 
	    auto cwf = icomb.psi[i].merge_lr(); // on backone
	    // cwf[lr,a]*r[a,c1,c2]->wf[lr,c1,c2]
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
	    // psi[a,c2,r]->cwf[a,c2r] 
	    auto cwf = icomb.psi[i].merge_cr();
	    // l[l,c1,a]*cwf[a,c2r]->wf3[l,c1,c2r] 
	    auto wf3 = contract_qt3_qt2_r(icomb.lsites[p0],cwf.T());
	    // wf3[l,c1,c2r]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_c2r(wf.qver, wf.qcol, wf.dpt_c2r().second);
	    assert(wf4.get_dim() == nsub);
	    wfs.append(wf4);
	 }
      }else{
	 if(debug) cout << "|c1c2>(comb)" << endl;
	 //     c2
	 //      |
	 // c1---p0 [psi]
	 //      |
	 //  l---p1---r
	 //
	 for(int i=0; i<neig; i++){
            // psi[a,c1,c2]->cwf[a,c1c2]		 
	    auto cwf = icomb.psi[i].merge_cr(); // on branch
	    // l[l,a,r]->l[lr,a], l[lr,a]*cwf[a,c1c2]->wf2[lr,c1c2]
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
