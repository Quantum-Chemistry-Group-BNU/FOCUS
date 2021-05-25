#ifndef SWEEP_GUESS_H
#define SWEEP_GUESS_H

namespace ctns{

// generate initial guess for initial sweep optimization 
template <typename Km>
void onedot_guess_psi0(comb<Km>& icomb, const int nstates){
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

const bool debug_twodot_guess = true;
extern const bool debug_twodot_guess;

template <typename Km>
std::vector<qtensor4<typename Km::dtype>> 
twodot_guess(comb<Km>& icomb, 
	     const directed_bond& dbond,
	     const int nsub,
	     const int neig,
	     qtensor4<typename Km::dtype>& wf){
   if(debug_twodot_guess) std::cout << "ctns::twodot_guess ";
   auto p0 = dbond.p0;
   auto p1 = dbond.p1;
   assert(icomb.psi.size() == neig);
   std::vector<qtensor4<typename Km::dtype>> psi4;
   if(dbond.forward){
      if(!dbond.cturn){
         if(debug_twodot_guess) std::cout << "|lc1>" << std::endl;
	 for(int i=0; i<neig; i++){
	    // psi[l,c1,a]->cwf[lc1,a]
	    auto cwf = icomb.psi[i].merge_lc(); 
	    // cwf[lc1,a]*r[a,c2,r]->wf3[lc1,c2,r]
            auto wf3 = contract_qt3_qt2_l(icomb.rsites[p1],cwf); 
	    // wf3[lc1,c2,r]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_lc1(wf.qrow, wf.qmid, wf.dpt_lc1().second);
	    assert(wf4.get_dim() == nsub);
	    psi4.push_back(wf4);
         } // i
      }else{
	 if(debug_twodot_guess) std::cout << "|lr>(comb)" << std::endl;
	 //
	 //     c2
	 //      |
	 // c1---p1 
	 //      |
	 //  l---p0---r
	 //     [psi]
	 //
	 for(int i=0; i<neig; i++){
            // psi[l,a,r]->cwf[lr,a]		 
	    auto cwf = icomb.psi[i].merge_lr(); // on backone
	    // cwf[lr,a]*r[a,c1,c2]->wf3[lr,c1,c2]
            auto wf3 = contract_qt3_qt2_l(icomb.rsites[p1],cwf); 
	    // wf3[lr,c1,c2]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	    assert(wf4.get_dim() == nsub);
	    psi4.push_back(wf4);
	 } // i
      } // cturn
   }else{
      if(!dbond.cturn){
	 if(debug_twodot_guess) std::cout << "|c2r>" << std::endl;
	 for(int i=0; i<neig; i++){
	    // psi[a,c2,r]->cwf[a,c2r] 
	    auto cwf = icomb.psi[i].merge_cr();
	    // l[l,c1,a]*cwf[a,c2r]->wf3[l,c1,c2r] 
	    auto wf3 = contract_qt3_qt2_r(icomb.lsites[p0],cwf.T());
	    // wf3[l,c1,c2r]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_c2r(wf.qver, wf.qcol, wf.dpt_c2r().second);
	    assert(wf4.get_dim() == nsub);
	    psi4.push_back(wf4);
	 } // i
      }else{
	 if(debug_twodot_guess) std::cout << "|c1c2>(comb)" << std::endl;
	 //
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
	    // wf2[lr,c1c2]->wf3[l,c1c2,r]
	    auto wf3 = wf2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	    // wf3[l,c1c2,r]->wf4[l,c1,c2,r]
	    auto wf4 = wf3.split_c1c2(wf.qmid, wf.qver, wf.dpt_c1c2().second);
	    assert(wf4.get_dim() == nsub);
	    wf4 = wf4.permCR_signed(); // back to backbone
	    psi4.push_back(wf4);
	 } // i
      } // cturn
   } // forward
   return psi4;
}

} // ctns

#endif
