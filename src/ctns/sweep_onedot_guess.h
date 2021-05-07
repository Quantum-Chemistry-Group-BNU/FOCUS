#ifndef SWEEP_ONEDOT_GUESS_H
#define SWEEP_ONEDOT_GUESS_H

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

} // ctns

#endif
