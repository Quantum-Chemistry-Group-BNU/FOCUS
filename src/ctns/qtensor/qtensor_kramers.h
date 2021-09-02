#ifndef QTENSOR_KRAMERS_H
#define QTENSOR_KRAMERS_H

namespace ctns{

// generate matrix representation for Kramers paired operators
// suppose row and col are KRS-adapted basis, then
//    <r|\bar{O}|c> = (K<r|\bar{O}|c>)*
//    		    = p{O} <\bar{r}|O|\bar{c}>*
// using \bar{\bar{O}} = p{O} O (p{O}: 'parity' of operator)
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor2<Tm> qt2(sym.flip(), qrow, qcol, dir); // the symmetry is flipped
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk = qt2(br,bc);
         if(blk.size() == 0) continue;
	 const auto& blk1 = _qblocks[_addr(br,bc)];
	 int pr = qrow.get_parity(br);
	 int pc = qcol.get_parity(bc);
	 blk = fpo*blk1.time_reversal(pr, pc); 
      } // bc
   } // br
   return qt2;
}

// ZL20210413: application of time-reversal operation
template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor3<Tm> qt3(sym, qmid, qrow, qcol, dir); // assuming it only works for (N), no flip of symmetry is necessary
   for(int idx=0; idx<qt3._qblocks.size(); idx++){
      auto& blk = qt3._qblocks[idx];
      if(blk.size() == 0) continue;
      int bm,br,bc;
      _addr_unpack(idx,bm,br,bc);
      // qt3[c](l,r) = blk[bar{c}](bar{l},bar{r})^*
      const auto& blk1 = _qblocks[idx];
      int pm = qmid.get_parity(bm);
      int pr = qrow.get_parity(br);
      int pc = qcol.get_parity(bc);
      if(pm == 0){
         // c[e]
         for(int im=0; im<blk.size(); im++){
            blk[im] = fpo*blk1[im].time_reversal(pr, pc);
         }
      }else{
         assert(blk.size()%2 == 0);
         int dm2 = blk.size()/2;
         // c[o],c[\bar{o}]
         for(int im=0; im<dm2; im++){
            blk[im] = fpo*blk1[im+dm2].time_reversal(pr, pc);
         }
         for(int im=0; im<dm2; im++){
            blk[im+dm2] = -fpo*blk1[im].time_reversal(pr, pc);
         }
      } // pm
   } // idx
   return qt3;
}

// ZL20210510: application of time-reversal operation
template <typename Tm>
qtensor4<Tm> qtensor4<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor4<Tm> qt4(sym, qmid, qver, qrow, qcol); 
   for(int idx=0; idx<qt4._qblocks.size(); idx++){
      auto& blk = qt4._qblocks[idx];
      if(blk.size() == 0) continue;
      int bm,bv,br,bc;
      _addr_unpack(idx,bm,bv,br,bc);
      // qt4_new(c1c2)[l,r] = qt4(c1c2_bar)[l_bar,r_bar]^*
      const auto& blk1 = _qblocks[idx];
      int pm = qmid.get_parity(bm);
      int pv = qver.get_parity(bv);
      int pr = qrow.get_parity(br);
      int pc = qcol.get_parity(bc);
      int mdim = qmid.get_dim(bm);
      int vdim = qver.get_dim(bv);
      if(pm == 0 && pv == 0){
         for(int imv=0; imv<blk.size(); imv++){
            blk[imv] = fpo*blk1[imv].time_reversal(pr, pc);
         }
      }else if(pm == 0 && pv == 1){
	 assert(vdim%2 == 0);
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
               int imv  = iv*mdim + im;
	       int imv2 = (iv+vdim2)*mdim + im;
	       blk[imv] = fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
	       int imv  = (iv+vdim2)*mdim + im;
	       int imv2 = iv*mdim + im;
	       blk[imv] = -fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	 }
      }else if(pm == 1 && pv == 0){
	 assert(mdim%2 == 0);
	 int mdim2 = mdim/2;
	 for(int iv=0; iv<vdim; iv++){
	    for(int im=0; im<mdim2; im++){
               int imv  = iv*mdim + im;
	       int imv2 = iv*mdim + (im+mdim2);
	       blk[imv] = fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = iv*mdim + (im+mdim2);
	       int imv2 = iv*mdim + im;
	       blk[imv] = -fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	 }
      }else if(pm == 1 && pv == 1){
	 assert(mdim%2 == 0 && vdim%2 == 0);
	 int mdim2 = mdim/2;
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
               int imv  = iv*mdim + im;
	       int imv2 = (iv+vdim2)*mdim + (im+mdim2);
	       blk[imv] = fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = iv*mdim + (im+mdim2);
	       int imv2 = (iv+vdim2)*mdim + im;
	       blk[imv] = -fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
	       int imv  = (iv+vdim2)*mdim + im;
               int imv2 = iv*mdim + (im+mdim2);
	       blk[imv] = -fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = (iv+vdim2)*mdim + (im+mdim2);
	       int imv2 = iv*mdim + im;
	       blk[imv] = fpo*blk1[imv2].time_reversal(pr, pc);
	    }
	 }
      } // (pm,pv)
   } // idx
   return qt4;
}

} // ctns

#endif
