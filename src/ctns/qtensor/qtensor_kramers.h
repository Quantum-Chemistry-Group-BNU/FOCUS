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


//
// QTensor-related, used in sweep_dvdson.h
//
namespace kramers{

template <typename Tm, typename QTm> 
void get_krvec_qt(Tm* y, Tm* ykr, QTm& wf, const int parity=1){
   wf.from_array(y);
   wf.K(parity).to_array(ykr);
}

// Odd case: 
template <typename Tm, typename QTm>
int get_ortho_basis_qt(const int ndim,
  	      	       const int neig,
  		       const int nres,
  		       const std::vector<Tm>& vbas,
  		       std::vector<Tm>& rbas,
		       QTm& wf,
  		       const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // projection (1-V*V^+)*R = R-V*(V^+R)
   std::vector<Tm> vtr(neig*nres);
   for(int repeat=0; repeat<maxtimes; repeat++){
      linalg::xgemm("C","N",&neig,&nres,&ndim,
	            &one,vbas.data(),&ndim,rbas.data(),&ndim,
	            &zero,vtr.data(),&neig);
      linalg::xgemm("N","N",&ndim,&nres,&neig,
	            &mone,vbas.data(),&ndim,vtr.data(),&neig,
	            &one,rbas.data(),&ndim);
   }
   // form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas_new[nindp*ndim]);
      nindp += 1;
      // add its time-reversal partner
      get_krvec_qt(&rbas[i*ndim], krvec.data(), wf);
      std::copy(krvec.cbegin(), krvec.cend(), &rbas_new[nindp*ndim]);
      nindp += 1;
      // debug
      if(debug_ortho){
         linalg::matrix<Tm> V(ndim,neig+nindp);
	 std::copy(vbas.begin(), vbas.begin()+ndim*neig, V.col(0));
	 std::copy(rbas_new.begin(), rbas_new.begin()+ndim*nindp, V.col(neig));
         auto ova = xgemm("C","N",V,V);
         ova.print("ova");
	 auto dev = ova - linalg::identity_matrix<Tm>(neig+nindp);
	 double diff = normF(dev);
	 std::cout << "diff=" << diff << std::endl;
      }
      //-------------------------------------------------------------
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) continue;
      std::vector<Tm> vtr(neig*N), rtr(nindp*N);
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-V*V^+)*R_rest
	 linalg::xgemm("C","N",&neig,&N,&ndim,
                       &one,vbas.data(),&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,vtr.data(),&neig);
         linalg::xgemm("N","N",&ndim,&N,&neig,
                       &mone,vbas.data(),&ndim,vtr.data(),&neig,
                       &one,&rbas[(i+1)*ndim],&ndim);
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         linalg::xgemm("C","N",&nindp,&N,&ndim,
                       &one,&rbas_new[0],&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,rtr.data(),&nindp);
         linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas_new[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   rbas = rbas_new;
   // Orthonormality is essential
   linalg::check_orthogonality(ndim, nindp, rbas);
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm, typename QTm>
int get_ortho_basis_qt(const int ndim,
		       const int nres,
		       std::vector<Tm>& rbas,
		       QTm& wf,
		       const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas_new[nindp*ndim]);
      nindp += 1;
      // add its time-reversal partner
      get_krvec_qt(&rbas[i*ndim], krvec.data(), wf);
      std::copy(krvec.cbegin(), krvec.cend(), &rbas_new[nindp*ndim]);
      nindp += 1;
      // debug
      if(debug_ortho){
         // check psi[lr] = psi_bar[l_bar,r_bar]*
         std::vector<Tm> tmp(ndim);
         get_krvec_qt(&rbas_new[(nindp-1)*ndim], tmp.data(), wf, 0);
         std::transform(tmp.begin(), tmp.end(), &rbas[i*ndim], krvec.begin(),
                        [](const Tm& x, const Tm& y){ return x-y; }); 
	 auto diff = linalg::xnrm2(ndim, krvec.data());
         std::cout << "diff[psi]=" << diff << std::endl;
	 if(diff > 1.e-10) tools::exit("error: in consistent psi and psi_bar!");
	 // check overlap matrix for basis
         linalg::matrix<Tm> V(ndim,nindp,rbas_new.data());
         auto ova = xgemm("C","N",V,V);
         ova.print("ova");
      }
      //-------------------------------------------------------------
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      // R_rest = (1-Rnew*Rnew^+)*R_rest
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // rtr = Rnew^+*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,&rbas_new[0],&ndim,&rbas[(i+1)*ndim],&ndim,
               	       &zero,rtr.data(),&nindp);
	 // R_rest -= Rnew*rtr
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas_new[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   rbas = rbas_new;
   // Orthonormality is essential
   linalg::check_orthogonality(ndim, nindp, rbas);
   return nindp;
}

} // kramers

#endif
