#ifndef SWEEP_ONEDOT_DIAG_H
#define SWEEP_ONEDOT_DIAG_H

#include "oper_dict.h"

namespace ctns{

const bool debug_onedot_hdiag = false;
extern const bool debug_onedot_hdiag;

template <typename Tm>
void onedot_Hdiag(const oper_dictmap<Tm>& qops_dict,
		  const double ecore,
		  stensor3<Tm>& wf,
		  std::vector<double>& diag,
	       	  const int size,
	       	  const int rank){
   const auto& lqops = qops_dict.at("l"); 
   const auto& rqops = qops_dict.at("r"); 
   const auto& cqops = qops_dict.at("c"); 
   if(rank == 0 && debug_onedot_hdiag){
      std::cout << "ctns::onedot_Hdiag ifkr=" << lqops.ifkr 
	        << " size=" << size << std::endl;
   }

   // 1. local terms: <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   onedot_Hdiag_local(lqops, rqops, cqops, ecore/size, wf, size, rank);

   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //         B/Q^C
   //         |
   // B/Q^L---*---B/Q^R
   onedot_Hdiag_BQ("lc", lqops, cqops, wf, size, rank);
   onedot_Hdiag_BQ("lr", lqops, rqops, wf, size, rank);
   onedot_Hdiag_BQ("cr", cqops, rqops, wf, size, rank);

   // save to real vector
   std::transform(wf.data(), wf.data()+wf.size(), diag.begin(),
                  [](const Tm& x){ return std::real(x); });
}

// H[loc] = H[l]I[c]I[r] + I[l]H[c]I[r] + I[l]I[c]H[r]
template <typename Tm>
void onedot_Hdiag_local(const oper_dict<Tm>& lqops,
		        const oper_dict<Tm>& rqops,
		        const oper_dict<Tm>& cqops,
			const double ecore,
		        stensor3<Tm>& wf,
			const int size,
			const int rank){
   if(rank == 0 && debug_onedot_hdiag){ 
      std::cout << "onedot_Hdiag_local" << std::endl;
   }
   const auto& Hl = lqops('H').at(0);
   const auto& Hr = rqops('H').at(0);
   const auto& Hc = cqops('H').at(0);
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   int br, bc, bm;
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      wf.info._addr_unpack(idx, br, bc, bm);
      auto blk = wf(br,bc,bm);
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      // 1. local contributions: all four indices in c/l/r
      const auto blkl = Hl(br,br); // left->row 
      const auto blkr = Hr(bc,bc); // row->col
      const auto blkc = Hc(bm,bm); // central->mid 
      for(int im=0; im<mdim; im++){
         for(int ic=0; ic<cdim; ic++){
            for(int ir=0; ir<rdim; ir++){
               blk(ir,ic,im) = ecore + blkl(ir,ir) + blkr(ic,ic) + blkc(im,im);
            } // ir
         } // ic
      } // im
   } // i
}

template <typename Tm>
void onedot_Hdiag_BQ(const std::string superblock,
		     const oper_dict<Tm>& qops1,
		     const oper_dict<Tm>& qops2,
		     stensor3<Tm>& wf,
       	             const int size,
       	             const int rank){
   if(rank == 0 && debug_onedot_hdiag){
      std::cout << "onedot_Hdiag_BQ superblock=" << superblock << std::endl;
   }
   const bool ifkr = qops1.ifkr;
   const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
   auto bindex = oper_index_opB(cindex, ifkr);
   if(rank == 0 && debug_onedot_hdiag){ 
      std::cout << " superblock=" << superblock << " ifNC=" << ifNC 
	        << " " << BQ1 << BQ2 << " size=" << bindex.size() 
		<< std::endl;
   }
   // B^L*Q^R or Q^L*B^R 
   for(const auto& index : bindex){
      int iproc = distribute2(ifkr,size,index);
      if(iproc == rank){
         const Tm wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
         const auto& O1 = qops1(BQ1).at(index);
         const auto& O2 = qops2(BQ2).at(index);
         if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
	 if(superblock == "lc"){ 
            onedot_Hdiag_OlOc(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OlOc(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "cr"){
            onedot_Hdiag_OcOr(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OcOr(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "lr"){
            onedot_Hdiag_OlOr(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OlOr(O1.K(0),O2.K(0),wf,wt);
	 }
      } // iproc
   } // index
}

// Ol*Oc*Ir
template <typename Tm>
void onedot_Hdiag_OlOc(const stensor2<Tm>& Ol,
		       const stensor2<Tm>& Oc,
		       stensor3<Tm>& wf,
		       const Tm wt=1.0){
   int br, bc, bm;
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      wf.info._addr_unpack(idx, br, bc, bm);
      auto blk = wf(br,bc,bm);
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      // Ol*Or 
      const auto blkl = Ol(br,br);
      const auto blkc = Oc(bm,bm);
      for(int im=0; im<mdim; im++){
         for(int ic=0; ic<cdim; ic++){
            for(int ir=0; ir<rdim; ir++){
               blk(ir,ic,im) += wt*blkl(ir,ir)*blkc(im,im);
            } // ir
         } // ic
      } // im
   } // i
}

// Ol*Ic*Or
template <typename Tm>
void onedot_Hdiag_OlOr(const stensor2<Tm>& Ol,
		       const stensor2<Tm>& Or,
		       stensor3<Tm>& wf,
		       const Tm wt=1.0){
   int br, bc, bm;
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      wf.info._addr_unpack(idx, br, bc, bm);
      auto blk = wf(br,bc,bm);
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      // Ol*Or
      const auto blkl = Ol(br,br);
      const auto blkr = Or(bc,bc);
      for(int im=0; im<mdim; im++){
         for(int ic=0; ic<cdim; ic++){
            for(int ir=0; ir<rdim; ir++){
               blk(ir,ic,im) += wt*blkl(ir,ir)*blkr(ic,ic);
            } // ir
         } // ic
      } // im
   } // i
}

// Il*Oc*Or
template <typename Tm>
void onedot_Hdiag_OcOr(const stensor2<Tm>& Oc,
		       const stensor2<Tm>& Or,
		       stensor3<Tm>& wf,
		       const Tm wt=1.0){
   int br, bc, bm;
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      wf.info._addr_unpack(idx, br, bc, bm);
      auto blk = wf(br,bc,bm);
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      // Oc*Or
      const auto blkc = Oc(bm,bm);
      const auto blkr = Or(bc,bc);
      for(int im=0; im<mdim; im++){
         for(int ic=0; ic<cdim; ic++){
            for(int ir=0; ir<rdim; ir++){
               blk(ir,ic,im) += wt*blkc(im,im)*blkr(ic,ic);
            } // ir
         } // ic
      } // im
   } // i
}

} // ctns

#endif
