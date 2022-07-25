#ifndef SWEEP_ONEDOT_DIAG_H
#define SWEEP_ONEDOT_DIAG_H

#include "oper_dict.h"

namespace ctns{

const bool debug_onedot_diag = false;
extern const bool debug_onedot_diag;

template <typename Tm>
void onedot_diag(const oper_dictmap<Tm>& qops_dict,
		  const double ecore,
		  stensor3<Tm>& wf,
		  std::vector<double>& diag,
	       	  const int size,
	       	  const int rank,
		  const bool ifdist1){
   const auto& lqops = qops_dict.at("l"); 
   const auto& rqops = qops_dict.at("r"); 
   const auto& cqops = qops_dict.at("c"); 
   if(rank == 0 && debug_onedot_diag){
      std::cout << "ctns::onedot_diag ifkr=" << lqops.ifkr 
	        << " size=" << size << std::endl;
   }

   // 1. local terms: <lcr|H|lcr> = Hll + Hcc + Hrr
   // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
   // whose contributions to Diag need to be taken into aacount.
   if(!ifdist1 || rank == 0){
      onedot_diag_local(lqops, rqops, cqops, wf, size, rank);
   }else{
      wf.clear();
   }

   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //         B/Q^C
   //         |
   // B/Q^L---*---B/Q^R
   onedot_diag_BQ("lc", lqops, cqops, wf, size, rank);
   onedot_diag_BQ("lr", lqops, rqops, wf, size, rank);
   onedot_diag_BQ("cr", cqops, rqops, wf, size, rank);

   // save to real vector
   double efac = ecore/size;
   std::transform(wf.data(), wf.data()+wf.size(), diag.begin(),
                  [&efac](const Tm& x){ return std::real(x)+efac; });
}

// H[loc] 
template <typename Tm>
void onedot_diag_local(const oper_dict<Tm>& lqops,
		        const oper_dict<Tm>& rqops,
		        const oper_dict<Tm>& cqops,
		        stensor3<Tm>& wf,
			const int size,
			const int rank){
   if(rank == 0 && debug_onedot_diag){ 
      std::cout << "onedot_diag_local" << std::endl;
   }
   const auto& Hl = lqops('H').at(0);
   const auto& Hr = rqops('H').at(0);
   const auto& Hc = cqops('H').at(0);
   int br, bc, bm;
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      wf.info._addr_unpack(idx, br, bc, bm);
      auto blk = wf(br,bc,bm);
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      const auto blkl = Hl(br,br);
      const auto blkr = Hr(bc,bc);
      const auto blkc = Hc(bm,bm);
      for(int im=0; im<mdim; im++){
         for(int ic=0; ic<cdim; ic++){
            for(int ir=0; ir<rdim; ir++){
               blk(ir,ic,im) = blkl(ir,ir) + blkr(ic,ic) + blkc(im,im);
            } // ir
         } // ic
      } // im
   } // i
}

template <typename Tm>
void onedot_diag_BQ(const std::string superblock,
		     const oper_dict<Tm>& qops1,
		     const oper_dict<Tm>& qops2,
		     stensor3<Tm>& wf,
       	             const int size,
       	             const int rank){
   if(rank == 0 && debug_onedot_diag){
      std::cout << "onedot_diag_BQ superblock=" << superblock << std::endl;
   }
   const bool ifkr = qops1.ifkr;
   const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
   auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank);
   if(rank == 0 && debug_onedot_diag){ 
      std::cout << " superblock=" << superblock << " ifNC=" << ifNC 
	        << " " << BQ1 << BQ2 << " size=" << bindex_dist.size() 
		<< std::endl;
   }
   // B^L*Q^R or Q^L*B^R 
   for(const auto& index : bindex_dist){
      const Tm wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
      const auto& O1 = qops1(BQ1).at(index);
      const auto& O2 = qops2(BQ2).at(index);
      if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
      if(superblock == "lc"){ 
         onedot_diag_OlOc(O1,O2,wf,wt);
         if(ifkr) onedot_diag_OlOc(O1.K(0),O2.K(0),wf,wt);
      }else if(superblock == "cr"){
         onedot_diag_OcOr(O1,O2,wf,wt);
         if(ifkr) onedot_diag_OcOr(O1.K(0),O2.K(0),wf,wt);
      }else if(superblock == "lr"){
         onedot_diag_OlOr(O1,O2,wf,wt);
         if(ifkr) onedot_diag_OlOr(O1.K(0),O2.K(0),wf,wt);
      }
   } // index
}

// Ol*Oc*Ir
template <typename Tm>
void onedot_diag_OlOc(const stensor2<Tm>& Ol,
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
void onedot_diag_OlOr(const stensor2<Tm>& Ol,
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
void onedot_diag_OcOr(const stensor2<Tm>& Oc,
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
