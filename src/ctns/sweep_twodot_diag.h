#ifndef SWEEP_TWODOT_DIAG_H
#define SWEEP_TWODOT_DIAG_H

#include "oper_dict.h"

namespace ctns{

const bool debug_twodot_diag = false;
extern const bool debug_twodot_diag;

template <typename Tm>
void twodot_diag(const oper_dictmap<Tm>& qops_dict,
		 const stensor4<Tm>& wf,
		 double* diag,
	       	 const int size,
	       	 const int rank,
		 const bool ifdist1){
   const auto& lqops  = qops_dict.at("l");
   const auto& rqops  = qops_dict.at("r");
   const auto& c1qops = qops_dict.at("c1");
   const auto& c2qops = qops_dict.at("c2");
   if(rank == 0 && debug_twodot_diag){
      std::cout << "ctns::twodot_diag ifkr=" << lqops.ifkr 
	        << " size=" << size << std::endl;
   }
   
   // 1. local terms: <lc1c2r|H|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
   // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
   // whose contributions to Diag need to be taken into aacount.
   if(!ifdist1 || rank == 0){
      twodot_diag_local(lqops, rqops, c1qops, c2qops, wf, diag, size, rank);
   }

   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //        B/Q^C1 B/Q^C2
   //         |      |
   // B/Q^L---*------*---B/Q^R
   twodot_diag_BQ("lc1" ,  lqops, c1qops, wf, diag, size, rank);
   twodot_diag_BQ("lc2" ,  lqops, c2qops, wf, diag, size, rank);
   twodot_diag_BQ("lr"  ,  lqops,  rqops, wf, diag, size, rank);
   twodot_diag_BQ("c1c2", c1qops, c2qops, wf, diag, size, rank);
   twodot_diag_BQ("c1r" , c1qops,  rqops, wf, diag, size, rank);
   twodot_diag_BQ("c2r" , c2qops,  rqops, wf, diag, size, rank);
}

template <typename Tm>
void twodot_diag_BQ(const std::string superblock,
		    const oper_dict<Tm>& qops1,
		    const oper_dict<Tm>& qops2,
		    const stensor4<Tm>& wf,
		    double* diag,
       	            const int size,
       	            const int rank){
   const bool ifkr = qops1.ifkr;
   const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
   auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank);
   if(rank == 0 && debug_twodot_diag){ 
      std::cout << "twodot_diag_BQ superblock=" << superblock
      	        << " ifNC=" << ifNC << " " << BQ1 << BQ2 
		<< " size=" << bindex_dist.size() 
		<< std::endl;
   }

/*
   // B^L*Q^R or Q^L*B^R 
#ifdef _OPENMP

   size_t ndim = wf.size();
   #pragma omp parallel
   {
   double* di = new double[ndim];
   memset(di, 0, ndim*sizeof(double));
   #pragma omp for schedule(dynamic) nowait
   for(const auto& index : bindex_dist){
      const auto& O1 = qops1(BQ1).at(index);
      const auto& O2 = qops2(BQ2).at(index);
      if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
      const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
      if(superblock == "lc1"){ 
         twodot_diag_OlOc1(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_OlOc1(wt, O1.K(0), O2.K(0), wf, di);
      }else if(superblock == "lc2"){ 
         twodot_diag_OlOc2(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_OlOc2(wt, O1.K(0), O2.K(0), wf, di);
      }else if(superblock == "lr"){
         twodot_diag_OlOr(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, di);
      }else if(superblock == "c1c2"){
         twodot_diag_Oc1Oc2(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_Oc1Oc2(wt, O1.K(0), O2.K(0), wf, di);
      }else if(superblock == "c1r"){
         twodot_diag_Oc1Or(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_Oc1Or(wt, O1.K(0), O2.K(0), wf, di);
      }else if(superblock == "c2r"){
         twodot_diag_Oc2Or(wt, O1, O2, wf, di);
         if(ifkr) twodot_diag_Oc2Or(wt, O1.K(0), O2.K(0), wf, di);
      }
   } // index
   #pragma omp critical
   linalg::xaxpy(ndim, 1.0, di, diag);
   delete[] di;
   }

#else
*/

   for(const auto& index : bindex_dist){
      const auto& O1 = qops1(BQ1).at(index);
      const auto& O2 = qops2(BQ2).at(index);
      if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
      const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
      if(superblock == "lc1"){ 
         twodot_diag_OlOc1(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_OlOc1(wt, O1.K(0), O2.K(0), wf, diag);
      }else if(superblock == "lc2"){ 
         twodot_diag_OlOc2(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_OlOc2(wt, O1.K(0), O2.K(0), wf, diag);
      }else if(superblock == "lr"){
         twodot_diag_OlOr(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, diag);
      }else if(superblock == "c1c2"){
         twodot_diag_Oc1Oc2(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_Oc1Oc2(wt, O1.K(0), O2.K(0), wf, diag);
      }else if(superblock == "c1r"){
         twodot_diag_Oc1Or(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_Oc1Or(wt, O1.K(0), O2.K(0), wf, diag);
      }else if(superblock == "c2r"){
         twodot_diag_Oc2Or(wt, O1, O2, wf, diag);
         if(ifkr) twodot_diag_Oc2Or(wt, O1.K(0), O2.K(0), wf, diag);
      }
   } // index

//#endif
}

// H[loc] 
template <typename Tm>
void twodot_diag_local(const oper_dict<Tm>& lqops,
		       const oper_dict<Tm>& rqops,
		       const oper_dict<Tm>& c1qops,
		       const oper_dict<Tm>& c2qops,
		       const stensor4<Tm>& wf,
		       double* diag,
		       const int size,
		       const int rank){
   if(rank == 0 && debug_twodot_diag){ 
      std::cout << "twodot_diag_local" << std::endl;
   }
   const auto& Hl  = lqops('H').at(0);
   const auto& Hr  = rqops('H').at(0);
   const auto& Hc1 = c1qops('H').at(0);
   const auto& Hc2 = c2qops('H').at(0);
   Tm Odiagl[maxdim_per_sym];
   Tm Odiagr[maxdim_per_sym];
   Tm Odiagc1[maxdim_per_sym];
   Tm Odiagc2[maxdim_per_sym];

   for(int br=0; br<wf.rows(); br++){
      int rdim = wf.info.qrow.get_dim(br);
      linalg::xcopy(rdim, Hl.start_ptr(br,br), rdim+1, Odiagl, 1);
   for(int bc=0; bc<wf.cols(); bc++){
      int cdim = wf.info.qcol.get_dim(bc);
      linalg::xcopy(cdim, Hr.start_ptr(bc,bc), cdim+1, Odiagr, 1);
   for(int bm=0; bm<wf.mids(); bm++){
      int mdim = wf.info.qmid.get_dim(bm);
      linalg::xcopy(mdim, Hc1.start_ptr(bm,bm), mdim+1, Odiagc1, 1);
   for(int bv=0; bv<wf.vers(); bv++){
      int vdim = wf.info.qver.get_dim(bv);
      linalg::xcopy(vdim, Hc2.start_ptr(bv,bv), vdim+1, Odiagc2, 1);

      size_t off = wf.info._offset[wf.info._addr(br,bc,bm,bv)];
      if(off == 0) continue; 
      size_t ircmv = off-1;
      for(int iv=0; iv<vdim; iv++){
         double dc2 = std::real(Odiagc2[iv]);
         for(int im=0; im<mdim; im++){
	    double dc2c1 = dc2 + std::real(Odiagc1[im]);
            for(int ic=0; ic<cdim; ic++){
	       double dc2c1r = dc2c1 + std::real(Odiagr[ic]);
               for(int ir=0; ir<rdim; ir++){
                  diag[ircmv] += dc2c1r + std::real(Odiagl[ir]);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv

   } // bv
   } // bm
   } // bc
   } // br


/*
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      linalg::xcopy(rdim, Hl.start_ptr(br,br), rdim+1, Odiagl, 1);
      linalg::xcopy(cdim, Hr.start_ptr(bc,bc), cdim+1, Odiagr, 1);
      linalg::xcopy(mdim, Hc1.start_ptr(bm,bm), mdim+1, Odiagc1, 1);
      linalg::xcopy(vdim, Hc2.start_ptr(bv,bv), vdim+1, Odiagc2, 1);
      size_t ircmv = wf.info._offset[idx]-1;
      for(int iv=0; iv<vdim; iv++){
         double dc2 = std::real(Odiagc2[iv]);
         for(int im=0; im<mdim; im++){
	    double dc2c1 = dc2 + std::real(Odiagc1[im]);
            for(int ic=0; ic<cdim; ic++){
	       double dc2c1r = dc2c1 + std::real(Odiagr[ic]);
               for(int ir=0; ir<rdim; ir++){
                  diag[ircmv] += dc2c1r + std::real(Odiagl[ir]);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
*/
}

// Ol*Oc1
template <typename Tm>
void twodot_diag_OlOc1(const double wt,
		       const stensor2<Tm>& Ol,
		       const stensor2<Tm>& Oc1,
		       const stensor4<Tm>& wf,
		       double* diag){
   Tm Odiagl[maxdim_per_sym];
   Tm Odiagc1[maxdim_per_sym];
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Ol*Oc1
      linalg::xcopy(rdim, Ol.start_ptr(br,br), rdim+1, Odiagl, 1);
      linalg::xcopy(mdim, Oc1.start_ptr(bm,bm), mdim+1, Odiagc1, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
	    Tm dc1 = Odiagc1[im];
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
		  Tm dl = Odiagl[ir];
                  diag[ircmv] += wt*std::real(dl*dc1);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
}

// Ol*Oc2
template <typename Tm>
void twodot_diag_OlOc2(const double wt,
		       const stensor2<Tm>& Ol,
		       const stensor2<Tm>& Oc2,
		       const stensor4<Tm>& wf,
		       double* diag){
   Tm Odiagl[maxdim_per_sym];
   Tm Odiagc2[maxdim_per_sym];
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Ol*Oc2
      linalg::xcopy(rdim, Ol.start_ptr(br,br), rdim+1, Odiagl, 1);
      linalg::xcopy(vdim, Oc2.start_ptr(bv,bv), vdim+1, Odiagc2, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
	 Tm dc2 = Odiagc2[iv];
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
		  Tm dl = Odiagl[ir];
                  diag[ircmv] += wt*std::real(dl*dc2);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
}

// Ol*Or
template <typename Tm>
void twodot_diag_OlOr(const double wt,
		      const stensor2<Tm>& Ol,
		      const stensor2<Tm>& Or,
		      const stensor4<Tm>& wf,
		      double* diag){
   Tm Odiagl[maxdim_per_sym];
   Tm Odiagr[maxdim_per_sym];

   for(int br=0; br<wf.rows(); br++){
      int rdim = wf.info.qrow.get_dim(br);
      linalg::xcopy(rdim, Ol.start_ptr(br,br), rdim+1, Odiagl, 1);
   for(int bc=0; bc<wf.cols(); bc++){
      int cdim = wf.info.qcol.get_dim(bc);
      linalg::xcopy(cdim, Or.start_ptr(bc,bc), cdim+1, Odiagr, 1);
   for(int bm=0; bm<wf.mids(); bm++){
      int mdim = wf.info.qmid.get_dim(bm);
   for(int bv=0; bv<wf.vers(); bv++){
      int vdim = wf.info.qver.get_dim(bv);

      size_t off = wf.info._offset[wf.info._addr(br,bc,bm,bv)];
      if(off == 0) continue; 
      size_t ircmv = off-1;
      // Ol*Or
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
	       Tm dr = Odiagr[ic];
               for(int ir=0; ir<rdim; ir++){
		  Tm dl = Odiagl[ir];
                  diag[ircmv] += wt*std::real(dl*dr);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
 
   } // bv
   } // bm
   } // bc
   } // br

/*
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Ol*Or
      linalg::xcopy(rdim, Ol.start_ptr(br,br), rdim+1, Odiagl, 1);
      linalg::xcopy(cdim, Or.start_ptr(bc,bc), cdim+1, Odiagr, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
	       Tm dr = Odiagr[ic];
               for(int ir=0; ir<rdim; ir++){
		  Tm dl = Odiagl[ir];
                  diag[ircmv] += wt*std::real(dl*dr);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
*/

}

// Oc1*Oc2
template <typename Tm>
void twodot_diag_Oc1Oc2(const double wt,
			const stensor2<Tm>& Oc1,
		        const stensor2<Tm>& Oc2,
		        const stensor4<Tm>& wf,
		        double* diag){
   Tm Odiagc1[maxdim_per_sym];
   Tm Odiagc2[maxdim_per_sym];
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Oc1*Oc2
      linalg::xcopy(mdim, Oc1.start_ptr(bm,bm), mdim+1, Odiagc1, 1);
      linalg::xcopy(vdim, Oc2.start_ptr(bv,bv), vdim+1, Odiagc2, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
	 Tm dc2 = Odiagc2[iv];
         for(int im=0; im<mdim; im++){
	    Tm dc1 = Odiagc1[im];
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  diag[ircmv] += wt*std::real(dc1*dc2);
		  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
}

// Oc1*Or
template <typename Tm>
void twodot_diag_Oc1Or(const double wt,
		       const stensor2<Tm>& Oc1,
		       const stensor2<Tm>& Or,
		       const stensor4<Tm>& wf,
		       double* diag){
   Tm Odiagr[maxdim_per_sym];
   Tm Odiagc1[maxdim_per_sym];
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Oc1*Or
      linalg::xcopy(cdim, Or.start_ptr(bc,bc), cdim+1, Odiagr, 1);
      linalg::xcopy(mdim, Oc1.start_ptr(bm,bm), mdim+1, Odiagc1, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
	    Tm dc1 = Odiagc1[im];
            for(int ic=0; ic<cdim; ic++){
	       Tm dr = Odiagr[ic];
               for(int ir=0; ir<rdim; ir++){
                  diag[ircmv] += wt*std::real(dc1*dr);
	          ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
}

// Oc2*Or
template <typename Tm>
void twodot_diag_Oc2Or(const double wt,
		       const stensor2<Tm>& Oc2,
		       const stensor2<Tm>& Or,
		       const stensor4<Tm>& wf,
		       double* diag){
   Tm Odiagr[maxdim_per_sym];
   Tm Odiagc2[maxdim_per_sym];
   for(int i=0; i<wf.info._nnzaddr.size(); i++){
      int idx = wf.info._nnzaddr[i];
      int br, bc, bm, bv;
      wf.info._addr_unpack(idx, br, bc, bm, bv);
      const auto blk = wf(br, bc, bm, bv); 
      int rdim = blk.dim0;
      int cdim = blk.dim1;
      int mdim = blk.dim2;
      int vdim = blk.dim3;
      // Oc2*Or
      linalg::xcopy(cdim, Or.start_ptr(bc,bc), cdim+1, Odiagr, 1);
      linalg::xcopy(vdim, Oc2.start_ptr(bv,bv), vdim+1, Odiagc2, 1);
      size_t ircmv = wf.info._offset[idx]-1;  
      for(int iv=0; iv<vdim; iv++){
	 Tm dc2 = Odiagc2[iv];
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
	       Tm dr = Odiagr[ic];
               for(int ir=0; ir<rdim; ir++){
                  diag[ircmv] += wt*std::real(dc2*dr);
	          ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   } // i
}

} // ctns

#endif
