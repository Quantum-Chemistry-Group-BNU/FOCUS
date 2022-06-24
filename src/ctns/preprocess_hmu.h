#ifndef PREPROCESS_HMU_H
#define PREPROCESS_HMU_H

#include "preprocess_hxlist.h"

namespace ctns{

// H[mu] = Ol*Or*Oc1*Oc2
template <typename Tm>
struct Hmu_ptr{
public:
   size_t gen_Hxlist(const qinfo4<Tm>& wf, 
		     Hxlist<Tm>& Hxblks,
		     const bool ifdagger) const;
public:
   bool parity[4] = {false,false,false,false};
   bool dagger[4] = {false,false,false,false};
   int location[4] = {-1,-1,-1,-1}; 
   size_t offop[4] = {0,0,0,0};
   qinfo2<Tm>* info[4] = {nullptr,nullptr,nullptr,nullptr};
   Tm coeff = 1.0, coeffH = 1.0; 
};

// sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
// 			Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv'] 
// 			wf[br',bc',bm',bv']
template <typename Tm>
size_t Hmu_ptr<Tm>::gen_Hxlist(const qinfo4<Tm>& wf_info,
			       Hxlist<Tm>& Hxlst,
			       const bool ifdagger) const{
   size_t blksize = 0;
   int bo[4], bi[4];
   for(int i=0; i<wf_info._nnzaddr.size(); i++){
      int idx = wf_info._nnzaddr[i];
      wf_info._addr_unpack(idx,bo[0],bo[1],bo[2],bo[3]);
      Hxblock<Tm> Hxblk;
      Hxblk.offout = wf_info._offset[idx]-1;
      Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
      Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
      Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
      Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
      Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
      // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
      bool symAllowed = true;
      for(int k=0; k<4; k++){
	 Hxblk.dagger[k] = dagger[k]^ifdagger;
         if(location[k] == -1){ // identity operator
            bi[k] = bo[k];
         }else{
	    bool iftrans = dagger[k]^ifdagger;
	    bi[k] = iftrans? info[k]->_bc2br[bo[k]] : info[k]->_br2bc[bo[k]];
	    if(bi[k] == -1){
	       symAllowed = false;
	       break;
	    }else{
	       Hxblk.location[k] = location[k];
	       int jdx = iftrans? info[k]->_addr(bi[k],bo[k]) : info[k]->_addr(bo[k],bi[k]);
	       assert(info[k]->_offset[jdx] != 0);
	       Hxblk.offop[k] = offop[k]+(info[k]->_offset[jdx]-1);
	    }
         }
      }
      if(!symAllowed) continue;
      size_t offin = wf_info._offset[wf_info._addr(bi[0],bi[1],bi[2],bi[3])];
      if(offin == 0) continue; // in case of no matching contractions
      Hxblk.offin = offin-1;
      Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
      Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
      Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
      Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
      // compute sign due to parity
      Hxblk.coeff = ifdagger? coeffH : coeff;
      int pl  = wf_info.qrow.get_parity(bi[0]);
      int pc1 = wf_info.qmid.get_parity(bi[2]);
      int pc2 = wf_info.qver.get_parity(bi[3]);
      if(parity[1] && (pl+pc1+pc2)%2==1) Hxblk.coeff *= -1.0; // Or
      if(parity[3] && (pl+pc1)%2==1) Hxblk.coeff *= -1.0;  // Oc2
      if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
      Hxlst.push_back(Hxblk);
      blksize = std::max(blksize, std::max(Hxblk.dimin[0],Hxblk.dimout[0])*
		      	          std::max(Hxblk.dimin[1],Hxblk.dimout[1])*
				  std::max(Hxblk.dimin[2],Hxblk.dimout[2])*
				  std::max(Hxblk.dimin[3],Hxblk.dimout[3]));
   } // i
   return blksize;
}

} // ctns

#endif
