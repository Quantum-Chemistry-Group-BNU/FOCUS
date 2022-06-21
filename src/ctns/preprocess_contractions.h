#ifndef PREPROCESS_CONTRACTIONS_H
#define PREPROCESS_CONTRACTIONS_H

namespace ctns{

template <typename Tm>
struct Hxblock{
public:
   int dimout[4] = {0,0,0,0};
   int dimin[4] = {0,0,0,0};
   Tm* data[4] = {nullptr,nullptr,nullptr,nullptr};
   size_t offout = 0, offin = 0;
};
template <typename Tm>
using Hxblocks = std::vector<Hxblock<Tm>>;

template <typename Tm>
struct Hmu_ptr{
public:
   Hxblocks<Tm> gen_Hxblocks(const qinfo4<Tm>& wf);
public:
   qinfo2<Tm>* info[4] = {nullptr,nullptr,nullptr,nullptr};
   Tm* data[4] = {nullptr,nullptr,nullptr,nullptr};
   bool parity[4] = {false,false,false,false};
   bool dagger[4] = {false,false,false,false};
   Tm coeff = 1.0; 
};

//
// sigma[br,bc,bm,bv] = Ol[br,br'] Or[bc,bc'] Oc1[bm,bm'] Oc2[bv,bv'] wf[br',bc',bm',bv']
//
template <typename Tm>
Hxblocks<Tm> Hmu_ptr<Tm>::gen_Hxblocks(const qinfo4<Tm>& wf_info){
   Hxblocks<Tm> Hxblks;
/*
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
*/
/*
      // finding the block
      for(int k=0; k<4; k++){
         if(info[k]==nullptr){ // identity operator
            bi[k] = bo[k];
         }else{
	    bi[k] = info[0].get_bcol(bo[k]);
	    Hxblk.data[4] = Hmu_ptr.data[k]+
         }
      }
      if(bi[0]==-1 || bi[1]==-1 || bi[2]==-1 || bi[3]==-1) continue;
      size_t offin = wf_info._offset[wf.info._addr(bi[0],bi[1],bi[2],bi[3])];
      if(offin == 0) continue; // in case of no matching contractions
      Hxblk.offin = offin-1;
      Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
      Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
      Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
      Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);


      // how to deal with transpose?    

      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._rows; bx++){
	 size_t off4a = qt4a_info._offset[qt4a_info._addr(bx,bc,bm,bv)];
	 if(off4a == 0) continue;
	 int jdx = iftrans? qt2_info._addr(bx,br) : qt2_info._addr(br,bx);
	 size_t off2 = qt2_info._offset[jdx];
         if(off2 == 0) continue;
	 ifzero = false; 
         // qt4(r,c,m,v) = \sum_x qt2(r,x)*qt4a(x,c,m,v) ; iftrans=false 
         // 		 = \sum_x qt2(x,r)*qt4a(x,c,m,v) ; iftrans=true
         const Tm* blk4a = qt4a_data + off4a-1;
         const Tm* blk2 = qt2_data + off2-1;
         int xdim = qt4a_info.qrow.get_dim(bx);
         int LDA = iftrans? xdim : rdim;
         int cmvdim = cdim*mdim*vdim;
	 linalg::xgemm(transa, "N", &rdim, &cmvdim, &xdim, &alpha,
		       blk2, &LDA, blk4a, &xdim, &beta,
		       blk4, &rdim);
      } // bx
      if(ifzero) memset(blk4, 0, size*sizeof(Tm));
   } // i
*/
   return Hxblks;
}

} // ctns

#endif
