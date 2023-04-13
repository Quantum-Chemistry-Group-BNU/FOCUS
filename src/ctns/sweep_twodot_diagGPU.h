#ifdef GPU

#ifndef SWEEP_TWODOT_DIAGGPU_H
#define SWEEP_TWODOT_DIAGGPU_H

#include "oper_dict.h"
#include "gpu_kernel/twodot_diagGPU_kernel.h"

namespace ctns{

   template <typename Tm>
      void twodot_diagGPU_BQ(const std::string superblock,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor4<Tm>& wf,
            const size_t nblk,
            double* dev_diag,
            size_t* dev_dims,
            std::vector<size_t>& opoffs,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank);
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diagGPU_BQ superblock=" << superblock
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }

         // B^L*Q^R or Q^L*B^R 
         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
            const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H

            if(superblock == "lc1"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(br,br)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bm,bm)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_OlOc1(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            }else if(superblock == "lc2"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(br,br)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bv,bv)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_OlOc2(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            }else if(superblock == "lr"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(br,br)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bc,bc)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_OlOr(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            }else if(superblock == "c1c2"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(bm,bm)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bv,bv)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_Oc1Oc2(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            }else if(superblock == "c1r"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(bm,bm)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bc,bc)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_Oc1Or(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            }else if(superblock == "c2r"){

               for(int i=0; i<wf.info._nnzaddr.size(); i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info._offset[O1.info._addr(bv,bv)]-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info._offset[O2.info._addr(bc,bc)]-1;
               }
               GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_Oc2Or(nblk, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt);

            } // endif

         } // index
      }

   template <typename Tm>
      void twodot_diagGPU(const oper_dictmap<Tm>& qops_dict,
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
            std::cout << "ctns::twodot_diagGPU ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }

         size_t nblk = wf.info._nnzaddr.size();
         size_t ndim = wf.size();
         double* dev_diag = (double*)GPUmem.allocate(ndim*sizeof(ndim));
         size_t* dev_dims = (size_t*)GPUmem.allocate(nblk*9*sizeof(size_t));

         std::vector<size_t> blkdims(nblk*5,0);
         for(int i=0; i<nblk; i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            blkdims[5*i]   = blk.dim0;
            blkdims[5*i+1] = blk.dim1;
            blkdims[5*i+2] = blk.dim2;
            blkdims[5*i+3] = blk.dim3;
            blkdims[5*i+4] = wf.info._offset[idx]-1;
         }
         GPUmem.to_gpu(dev_dims, blkdims.data(), nblk*5*sizeof(size_t));

         std::vector<size_t> opoffs(nblk*4,0);
         // 1. local terms: <lc1c2r|H|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
         // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
         // whose contributions to Diag need to be taken into aacount.
         if(!ifdist1 || rank == 0){
            const auto& Hl  = lqops('H').at(0);
            const auto& Hr  = rqops('H').at(0);
            const auto& Hc1 = c1qops('H').at(0);
            const auto& Hc2 = c2qops('H').at(0);
            for(int i=0; i<nblk; i++){
               int idx = wf.info._nnzaddr[i];
               int br, bc, bm, bv;
               wf.info._addr_unpack(idx, br, bc, bm, bv);
               opoffs[4*i]   =  lqops._offset.at(std::make_pair('H',0)) + Hl.info._offset[Hl.info._addr(br,br)]-1;
               opoffs[4*i+1] =  rqops._offset.at(std::make_pair('H',0)) + Hr.info._offset[Hl.info._addr(bc,bc)]-1;
               opoffs[4*i+2] = c1qops._offset.at(std::make_pair('H',0)) + Hc1.info._offset[Hl.info._addr(bm,bm)]-1;
               opoffs[4*i+3] = c2qops._offset.at(std::make_pair('H',0)) + Hc2.info._offset[Hl.info._addr(bv,bv)]-1;
            }
            GPUmem.to_gpu(dev_dims+nblk*5, opoffs.data(), nblk*4*sizeof(size_t));
            twodot_diagGPU_local(nblk, dev_diag, dev_dims, 
                  lqops._dev_data, rqops._dev_data,
                  c1qops._dev_data, c2qops._dev_data); 
         }

         // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
         //        B/Q^C1 B/Q^C2
         //         |      |
         // B/Q^L---*------*---B/Q^R
         twodot_diagGPU_BQ("lc1" ,  lqops, c1qops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("lc2" ,  lqops, c2qops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("lr"  ,  lqops,  rqops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c1c2", c1qops, c2qops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c1r" , c1qops,  rqops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c2r" , c2qops,  rqops, wf, nblk, dev_diag, dev_dims, opoffs, size, rank);

         GPUmem.to_cpu(diag, dev_diag, ndim*sizeof(ndim));         
         GPUmem.deallocate(dev_dims, nblk*9*sizeof(ndim));
         GPUmem.deallocate(dev_diag, ndim*sizeof(ndim));
      }

} // ctns

#endif

#endif
