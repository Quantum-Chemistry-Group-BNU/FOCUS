#ifdef GPU

#ifndef SWEEP_TWODOT_DIAGGPU_H
#define SWEEP_TWODOT_DIAGGPU_H

#include "oper_dict.h"
#include "gpu_kernel/twodot_diagGPU_kernel.h"

namespace ctns{

   template <typename Tm>
      void twodot_diagGPU(const oper_dictmap<Tm>& qops_dict,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifnccl,
            const bool diagcheck){
         const auto& lqops  = qops_dict.at("l");
         const auto& rqops  = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         if(rank == 0 && debug_twodot_diag){
            std::cout << "ctns::twodot_diagGPU ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }
         auto t0 = tools::get_time();

         // initialize dev_diag on GPU
         size_t used = GPUmem.used();
         size_t nblk = wf.info._nnzaddr.size();
         size_t ndim = wf.size();
         double* dev_diag = (double*)GPUmem.allocate(ndim*sizeof(double));
         GPUmem.memset(dev_diag, ndim*sizeof(double));
         auto t1 = tools::get_time();

         // pack block information (dim0,dim1,dim2,dim3,offset)
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
         auto t2 = tools::get_time();

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
               opoffs[4*i]   =  lqops._offset.at(std::make_pair('H',0)) + Hl.info.get_offset(br,br)-1;
               opoffs[4*i+1] =  rqops._offset.at(std::make_pair('H',0)) + Hr.info.get_offset(bc,bc)-1;
               opoffs[4*i+2] = c1qops._offset.at(std::make_pair('H',0)) + Hc1.info.get_offset(bm,bm)-1;
               opoffs[4*i+3] = c2qops._offset.at(std::make_pair('H',0)) + Hc2.info.get_offset(bv,bv)-1;
            }
            GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*4*sizeof(size_t));
            twodot_diagGPU_local(nblk, ndim, dev_diag, dev_dims, 
                  lqops._dev_data, rqops._dev_data,
                  c1qops._dev_data, c2qops._dev_data);
         }
         auto t3 = tools::get_time();

         // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
         //        B/Q^C1 B/Q^C2
         //         |      |
         // B/Q^L---*------*---B/Q^R
         twodot_diagGPU_BQ("lc1" ,  lqops, c1qops, wf, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("lc2" ,  lqops, c2qops, wf, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("lr"  ,  lqops,  rqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c1c2", c1qops, c2qops, wf, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c1r" , c1qops,  rqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         twodot_diagGPU_BQ("c2r" , c2qops,  rqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         auto t4 = tools::get_time();

         // debug
         if(diagcheck) twodot_diagGPU_check(ndim, dev_diag, qops_dict, wf, size, rank, ifdist1);

         GPUmem.deallocate(dev_dims, nblk*9*sizeof(size_t));
         auto t5 = tools::get_time();
         if(!ifnccl){
            GPUmem.to_cpu(diag, dev_diag, ndim*sizeof(double));
#ifdef NCCL
         }else{
            nccl_comm.reduce(dev_diag, ndim, 0);
            if(rank==0) GPUmem.to_cpu(diag, dev_diag, ndim*sizeof(double));
#endif
         }
         auto t6 = tools::get_time();
         GPUmem.deallocate(dev_diag, ndim*sizeof(double));
         assert(used == GPUmem.used());
         auto t7 = tools::get_time();
         if(rank == 0){
            std::cout << "### DIAG TIMING: total=" 
               << tools::get_duration(t7-t0)
               << " t1=" << tools::get_duration(t1-t0) << ","
               << " t2=" << tools::get_duration(t2-t1) << ","
               << " t3=" << tools::get_duration(t3-t2) << ","
               << " t4=" << tools::get_duration(t4-t3) << ","
               << " t5=" << tools::get_duration(t5-t4) << ","
               << " t6=" << tools::get_duration(t6-t5) << ","
               << " t7=" << tools::get_duration(t7-t6) 
               << std::endl;
         }
      }

   template <bool ifab, typename Tm>
      void twodot_diagGPU_check(const int ndim,
            const double* dev_diag,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const qtensor4<ifab,Tm>& wf,
            const int size,
            const int rank,
            const bool ifdist1){
         double* diag1 = new double[ndim];
         GPUmem.to_cpu(diag1, dev_diag, ndim*sizeof(double));
         GPUmem.sync();
         double* diag2 = new double[ndim];
         twodot_diag(qops_dict, wf, diag2, size, rank, ifdist1);
         for(int i=0; i<ndim; i++){
            std::cout << "rank=" << rank << " i=" << i 
               << " di[gpu]=" << diag1[i] << " di[cpu]=" << diag2[i]
               << " diff=" << diag2[i]-diag1[i]
               << std::endl;
         }
         linalg::xaxpy(ndim, -1.0, diag1, diag2);
         auto diff = linalg::xnrm2(ndim, diag2);
         std::cout << "rank=" << rank << " diff[tot]=" << diff << std::endl;
         if(diff > 1.e-8){
            std::cout << "error: diff is too large!"  << std::endl;
            exit(1);
         }
         delete[] diag1;
         delete[] diag2;
      }

   template <typename Tm>
      void twodot_diagGPU_BQ(const std::string superblock,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor4<Tm>& wf,
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
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diagGPU_BQ superblock=" << superblock
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }

         // B^L*Q^R or Q^L*B^R 
         size_t nblk = wf.info._nnzaddr.size();
         size_t ndim = wf.size();
         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
            const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H

            if(superblock == "lc1"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bm,bm)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 0, 2);

            }else if(superblock == "lc2"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bv,bv)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 0, 3);

            }else if(superblock == "lr"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 0, 1);

            }else if(superblock == "c1c2"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bv,bv)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 2, 3);

            }else if(superblock == "c1r"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 2, 1);

            }else if(superblock == "c2r"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm, bv;
                  wf.info._addr_unpack(idx, br, bc, bm, bv);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bv,bv)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 3, 1);

            } // endif

         } // index
      }

} // ctns

#endif

#endif
