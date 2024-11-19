#ifdef GPU

#ifndef SWEEP_ONEDOT_DIAGGPU_H
#define SWEEP_ONEDOT_DIAGGPU_H

#include "oper_dict.h"
#include "gpu_kernel/onedot_diagGPU_kernel.h"

namespace ctns{

   template <typename Tm>
      void onedot_diagGPU(const oper_dictmap<Tm>& qops_dict,
            const stensor3<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifnccl,
            const bool diagcheck){
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         if(rank == 0 && debug_onedot_diag){
            std::cout << "ctns::onedot_diagGPU ifkr=" << lqops.ifkr 
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

         // pack block information (dim0,dim1,dim2,offset)
         size_t* dev_dims = (size_t*)GPUmem.allocate(nblk*7*sizeof(size_t));
         std::vector<size_t> blkdims(nblk*4,0);
         for(int i=0; i<nblk; i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm;
            wf.info._addr_unpack(idx, br, bc, bm);
            auto blk = wf(br,bc,bm);
            blkdims[4*i]   = blk.dim0;
            blkdims[4*i+1] = blk.dim1;
            blkdims[4*i+2] = blk.dim2;
            blkdims[4*i+3] = wf.info._offset[idx]-1;
         }
         GPUmem.to_gpu(dev_dims, blkdims.data(), nblk*4*sizeof(size_t));
         auto t2 = tools::get_time();

         std::vector<size_t> opoffs(nblk*3,0);
         // 1. local terms: <lcr|H|lcr> = Hll + Hcc + Hrr
         // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
         // whose contributions to Diag need to be taken into aacount.
         if(!ifdist1 || rank == 0){
            const auto& Hl = lqops('H').at(0);
            const auto& Hr = rqops('H').at(0);
            const auto& Hc = cqops('H').at(0);
            for(int i=0; i<nblk; i++){
               int idx = wf.info._nnzaddr[i];
               int br, bc, bm;
               wf.info._addr_unpack(idx, br, bc, bm);
               opoffs[3*i]   = lqops._offset.at(std::make_pair('H',0)) + Hl.info.get_offset(br,br)-1;
               opoffs[3*i+1] = rqops._offset.at(std::make_pair('H',0)) + Hr.info.get_offset(bc,bc)-1;
               opoffs[3*i+2] = cqops._offset.at(std::make_pair('H',0)) + Hc.info.get_offset(bm,bm)-1;
            }
            GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*3*sizeof(size_t));
            onedot_diagGPU_local(nblk, ndim, dev_diag, dev_dims, 
                  lqops._dev_data, rqops._dev_data,
                  cqops._dev_data);
         }
         auto t3 = tools::get_time();

         // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
         //        B/Q^C
         //         |     
         // B/Q^L---*---B/Q^R
         onedot_diagGPU_BQ("lc", lqops, cqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         onedot_diagGPU_BQ("lr", lqops, rqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         onedot_diagGPU_BQ("cr", cqops, rqops, wf, dev_diag, dev_dims, opoffs, size, rank);
         auto t4 = tools::get_time();

         // debug
         if(diagcheck) onedot_diagGPU_check(ndim, dev_diag, qops_dict, wf, size, rank, ifdist1);

         GPUmem.deallocate(dev_dims, nblk*7*sizeof(size_t));
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
      void onedot_diagGPU_check(const int ndim,
            const double* dev_diag,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const qtensor3<ifab,Tm>& wf,
            const int size,
            const int rank,
            const bool ifdist1){
         double* diag1 = new double[ndim];
         GPUmem.to_cpu(diag1, dev_diag, ndim*sizeof(double));
         GPUmem.sync();
         double* diag2 = new double[ndim];
         onedot_diag(qops_dict, wf, diag2, size, rank, ifdist1);
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
      void onedot_diagGPU_BQ(const std::string superblock,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor3<Tm>& wf,
            double* dev_diag,
            size_t* dev_dims,
            std::vector<size_t>& opoffs,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const size_t csize1 = qops1.cindex.size();
         const size_t csize2 = qops2.cindex.size();
         const bool ifNC = determine_NCorCN_BQ(qops1.oplist, qops2.oplist, csize1, csize2);
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         if(rank == 0 && debug_onedot_diag){ 
            std::cout << "onedot_diagGPU_BQ superblock=" << superblock
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

            if(superblock == "lc"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm;
                  wf.info._addr_unpack(idx, br, bc, bm);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bm,bm)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 0, 2);

            }else if(superblock == "lr"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm;
                  wf.info._addr_unpack(idx, br, bc, bm);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 0, 1);

            }else if(superblock == "cr"){

               for(int i=0; i<nblk; i++){
                  int idx = wf.info._nnzaddr[i];
                  int br, bc, bm;
                  wf.info._addr_unpack(idx, br, bc, bm);
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, wt, 2, 1);

            } // endif

         } // index
      }

} // ctns

#endif

#endif
