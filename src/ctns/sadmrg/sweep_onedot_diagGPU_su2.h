#ifdef GPU

#ifndef SWEEP_ONEDOT_DIAGGPU_SU2_H
#define SWEEP_ONEDOT_DIAGGPU_SU2_H

#include "oper_dict.h"
#include "../gpu_kernel/onedot_diagGPU_kernel.h"
#include "../sweep_onedot_diag.h"
#include "sweep_onedot_diag_su2.h"

namespace ctns{

   template <typename Tm>
      void onedot_diagGPU(const opersu2_dictmap<Tm>& qops_dict,
            const stensor3su2<Tm>& wf,
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
            std::cout << "ctns::onedot_diagGPU(su2) ifkr=" << lqops.ifkr 
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

         size_t* dev_dims = (size_t*)GPUmem.allocate(nblk*7*sizeof(size_t));
         std::vector<size_t> blkdims(nblk*4,0);
         for(int i=0; i<nblk; i++){
            auto key = wf.info._nnzaddr[i];
            int br = std::get<0>(key);
            int bc = std::get<1>(key);
            int bm = std::get<2>(key);
            int tsi = std::get<3>(key);
            auto blk = wf(br,bc,bm,tsi);
            blkdims[4*i]   = blk.dim0;
            blkdims[4*i+1] = blk.dim1;
            blkdims[4*i+2] = blk.dim2;
            blkdims[4*i+3] = wf.info.get_offset(br,bc,bm,tsi)-1;
         }
         GPUmem.to_gpu(dev_dims, blkdims.data(), nblk*4*sizeof(size_t));
         auto t2 = tools::get_time();

         std::vector<size_t> opoffs(nblk*3,0);
         // 1. local terms: <lc1c2r|H|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
         // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
         // whose contributions to Diag need to be taken into aacount.
         if(!ifdist1 || rank == 0){
            const auto& Hl = lqops('H').at(0);
            const auto& Hr = rqops('H').at(0);
            const auto& Hc = cqops('H').at(0);
            for(int i=0; i<nblk; i++){
               auto key = wf.info._nnzaddr[i];
               int br = std::get<0>(key);
               int bc = std::get<1>(key);
               int bm = std::get<2>(key);
               int tsi = std::get<3>(key);
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
         //        B/Q^C1 B/Q^C2
         //         |      |
         // B/Q^L---*------*---B/Q^R
         std::vector<double> vec_fac(nblk);
         double* dev_fac = (double*)GPUmem.allocate(nblk*sizeof(double)); 
         onedot_diagGPU_BQ("lc", lqops, cqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         onedot_diagGPU_BQ("lr", lqops, rqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         onedot_diagGPU_BQ("cr", cqops, rqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         GPUmem.deallocate(dev_fac, nblk*sizeof(double));
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

   template <typename Tm>
      void onedot_diagGPU_BQ(const std::string superblock,
            const opersu2_dict<Tm>& qops1,
            const opersu2_dict<Tm>& qops2,
            const stensor3su2<Tm>& wf,
            double* dev_diag,
            size_t* dev_dims,
            std::vector<size_t>& opoffs,
            std::vector<double>& vec_fac,
            double* dev_fac,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         if(rank == 0 && debug_onedot_diag){ 
            std::cout << "onedot_diagGPU_BQ(su2) superblock=" << superblock
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
            assert(O1.info.sym.ne() == 0);
            // determine spin rank
            auto pq = oper_unpack(index);
            int p = pq.first, kp = p/2, sp = p%2;
            int q = pq.second, kq = q/2, sq = q%2;
            int ts = (sp!=sq)? 2 : 0;
            double fac = (kp==kq)? 0.5 : 1.0;
            double wt = ((ts==0)? 1.0 : -std::sqrt(3.0))*fac*2.0; // 2.0 from B^H*Q^H

            if(superblock == "lc"){

               int tsOl = ts;
               int tsOc = ts;
               int tsOr = 0;
               int tsO1 = 0, tsO2 = 0, tsOtot = 0;
               if(wf.info.couple == CRcouple){
                  // l|cr: Ol*(Oc*Ir)
                  tsO1 = ts; tsO2 = ts;
               }else{
                  // lc|r: (Ol*Oc)*Ir
                  tsO1 = 0; tsO2 = 0;
               }
               int br, bc, bm, tsi;
               for(int i=0; i<nblk; i++){
                  double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                        tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bm,bm)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 0, 2);

            }else if(superblock == "lr"){

               int tsOl = ts;
               int tsOc = 0;
               int tsOr = ts;
               int tsO1 = 0, tsO2 = 0, tsOtot = 0;
               if(wf.info.couple == CRcouple){
                  // l|cr: Ol*(Ic*Or)
                  tsO1 = ts; tsO2 = ts;
               }else{
                  // lc|r: (Ol*Ic)*Or
                  tsO1 = ts; tsO2 = ts;
               }
               int br, bc, bm, tsi;
               for(int i=0; i<nblk; i++){
                  double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                        tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 0, 1);

            }else if(superblock == "cr"){

               int tsOl = 0;
               int tsOc = ts;
               int tsOr = ts;
               int tsO1 = 0, tsO2 = 0, tsOtot = 0;
               if(wf.info.couple == CRcouple){
                  // l|cr: Il*(Oc*Or)
                  tsO1 = 0; tsO2 = 0;
               }else{
                  // lc|r: (Il*Oc)*Or
                  tsO1 = ts; tsO2 = ts;
               }
               int br, bc, bm, tsi;
               for(int i=0; i<nblk; i++){
                  double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                        tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*4], opoffs.data(), nblk*2*sizeof(size_t));
               onedot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 2, 1);

            } // endif

         } // index
      }

} // ctns

#endif

#endif
