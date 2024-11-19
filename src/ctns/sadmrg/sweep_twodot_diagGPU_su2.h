#ifdef GPU

#ifndef SWEEP_TWODOT_DIAGGPU_SU2_H
#define SWEEP_TWODOT_DIAGGPU_SU2_H

#include "oper_dict.h"
#include "../gpu_kernel/twodot_diagGPU_kernel.h"
#include "../sweep_twodot_diag.h"
#include "sweep_twodot_diag_su2.h"

namespace ctns{

   template <typename Tm>
      void twodot_diagGPU(const opersu2_dictmap<Tm>& qops_dict,
            const stensor4su2<Tm>& wf,
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
            std::cout << "ctns::twodot_diagGPU(su2) ifkr=" << lqops.ifkr 
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

         size_t* dev_dims = (size_t*)GPUmem.allocate(nblk*9*sizeof(size_t));
         std::vector<size_t> blkdims(nblk*5,0);
         for(int i=0; i<nblk; i++){
            auto key = wf.info._nnzaddr[i];
            int br = std::get<0>(key);
            int bc = std::get<1>(key);
            int bm = std::get<2>(key);
            int bv = std::get<3>(key);
            int tslc1 = std::get<4>(key);
            int tsc2r = std::get<5>(key);
            auto blk = wf(br,bc,bm,bv,tslc1,tsc2r);
            blkdims[5*i]   = blk.dim0;
            blkdims[5*i+1] = blk.dim1;
            blkdims[5*i+2] = blk.dim2;
            blkdims[5*i+3] = blk.dim3;
            blkdims[5*i+4] = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
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
               auto key = wf.info._nnzaddr[i];
               int br = std::get<0>(key);
               int bc = std::get<1>(key);
               int bm = std::get<2>(key);
               int bv = std::get<3>(key);
               int tslc1 = std::get<4>(key);
               int tsc2r = std::get<5>(key);
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
         std::vector<double> vec_fac(nblk);
         double* dev_fac = (double*)GPUmem.allocate(nblk*sizeof(double)); 
         twodot_diagGPU_BQ("lc1" ,  lqops, c1qops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         twodot_diagGPU_BQ("lc2" ,  lqops, c2qops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         twodot_diagGPU_BQ("lr"  ,  lqops,  rqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         twodot_diagGPU_BQ("c1c2", c1qops, c2qops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         twodot_diagGPU_BQ("c1r" , c1qops,  rqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         twodot_diagGPU_BQ("c2r" , c2qops,  rqops, wf, dev_diag, dev_dims, opoffs, vec_fac, dev_fac, size, rank);
         GPUmem.deallocate(dev_fac, nblk*sizeof(double));
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

   template <typename Tm>
      void twodot_diagGPU_BQ(const std::string superblock,
            const opersu2_dict<Tm>& qops1,
            const opersu2_dict<Tm>& qops2,
            const stensor4su2<Tm>& wf,
            double* dev_diag,
            size_t* dev_dims,
            std::vector<size_t>& opoffs,
            std::vector<double>& vec_fac,
            double* dev_fac,
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
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diagGPU_BQ(su2) superblock=" << superblock
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

            if(superblock == "lc1"){

               int tsOl = ts;
               int tsOc1 = ts;
               int tsOlc1 = 0;
               int tsOc2 = 0;
               int tsOr = 0;
               int tsOc2r = 0;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bm,bm)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 0, 2);

            }else if(superblock == "lc2"){

               int tsOl = ts;
               int tsOc1 = 0;
               int tsOlc1 = ts;
               int tsOc2 = ts;
               int tsOr = 0;
               int tsOc2r = ts;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bv,bv)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 0, 3);

            }else if(superblock == "lr"){

               int tsOl = ts;
               int tsOc1 = 0;
               int tsOlc1 = ts;
               int tsOc2 = 0;
               int tsOr = ts;
               int tsOc2r = ts;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(br,br)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 0, 1);

            }else if(superblock == "c1c2"){

               int tsOl = 0;
               int tsOc1 = ts;
               int tsOlc1 = ts;
               int tsOc2 = ts;
               int tsOr = 0;
               int tsOc2r = ts;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bv,bv)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 2, 3);

            }else if(superblock == "c1r"){

               int tsOl = 0;
               int tsOc1 = ts;
               int tsOlc1 = ts;
               int tsOc2 = 0;
               int tsOr = ts;
               int tsOc2r = ts;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bm,bm)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 2, 1);

            }else if(superblock == "c2r"){

               int tsOl = 0;
               int tsOc1 = 0;
               int tsOlc1 = 0;
               int tsOc2 = ts;
               int tsOr = ts;
               int tsOc2r = 0;
               int tsOtot = 0;
               int br, bc, bm, bv, tslc1, tsc2r;
               for(int i=0; i<nblk; i++){
                  double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                        tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
                  vec_fac[i] = fac;
                  if(std::abs(fac) < thresh_diag_angular) continue;
                  opoffs[2*i]   = qops1._offset.at(std::make_pair(BQ1,index)) + O1.info.get_offset(bv,bv)-1;
                  opoffs[2*i+1] = qops2._offset.at(std::make_pair(BQ2,index)) + O2.info.get_offset(bc,bc)-1;
               }
               GPUmem.to_gpu(dev_fac, vec_fac.data(), nblk*sizeof(double));
               GPUmem.to_gpu(&dev_dims[nblk*5], opoffs.data(), nblk*2*sizeof(size_t));
               twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims, qops1._dev_data, qops2._dev_data, dev_fac, 3, 1);

            } // endif

         } // index
      }

} // ctns

#endif

#endif
