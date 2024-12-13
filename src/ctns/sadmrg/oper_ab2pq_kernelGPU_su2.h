#if defined(GPU) && defined(NCCL)

#ifndef OPER_AB2PQ_KERNELGPU_SU2_H
#define OPER_AB2PQ_KERNELGPU_SU2_H

#include "oper_ab2pq_kernel_su2.h"

namespace ctns{

   // su2 case
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void oper_a2pGPU(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            const int alg_a2p){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         auto t_start = tools::get_time();
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(alg_a2p == 3);

         // Create a CUDA stream for asynchronous operations
         cudaStream_t stream;
         cudaStreamCreate(&stream);

         // loop over rank
         for(int iproc=0; iproc<size; iproc++){
            auto aindex_iproc = oper_index_opA_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb);
            if(aindex_iproc.size() == 0) continue;
            // broadcast {opCrs} for given sym from iproc
            qoper_dict<Qm::ifabelian,Tm> qops_tmp;
            qops_tmp.sorb = qops.sorb;
            qops_tmp.isym = qops.isym;
            qops_tmp.ifkr = qops.ifkr;
            qops_tmp.cindex = qops.cindex;
            qops_tmp.krest = qops.krest;
            qops_tmp.qbra = qops.qbra;
            qops_tmp.qket = qops.qket;
            qops_tmp.oplist = "M";
            qops_tmp.mpisize = size;
            qops_tmp.mpirank = iproc; // not rank
            qops_tmp.ifdist2 = true;
            qops_tmp.init();
            if(qops_tmp.size() == 0) continue;
            // allocate space
            qops_tmp.allocate_gpu();

            // convert opA to opA.H()
            if(iproc == rank){
               auto t0x = tools::get_time();
               for(int idx=0; idx<aindex_iproc.size(); idx++){
                  auto isr = aindex_iproc[idx];
                  // copy opA from gpu to cpu
                  size_t opA_offset = qops._offset.at(std::make_pair('A',isr));
                  const Tm* ptr_opA_gpu = qops._dev_data + opA_offset;
                  Tm* ptr_opA_cpu = qops._data + opA_offset;
                  size_t size = qops('A').at(isr).size()*sizeof(Tm);
                  CUDA_CHECK(cudaMemcpy(ptr_opA_cpu, ptr_opA_gpu, size, cudaMemcpyDeviceToHost));
                  // HermitianConjugate on CPU 
                  HermitianConjugate(qops('A').at(isr), qops_tmp('M')[isr], true);
                  // copy opM from cpu to gpu 
                  size_t opM_offset = qops_tmp._offset.at(std::make_pair('M',isr));
                  const Tm* ptr_opM_cpu = qops_tmp._data + opM_offset;
                  Tm* ptr_opM_gpu = qops_tmp._dev_data + opM_offset;
                  CUDA_CHECK(cudaMemcpyAsync(ptr_opM_gpu, ptr_opM_cpu, size, cudaMemcpyHostToDevice, stream));
               }
               // Wait for all asynchronous operations to finish
               cudaStreamSynchronize(stream);
               auto t1x = tools::get_time();
               tadjt += tools::get_duration(t1x-t0x);
            }
            // broadcast opA.H()
#ifndef SERIAL
            if(size > 1){
               auto t0x = tools::get_time();
               nccl_comm.broadcast(qops_tmp.ptr_ops_gpu('M'), qops_tmp.size_ops('M'), iproc);
               auto t1x = tools::get_time();
               double tbcast = tools::get_duration(t1x-t0x);
               tcomm += tbcast;
               if(rank == 0){
                  size_t data_size = qops_tmp.size_ops('M');
                  std::cout << " iproc=" << iproc << " rank=" << rank 
                     << " size(opA)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                     << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                     << std::endl;
               }
            }
#endif

            // construct opP from opA, if opP is exist on the current process
            if(qops2.num_ops('P') == 0) continue;
            auto t0z = tools::get_time();
            const auto& pmap = qops2.indexmap('P');
            // Ppq = xpqsr*Asr
            const auto& amap = qops_tmp.indexmap('M');
            for(const auto& pr : amap){
               const auto& symP = pr.first;
               const auto& aindex = pr.second;
               if(pmap.find(symP) == pmap.end()) continue;
               int ts = symP.ts();
               const auto& pindex = pmap.at(symP);
               size_t opsize = qops2('P').at(pindex[0]).size();
               if(opsize == 0) continue;
               // construct coefficient matrix
               int rows = aindex.size();
               int cols = pindex.size();
               auto cmat = get_A2Pmat_su2(aindex, pindex, ts, int2e);
               // copy it to gpu
               size_t gpumem_cmat = cmat.size()*sizeof(Tm);
               Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
               GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat); 
               // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
               const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
               const Tm* ptr_opM_gpu = qops_tmp._dev_data + qops_tmp._offset.at(std::make_pair('M',aindex[0]));
               Tm* ptr_opP_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('P',pindex[0]));
               linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                     ptr_opM_gpu, opsize, cmat_gpu, rows, beta,
                     ptr_opP_gpu, opsize);
               GPUmem.deallocate(cmat_gpu, gpumem_cmat);
            } // amap
            auto t1z = tools::get_time();
            tcomp += tools::get_duration(t1z-t0z);
         } // iproc

         // Clean up
         cudaStreamDestroy(stream);

         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2pGPU(su2) : " << t_tot << " S"
               << " T(adjt/bcast/comp/rest)=" << tadjt << "," << tcomm << "," 
               << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void oper_b2qGPU(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            const int alg_b2q){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         auto t_start = tools::get_time();
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(qops.ifhermi);
         assert(alg_b2q == 3);
        
         // Create a CUDA stream for asynchronous operations
         cudaStream_t stream;
         cudaStreamCreate(&stream);

         // loop over rank
         for(int iproc=0; iproc<size; iproc++){
            auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb); 
            if(bindex_iproc.size() == 0) continue;
            // broadcast {opBqr} for given sym from iproc
            qoper_dict<Qm::ifabelian,Tm> qops_tmp;
            qops_tmp.sorb = qops.sorb;
            qops_tmp.isym = qops.isym;
            qops_tmp.ifkr = qops.ifkr;
            qops_tmp.cindex = qops.cindex;
            qops_tmp.krest = qops.krest;
            qops_tmp.qbra = qops.qbra;
            qops_tmp.qket = qops.qket;
            qops_tmp.oplist = "BN"; 
            qops_tmp.mpisize = size;
            qops_tmp.mpirank = iproc; // not rank
            qops_tmp.ifdist2 = true;
            qops_tmp.init();
            if(qops_tmp.size() == 0) continue; 
            // allocate space
            qops_tmp.allocate_gpu();

            if(iproc == rank){
               auto t0x = tools::get_time();
               linalg::xcopy_gpu(qops.size_ops('B'), qops.ptr_ops_gpu('B'), qops_tmp.ptr_ops_gpu('B'));
               auto t1x = tools::get_time();
               tadjt += tools::get_duration(t1x-t0x);
            }
#ifndef SERIAL
            if(size > 1){
               auto t0x = tools::get_time();
               nccl_comm.broadcast(qops_tmp.ptr_ops_gpu('B'), qops_tmp.size_ops('B'), iproc);
               auto t1x = tools::get_time();
               double tbcast = tools::get_duration(t1x-t0x);
               tcomm += tbcast;
               if(rank == 0){
                  size_t data_size = qops_tmp.size_ops('B');
                  std::cout << " iproc=" << iproc << " rank=" << rank 
                     << " size(opB)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                     << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                     << std::endl;
               }
            }
#endif
            // convert opB to opB.H()
            auto t0y = tools::get_time();
            for(int idx=0; idx<bindex_iproc.size(); idx++){
               auto iqr = bindex_iproc[idx];
               // copy opB from gpu to cpu
               size_t opB_offset = qops._offset.at(std::make_pair('B',iqr));
               const Tm* ptr_opB_gpu = qops._dev_data + opB_offset;
               Tm* ptr_opB_cpu = qops._data + opB_offset;
               size_t size = qops('B').at(iqr).size()*sizeof(Tm);
               CUDA_CHECK(cudaMemcpy(ptr_opB_cpu, ptr_opB_gpu, size, cudaMemcpyDeviceToHost));
               // HermitianConjugate on CPU 
               HermitianConjugate(qops('B').at(iqr), qops_tmp('N')[iqr], true);
               // copy opM from cpu to gpu 
               size_t opN_offset = qops_tmp._offset.at(std::make_pair('N',iqr));
               const Tm* ptr_opN_cpu = qops_tmp._data + opN_offset;
               Tm* ptr_opN_gpu = qops_tmp._dev_data + opN_offset;
               CUDA_CHECK(cudaMemcpyAsync(ptr_opN_gpu, ptr_opN_cpu, size, cudaMemcpyHostToDevice, stream));
            }
            // Wait for all asynchronous operations to finish
            cudaStreamSynchronize(stream);
            auto t1y = tools::get_time();
            tadjt += tools::get_duration(t1y-t0y);

            // only perform calculation if opQ is exist on the current process
            if(qops2.num_ops('Q') == 0) continue; 
            auto t0z = tools::get_time();
            const auto& qmap = qops2.indexmap('Q');
            // Qps = wqr*vpqsr*Bqr
            const auto& bmap = qops_tmp.indexmap('B');
            for(const auto& pr : bmap){
               const auto& symQ = pr.first;
               const auto& bindex = pr.second;
               if(qmap.find(symQ) == qmap.end()) continue;
               int ts = symQ.ts();
               const auto& qindex = qmap.at(symQ);
               size_t opsize = qops2('Q').at(qindex[0]).size();
               if(opsize == 0) continue; 
               // construct coefficient matrix
               int rows = bindex.size();
               int cols = qindex.size();
               auto cmat = get_B2Qmat_su2(bindex, qindex, ts, int2e, false);
               // copy it to gpu
               size_t gpumem_cmat = cmat.size()*sizeof(Tm);
               Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
               GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat);
               // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
               const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
               const Tm* ptr_opB_gpu = qops_tmp._dev_data + qops_tmp._offset.at(std::make_pair('B',bindex[0]));
               Tm* ptr_opQ_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('Q',qindex[0]));
               linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                     ptr_opB_gpu, opsize, cmat_gpu, rows, beta,
                     ptr_opQ_gpu, opsize);
               GPUmem.deallocate(cmat_gpu, gpumem_cmat);
            } // bmap
            // Qps = (-1)^k*wqr*vprsq*Bqr.H()
            const auto& nmap = qops_tmp.indexmap('N');
            for(const auto& pr : nmap){
               const auto& symQ = pr.first;
               const auto& bindex = pr.second;
               if(qmap.find(symQ) == qmap.end()) continue;
               int ts = symQ.ts();
               const auto& qindex = qmap.at(symQ);
               size_t opsize = qops2('Q').at(qindex[0]).size();
               if(opsize == 0) continue;
               // construct coefficient matrix
               int rows = bindex.size();
               int cols = qindex.size();
               auto cmat = get_B2Qmat_su2(bindex, qindex, ts, int2e, true);
               // copy it to gpu
               size_t gpumem_cmat = cmat.size()*sizeof(Tm);
               Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
               GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat); 
               // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
               const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
               const Tm* ptr_opN_gpu = qops_tmp._dev_data + qops_tmp._offset.at(std::make_pair('N',bindex[0]));
               Tm* ptr_opQ_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('Q',qindex[0]));
               linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                     ptr_opN_gpu, opsize, cmat_gpu, rows, beta,
                     ptr_opQ_gpu, opsize);
               GPUmem.deallocate(cmat_gpu, gpumem_cmat);
            } // nmap
            auto t1z = tools::get_time();
            tcomp += tools::get_duration(t1z-t0z);
         } // iproc

         // Clean up
         cudaStreamDestroy(stream);
         
         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2qGPU(su2) : " << t_tot << " S"
               << " T(adjt/bcast/comp/rest)=" << tadjt << "," << tcomm << "," 
               << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif

#endif
