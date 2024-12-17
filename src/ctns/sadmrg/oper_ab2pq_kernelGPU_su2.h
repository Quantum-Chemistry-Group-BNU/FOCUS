#if defined(GPU) && defined(NCCL)

#ifndef OPER_AB2PQ_KERNELGPU_SU2_H
#define OPER_AB2PQ_KERNELGPU_SU2_H

#include "oper_ab2pq_kernel_su2.h"
#include "../gpu_kernel/batched_Hermitian_Conjugate.h"

namespace ctns{

   template <bool ifab, typename Tm>
      void batchedHermitianConjugateGPU(const qoper_dict<ifab,Tm>& qops1,
            const char type1,
            qoper_dict<ifab,Tm>& qops2,
            const char type2,
            const bool adjoint){
         // count the number of task
         size_t nblks = 0;
         for(const auto& pr : qops2(type2)){
            const auto& index = pr.first;
            const auto& qt2 = pr.second;
            nblks += qt2.info._nnzaddr.size();
         }
         // setup tasks
         std::vector<size_t> offs(nblks*2);
         std::vector<int> dims(nblks*2);
         std::vector<Tm> facs(nblks);
         size_t iblk = 0;
         for(const auto& pr : qops2(type2)){
            const auto& index = pr.first;
            const auto& qt2 = pr.second;
            const auto& qt1 = qops1(type1).at(index);
            for(int i=0; i<qt2.info._nnzaddr.size(); i++){
               auto key = qt2.info._nnzaddr[i];
               int br = std::get<0>(key);
               int bc = std::get<1>(key);
               size_t loff2 = qt2.info.get_offset(br,bc);
               assert(loff2 > 0);
               size_t goff2 = qops2._offset.at(std::make_pair(type2,index)) + loff2-1;
               size_t loff1 = qt1.info.get_offset(bc,br);
               size_t goff1 = qops1._offset.at(std::make_pair(type1,index)) + loff1-1;
               offs[2*iblk] = goff2;
               offs[2*iblk+1] = goff1;
               auto blk = qt2(br,bc);
               dims[2*iblk] = blk.dim0;
               dims[2*iblk+1] = blk.dim1;
               if(!adjoint){
                  facs[iblk] = 1.0;
               }else{
                  // <br||Tk_bar||bc> = (-1)^{k-jc+jr}sqrt{[jc]/[jr]}<bc||Tk||br>*
                  auto symr = qt2.info.qrow.get_sym(br);
                  auto symc = qt2.info.qcol.get_sym(bc);
                  int tsr = symr.ts();
                  int tsc = symc.ts();
                  int deltats = (qt2.info.sym.ts() + tsr - tsc);
                  assert(deltats%2 == 0);
                  Tm fac = (deltats/2)%2==0? 1.0 : -1.0;
                  fac *= std::sqrt((tsc+1.0)/(tsr+1.0));
                  facs[iblk] = fac;
               }
               iblk += 1;
            }
         }
         // allocate memory
         size_t* dev_offs = (size_t*)GPUmem.allocate(nblks*2*sizeof(size_t));
         GPUmem.to_gpu(dev_offs, offs.data(), nblks*2*sizeof(size_t));
         int* dev_dims = (int*)GPUmem.allocate(nblks*2*sizeof(int));
         GPUmem.to_gpu(dev_dims, dims.data(), nblks*2*sizeof(int));
         Tm* dev_facs = (Tm*)GPUmem.allocate(nblks*sizeof(Tm));
         GPUmem.to_gpu(dev_facs, facs.data(), nblks*sizeof(Tm));
         // invoke kernel
         batched_Hermitian_Conjugate(nblks, dev_offs, dev_dims, dev_facs, qops1._dev_data, qops2._dev_data);
         // deallocate
         GPUmem.deallocate(dev_offs, nblks*2*sizeof(size_t));
         GPUmem.deallocate(dev_dims, nblks*2*sizeof(int));
         GPUmem.deallocate(dev_facs, nblks*sizeof(Tm));
      }

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
         double tinit = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0, taccum = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(alg_a2p >= 3);

         // Create a CUDA stream for asynchronous operations
         cudaStream_t stream;
         cudaStreamCreate(&stream);

         // loop over rank
         icomb.world.barrier();
         for(int iproc=0; iproc<size; iproc++){
            auto aindex_iproc = oper_index_opA_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb);
            if(aindex_iproc.size() == 0) continue;
            // broadcast {opCrs} for given sym from iproc
            auto t0i = tools::get_time();
            qoper_dict<Qm::ifabelian,Tm> qops_tmp;
            qops_tmp.sorb = qops.sorb;
            qops_tmp.isym = qops.isym;
            qops_tmp.ifkr = qops.ifkr;
            qops_tmp.cindex = qops.cindex;
            qops_tmp.krest = qops.krest;
            qops_tmp.qbra = qops.qbra;
            qops_tmp.qket = qops.qket;
            qops_tmp.oplist = "AM";
            qops_tmp.mpisize = size;
            qops_tmp.mpirank = iproc; // not rank
            qops_tmp.ifdist2 = true;
            qops_tmp.init();
            if(qops_tmp.size() == 0) continue;
            // allocate space
            qops_tmp.allocate_gpu();
            icomb.world.barrier();
            auto t1i = tools::get_time();
            tinit += tools::get_duration(t1i-t0i);
            taccum += tools::get_duration(t1i-t0i);
            if(rank == 0){
               std::cout << "iproc=" << iproc << std::endl;
               qops_tmp.print("qops_tmp");
               std::cout << "   init qops_tmp: t=" << tools::get_duration(t1i-t0i)
                  << " tinit=" << tinit << " taccum=" << taccum << std::endl;
            }

            auto t0x = tools::get_time();
            if(iproc == rank){

               // Algorithm-1:
               if(alg_a2p == 3){
                  auto t0a = tools::get_time();
                  // copy opA from GPU to CPU
                  size_t size = qops_tmp.size_ops('A')*sizeof(Tm);
                  CUDA_CHECK(cudaMemcpy(qops_tmp.ptr_ops('A'), qops.ptr_ops_gpu('A'), size, cudaMemcpyDeviceToHost));
                  auto t0b = tools::get_time();
                  // opA to opA.H() on CPU
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<aindex_iproc.size(); idx++){
                     auto isr = aindex_iproc[idx];
                     HermitianConjugate(qops_tmp('A').at(isr), qops_tmp('M')[isr], true);
                  }
                  auto t0c = tools::get_time();
                  // copy opA.H() from CPU to GPU
                  CUDA_CHECK(cudaMemcpy(qops_tmp.ptr_ops_gpu('M'), qops_tmp.ptr_ops('M'), size, cudaMemcpyHostToDevice));
                  auto t0d = tools::get_time();
                  if(rank==iproc) std::cout << "   D2H,toH,H2D,tot=" 
                     << tools::get_duration(t0b-t0a) << ","
                        << tools::get_duration(t0c-t0b) << ","
                        << tools::get_duration(t0d-t0c) << ","
                        << tools::get_duration(t0d-t0a) << std::endl;
                  // Algorithm-2:
               }else if(alg_a2p == 4){
                  // opA to opA.H() on GPU
                  batchedHermitianConjugateGPU(qops, 'A', qops_tmp, 'M', true); 
               }else{
                  std::cout << "error: no such option in a2pGPU for alg_a2p =" << alg_a2p << std::endl;
                  exit(1);
               } // alg_a2p 

            } // rank
            icomb.world.barrier();
            auto t1x = tools::get_time();
            tadjt += tools::get_duration(t1x-t0x);
            taccum += tools::get_duration(t1x-t0x);
            if(rank == 0) std::cout << "   from opA to opA.H(): size=" << aindex_iproc.size()
               << " t=" <<  tools::get_duration(t1x-t0x)
                  << " tadjt=" << tadjt << " taccum=" << taccum << std::endl;

#ifndef SERIAL
            // broadcast opA.H()
            auto t0b = tools::get_time();
            if(size > 1){
               nccl_comm.broadcast(qops_tmp.ptr_ops_gpu('M'), qops_tmp.size_ops('M'), iproc);
            }
            icomb.world.barrier();
            auto t1b = tools::get_time();
            double tbcast = tools::get_duration(t1b-t0b);
            tcomm += tbcast;
            taccum += tbcast;
            if(rank == 0){
               size_t data_size = qops_tmp.size_ops('M');
               std::cout << "   bcast: size(opA)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                  << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                  << " tcomm=" << tcomm << " taccum=" << taccum
                  << std::endl;
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
            taccum += tools::get_duration(t1z-t0z);
            if(rank == 0) std::cout << "   compute opP from opA: t=" << tools::get_duration(t1z-t0z) 
               << " tcomp=" << tcomp << " taccum=" << taccum << std::endl;
         } // iproc

         // Clean up
         cudaStreamDestroy(stream);

         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2pGPU(su2) : " << t_tot << " S"
               << " T(init/adjt/bcast/comp/rest)=" << tinit << "," 
               << tadjt << "," << tcomm << "," << tcomp << "," << trest << " -----"  
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
         double tinit = 0.0, tcopy = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0, taccum = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(qops.ifhermi);
         assert(alg_b2q >= 3);

         // Create a CUDA stream for asynchronous operations
         cudaStream_t stream;
         cudaStreamCreate(&stream);

         // loop over rank
         icomb.world.barrier();
         for(int iproc=0; iproc<size; iproc++){
            auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb); 
            if(bindex_iproc.size() == 0) continue;
            // broadcast {opBqr} for given sym from iproc
            auto t0i = tools::get_time();
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
            icomb.world.barrier();
            auto t1i = tools::get_time();
            tinit += tools::get_duration(t1i-t0i);
            taccum += tools::get_duration(t1i-t0i);
            if(rank == 0){
               std::cout << "iproc=" << iproc << std::endl;
               qops_tmp.print("qops_tmp");
               std::cout << "   init qops_tmp: t=" << tools::get_duration(t1i-t0i)
                  << " tinit=" << tinit << " taccum=" << taccum << std::endl;
            } 

            auto t0x = tools::get_time();
            if(iproc == rank){
               linalg::xcopy_gpu(qops.size_ops('B'), qops.ptr_ops_gpu('B'), qops_tmp.ptr_ops_gpu('B'));
            }
            icomb.world.barrier();
            auto t1x = tools::get_time();
            tcopy += tools::get_duration(t1x-t0x);
            taccum += tools::get_duration(t1x-t0x);
            if(rank == 0) std::cout << "   copy opB to qops_tmps:"
               << " t=" <<  tools::get_duration(t1x-t0x)
                  << " tcopy=" << tcopy << " taccum=" << taccum << std::endl;

#ifndef SERIAL
            auto t0b = tools::get_time();
            if(size > 1){
               nccl_comm.broadcast(qops_tmp.ptr_ops_gpu('B'), qops_tmp.size_ops('B'), iproc);
            }
            icomb.world.barrier();
            auto t1b = tools::get_time();
            double tbcast = tools::get_duration(t1b-t0b);
            tcomm += tbcast;
            taccum += tbcast;
            if(rank == 0){
               size_t data_size = qops_tmp.size_ops('B');
               std::cout << "   bcast: size(opB)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                  << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                  << " tcomm=" << tcomm << " taccum=" << taccum
                  << std::endl;
            }
#endif
            /*
            // convert opB to opB.H()
            auto t0y = tools::get_time();
            for(int idx=0; idx<bindex_iproc.size(); idx++){
            auto iqr = bindex_iproc[idx];
            // copy opB from gpu to cpu
            size_t opB_offset = qops_tmp._offset.at(std::make_pair('B',iqr));
            const Tm* ptr_opB_gpu = qops_tmp._dev_data + opB_offset;
            Tm* ptr_opB_cpu = qops_tmp._data + opB_offset;
            size_t size = qops_tmp('B').at(iqr).size()*sizeof(Tm);
            CUDA_CHECK(cudaMemcpy(ptr_opB_cpu, ptr_opB_gpu, size, cudaMemcpyDeviceToHost));
            // HermitianConjugate on CPU 
            HermitianConjugate(qops_tmp('B').at(iqr), qops_tmp('N')[iqr], true);
            // copy opM from cpu to gpu 
            size_t opN_offset = qops_tmp._offset.at(std::make_pair('N',iqr));
            const Tm* ptr_opN_cpu = qops_tmp._data + opN_offset;
            Tm* ptr_opN_gpu = qops_tmp._dev_data + opN_offset;
            CUDA_CHECK(cudaMemcpyAsync(ptr_opN_gpu, ptr_opN_cpu, size, cudaMemcpyHostToDevice, stream));
            }
            // Wait for all asynchronous operations to finish
            cudaStreamSynchronize(stream);
            icomb.world.barrier();
            auto t1y = tools::get_time();
            tadjt += tools::get_duration(t1y-t0y);
            taccum += tools::get_duration(t1y-t0y);
            if(rank == 0) std::cout << "   from opB to opB.H(): size=" << bindex_iproc.size()
            << " t=" <<  tools::get_duration(t1y-t0y)
            << " tadjt=" << tadjt << " taccum=" << taccum << std::endl;
            */

            // Algorithm-1:
            auto t0y = tools::get_time();
            {
               auto t0a = tools::get_time();
               // copy opB from GPU to CPU
               size_t size = qops_tmp.size_ops('B')*sizeof(Tm);
               CUDA_CHECK(cudaMemcpy(qops_tmp.ptr_ops('B'), qops_tmp.ptr_ops_gpu('B'), size, cudaMemcpyDeviceToHost));
               auto t0b = tools::get_time();
               // HermitianConjugate on CPU 
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int idx=0; idx<bindex_iproc.size(); idx++){
                  auto iqr = bindex_iproc[idx];
                  HermitianConjugate(qops_tmp('B').at(iqr), qops_tmp('N')[iqr], true);
               }
               auto t0c = tools::get_time();
               // copy opM from cpu to gpu 
               CUDA_CHECK(cudaMemcpy(qops_tmp.ptr_ops_gpu('N'), qops_tmp.ptr_ops('N'), size, cudaMemcpyHostToDevice));
               auto t0d = tools::get_time();
               if(rank==0) std::cout << "   D2H,toH,H2D,tot=" 
                  << tools::get_duration(t0b-t0a) << ","
                     << tools::get_duration(t0c-t0b) << ","
                     << tools::get_duration(t0d-t0c) << ","
                     << tools::get_duration(t0d-t0a) << std::endl;
            }
            icomb.world.barrier();
            auto t1y = tools::get_time();
            tadjt += tools::get_duration(t1y-t0y);
            taccum += tools::get_duration(t1y-t0y);
            if(rank == 0) std::cout << "   from opB to opB.H(): size=" << bindex_iproc.size()
               << " t=" <<  tools::get_duration(t1y-t0y)
                  << " tadjt=" << tadjt << " taccum=" << taccum << std::endl;

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
            taccum += tools::get_duration(t1z-t0z);
            if(rank == 0) std::cout << "   compute opQ from opB: t=" << tools::get_duration(t1z-t0z) 
               << " tcomp=" << tcomp << " taccum=" << taccum << std::endl;
         } // iproc

         // Clean up
         cudaStreamDestroy(stream);

         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tcopy - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2qGPU(su2) : " << t_tot << " S"
               << " T(init/copy/adjt/bcast/comp/rest)=" << tinit << "," << tcopy << ","
               << tadjt << "," << tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif

#endif
