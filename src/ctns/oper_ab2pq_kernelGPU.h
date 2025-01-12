#if defined(GPU) && defined(NCCL)

#ifndef OPER_AB2PQ_KERNELGPU_H
#define OPER_AB2PQ_KERNELGPU_H

#include "oper_ab2pq_util.h"

namespace ctns{

   template <typename Qm, typename Tm>
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
         const bool ifab = Qm::ifabelian;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(rank == 0){
            std::cout << "ctns::oper_a2p ifab=" << ifab << " ifkr=" << ifkr;
         }
         if(ifkr and ifab){
            tools::exit("error: oper_a2p does not support ifkr=true and ifab=true [cNK] yet!");
         }
         double tinit = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         assert(qops.ifhermi);
         assert(alg_a2p == 3);

         // determine the size of work space
         size_t maxsize = qops.size_ops('A');
#ifndef SERIAL
         if(size > 1){
            size_t local_maxsize = maxsize; // maxsize needs to be the same on all ranks
            boost::mpi::all_reduce(icomb.world, local_maxsize, maxsize, boost::mpi::maximum<size_t>());
         }
#endif
         if(rank == 0){
            std::cout << " maxsize=" << maxsize << ":" << tools::sizeGB<Tm>(maxsize) << "GB" << std::endl;
         }
         if(maxsize == 0) return;
         maxsize = maxsize*sizeof(Tm);
         Tm* dev_work = (Tm*)GPUmem.allocate(maxsize);

         // loop over rank
         for(int iproc=0; iproc<size; iproc++){
            auto aindex_iproc = oper_index_opA_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb);
            if(aindex_iproc.size() == 0) continue;
            // broadcast {opCrs} for given sym from iproc
            auto t0i = tools::get_time();
            qoper_dict<ifab,Tm> qops_tmp;
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
            qops_tmp.setup_opdict();
            if(qops_tmp.size() == 0) continue;
            auto t1i = tools::get_time();
            double dti = tools::get_duration(t1i-t0i);
            tinit += dti;
            if(rank == iproc){
               std::cout << "iproc=" << iproc << std::endl;
               qops_tmp.print("qops_tmp");
               std::cout << "   init qops_tmp: dt=" << dti << " tinit=" << tinit << std::endl;
            }

            // convert opA to opA.H() on GPU
            auto t0h = tools::get_time();
            if(iproc == rank){
               batchedHermitianConjugateGPU(qops, 'A', qops_tmp, 'M', true, qops._dev_data, dev_work);
               GPUmem.sync(); // ZL@2025/01/12: sync before broadcast
            } // rank
            auto t1h = tools::get_time();
            double dth = tools::get_duration(t1h-t0h);
            tadjt += dth;
            if(rank == iproc) std::cout << "   from opA to opA.H(): dt=" <<  dth 
               << " tadjt=" << tadjt << std::endl;

#ifndef SERIAL
            // broadcast opA.H()
            auto t0b = tools::get_time();
            if(size > 1){
               nccl_comm.broadcast(dev_work, qops_tmp.size_ops('M'), iproc);
            }
            auto t1b = tools::get_time();
            double dtb = tools::get_duration(t1b-t0b);
            tcomm += dtb;
            if(rank == iproc){
               size_t data_size = qops_tmp.size_ops('M');
               std::cout << "   bcast: size(opM)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                  << " dt=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                  << " tcomm=" << tcomm << std::endl;
            }
#endif

            // construct opP from opA, if opP is exist on the current process
            if(qops2.num_ops('P') > 0){
               auto t0c = tools::get_time();
               // Ppq = xpqsr*Asr
               const auto& pmap = qops2.indexmap('P');
               const auto& amap = qops_tmp.indexmap('M');
               for(const auto& pr : amap){
                  const auto& symP = pr.first;
                  const auto& aindex = pr.second;
                  if(pmap.find(symP) == pmap.end()) continue;
                  const auto& pindex = pmap.at(symP);
                  size_t opsize = qops2('P').at(pindex[0]).size();
                  if(opsize == 0) continue;
                  // construct coefficient matrix
                  linalg::matrix<Tm> cmat;
                  if(ifab){
                     cmat = get_A2Pmat(aindex, pindex, int2e);
                  }else{
                     int ts = symP.ts();
                     cmat = get_A2Pmat_su2(aindex, pindex, int2e, ts);
                  }
                  // copy it to gpu
                  size_t gpumem_cmat = cmat.size()*sizeof(Tm);
                  Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
                  GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat); 
                  // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                  int rows = aindex.size();
                  int cols = pindex.size();
                  const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                  const Tm* ptr_opM_gpu = dev_work + qops_tmp._offset.at(std::make_pair('M',aindex[0]));
                  Tm* ptr_opP_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('P',pindex[0]));
                  linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                        ptr_opM_gpu, opsize, cmat_gpu, rows, beta,
                        ptr_opP_gpu, opsize);
                  GPUmem.deallocate(cmat_gpu, gpumem_cmat);
               } // amap
               auto t1c = tools::get_time();
               double dtc = tools::get_duration(t1c-t0c);
               tcomp += dtc;
               if(rank == iproc) std::cout << "   compute opP from opA: dt=" << dtc
                  << " tcomp=" << tcomp << std::endl;
            } // p

         } // iproc

         // ZL@2024/12/31 synchronize
         GPUmem.sync();

         // Clean up
         GPUmem.deallocate(dev_work, maxsize);

         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2pGPU : " << t_tot << " S"
               << " T(init/adjt+bcast/comp/rest)=" << tinit << "," 
               << tadjt+tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

   template <typename Qm, typename Tm>
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
         const bool ifab = Qm::ifabelian;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(rank == 0){
            std::cout << "ctns::oper_b2q ifab=" << ifab << " ifkr=" << ifkr;
         }
         if(ifkr and ifab){
            tools::exit("error: oper_b2q does not support ifkr=true and ifab=true [cNK] yet!");
         }
         double tinit = 0.0, tcopy = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         assert(qops.ifhermi);
         assert(alg_b2q == 3);

         // determine the size of work space
         size_t maxsize = qops.size_ops('B');
#ifndef SERIAL
         if(size > 1){
            size_t local_maxsize = maxsize;
            boost::mpi::all_reduce(icomb.world, local_maxsize, maxsize, boost::mpi::maximum<size_t>());
         }
#endif
         if(rank == 0){
            std::cout << " maxsize=" << maxsize << ":" << tools::sizeGB<Tm>(maxsize) << "GB" << std::endl;
         }
         if(maxsize == 0) return;
         maxsize = maxsize*sizeof(Tm);
         Tm* dev_work = (Tm*)GPUmem.allocate(maxsize);

         // loop over rank
         for(int iproc=0; iproc<size; iproc++){
            auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb); 

            // broadcast {opBqr} for given sym from iproc
            if(bindex_iproc.size() > 0){
               auto t0i = tools::get_time();
               qoper_dict<ifab,Tm> qops_tmp;
               qops_tmp.sorb = qops.sorb;
               qops_tmp.isym = qops.isym;
               qops_tmp.ifkr = qops.ifkr;
               qops_tmp.cindex = qops.cindex;
               qops_tmp.krest = qops.krest;
               qops_tmp.qbra = qops.qbra;
               qops_tmp.qket = qops.qket;
               qops_tmp.oplist = "B"; 
               qops_tmp.mpisize = size;
               qops_tmp.mpirank = iproc; // not rank
               qops_tmp.ifdist2 = true;
               qops_tmp.setup_opdict();
               if(qops_tmp.size() == 0) continue; 
               auto t1i = tools::get_time();
               double dti = tools::get_duration(t1i-t0i);
               tinit += dti;
               if(rank == iproc){
                  std::cout << "iproc=" << iproc << std::endl;
                  qops_tmp.print("qops_tmp");
                  std::cout << "   init qops_tmp: dt=" << dti << " tinit=" << tinit << std::endl;
               }

               // copy opB on GPU
               auto t0c = tools::get_time();
               if(iproc == rank){
                  assert(qops.size_ops('B') == qops_tmp.size_ops('B'));
                  linalg::xcopy_gpu(qops_tmp.size_ops('B'), qops.ptr_ops_gpu('B'), dev_work);
                  GPUmem.sync(); // ZL@2025/01/12: sync before broadcast
               }
               auto t1c = tools::get_time();
               double dtc = tools::get_duration(t1c-t0c);
               tcopy += dtc;
               if(rank == iproc) std::cout << "   copy opB to qops_tmps: dt=" <<  dtc 
                  << " tcopy=" << tcopy << std::endl;

#ifndef SERIAL
               // broadcast opB
               auto t0b = tools::get_time();
               if(size > 1){
                  std::cout << "rank=" << rank << " size=" << qops_tmp.size_ops('B') << " iproc=" << iproc << std::endl;
                  nccl_comm.broadcast(dev_work, qops_tmp.size_ops('B'), iproc);
               }
               auto t1b = tools::get_time();
               double dtb = tools::get_duration(t1b-t0b);
               tcomm += dtb;
               if(rank == iproc){
                  size_t data_size = qops_tmp.size_ops('B');
                  std::cout << "   bcast: size(opB)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                     << " dt=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                     << " tcomm=" << tcomm << std::endl;
               }
#endif

               // only perform calculation if opQ is exist on the current process
               if(qops2.num_ops('Q') > 0){
                  auto t0c = tools::get_time();
                  // Qps = wqr*vpqsr*Bqr
                  const auto& qmap = qops2.indexmap('Q');
                  const auto& bmap = qops_tmp.indexmap('B');
                  for(const auto& pr : bmap){
                     const auto& symQ = pr.first; // = symB
                     const auto& bindex = pr.second;
                     if(qmap.find(symQ) == qmap.end()) continue;
                     const auto& qindex = qmap.at(symQ);
                     size_t opsize = qops2('Q').at(qindex[0]).size();
                     if(opsize == 0) continue; 
                     // construct coefficient matrix
                     linalg::matrix<Tm> cmat;
                     if(ifab){
                        cmat = get_B2Qmat(bindex, qindex, int2e, false);
                     }else{
                        int ts = symQ.ts();
                        cmat = get_B2Qmat_su2(bindex, qindex, int2e, ts, false);
                     }
                     // copy it to gpu
                     size_t gpumem_cmat = cmat.size()*sizeof(Tm);
                     Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
                     GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat);
                     // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                     int rows = bindex.size();
                     int cols = qindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opB_gpu = dev_work + qops_tmp._offset.at(std::make_pair('B',bindex[0]));
                     Tm* ptr_opQ_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('Q',qindex[0]));
                     linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                           ptr_opB_gpu, opsize, cmat_gpu, rows, beta,
                           ptr_opQ_gpu, opsize);
                     GPUmem.deallocate(cmat_gpu, gpumem_cmat);
                  } // bmap
                  auto t1c = tools::get_time();
                  double dtc = tools::get_duration(t1c-t0c);
                  tcomp += dtc;
                  if(rank == iproc) std::cout << "   compute opQ from opB: dt=" << dtc
                     << " tcomp=" << tcomp << std::endl;
               } // q
            } // b

            // broadcast {opBqr^H} for given sym from iproc
            if(bindex_iproc.size() > 0){
               auto t0i = tools::get_time();
               qoper_dict<ifab,Tm> qops_tmp;
               qops_tmp.sorb = qops.sorb;
               qops_tmp.isym = qops.isym;
               qops_tmp.ifkr = qops.ifkr;
               qops_tmp.cindex = qops.cindex;
               qops_tmp.krest = qops.krest;
               qops_tmp.qbra = qops.qbra;
               qops_tmp.qket = qops.qket;
               qops_tmp.oplist = "N"; 
               qops_tmp.mpisize = size;
               qops_tmp.mpirank = iproc; // not rank
               qops_tmp.ifdist2 = true;
               qops_tmp.setup_opdict();
               if(qops_tmp.size() == 0) continue;  
               auto t1i = tools::get_time();
               double dti = tools::get_duration(t1i-t0i);
               tinit += dti;
               if(rank == iproc){
                  std::cout << "iproc=" << iproc << std::endl;
                  qops_tmp.print("qops_tmp");
                  std::cout << "   init qops_tmp: dt=" << dti << " tinit=" << tinit << std::endl;
               }

               // opB to opB.H() on GPU
               auto t0h = tools::get_time();
               if(iproc == rank){
                  assert(qops.size_ops('B') == qops_tmp.size_ops('N'));
                  batchedHermitianConjugateGPU(qops, 'B', qops_tmp, 'N', true, qops._dev_data, dev_work); 
                  GPUmem.sync(); // ZL@2025/01/12: sync before broadcast
               } // rank
               auto t1h = tools::get_time();
               double dth = tools::get_duration(t1h-t0h);
               tadjt += dth;
               if(rank == iproc) std::cout << "   from opB to opB.H(): dt=" <<  dth 
                  << " tadjt=" << tadjt << std::endl;

#ifndef SERIAL
               // broadcast opB.H()
               auto t0b = tools::get_time();
               if(size > 1){
                  nccl_comm.broadcast(dev_work, qops_tmp.size_ops('N'), iproc);
               }
               auto t1b = tools::get_time();
               double dtb = tools::get_duration(t1b-t0b);
               tcomm += dtb;
               if(rank == iproc){
                  size_t data_size = qops_tmp.size_ops('N');
                  std::cout << "   bcast: size(opN)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                     << " dt=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                     << " tcomm=" << tcomm << std::endl;
               }
#endif

               // only perform calculation if opQ is exist on the current process
               if(qops2.num_ops('Q') > 0){
                  auto t0c = tools::get_time();
                  // Qps = (-1)^k*wqr*vprsq*Bqr.H()
                  const auto& qmap = qops2.indexmap('Q');
                  const auto& nmap = qops_tmp.indexmap('N');
                  for(const auto& pr : nmap){
                     const auto& symQ = pr.first;
                     const auto& bindex = pr.second;
                     if(qmap.find(symQ) == qmap.end()) continue;
                     const auto& qindex = qmap.at(symQ);
                     size_t opsize = qops2('Q').at(qindex[0]).size();
                     if(opsize == 0) continue;
                     // construct coefficient matrix
                     linalg::matrix<Tm> cmat;
                     if(ifab){
                        cmat = get_B2Qmat(bindex, qindex, int2e, true);
                     }else{
                        int ts = symQ.ts();
                        cmat = get_B2Qmat_su2(bindex, qindex, int2e, ts, true);
                     }
                     // copy it to gpu
                     size_t gpumem_cmat = cmat.size()*sizeof(Tm);
                     Tm* cmat_gpu = (Tm*)GPUmem.allocate(gpumem_cmat);
                     GPUmem.to_gpu(cmat_gpu, cmat.data(), gpumem_cmat); 
                     // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                     int rows = bindex.size();
                     int cols = qindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opN_gpu = dev_work + qops_tmp._offset.at(std::make_pair('N',bindex[0]));
                     Tm* ptr_opQ_gpu = qops2._dev_data + qops2._offset.at(std::make_pair('Q',qindex[0]));
                     linalg::xgemm_gpu("N", "N", opsize, cols, rows, alpha,
                           ptr_opN_gpu, opsize, cmat_gpu, rows, beta,
                           ptr_opQ_gpu, opsize);
                     GPUmem.deallocate(cmat_gpu, gpumem_cmat);
                  } // nmap
                  auto t1c = tools::get_time();
                  double dtc = tools::get_duration(t1c-t0c);
                  tcomp += dtc;
                  if(rank == iproc) std::cout << "   compute opQ from opB.H(): dt=" << dtc
                     << " tcomp=" << tcomp << std::endl;
               } // q
            } // b

         } // iproc
         
         // ZL@2024/12/31 synchronize
         GPUmem.sync();

         // Clean up
         GPUmem.deallocate(dev_work, maxsize);

         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tcopy - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2qGPU : " << t_tot << " S"
               << " T(init/copy/adjt+bcast/comp/rest)=" << tinit << "," << tcopy << ","
               << tadjt+tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif

#endif
