#ifndef OPER_AB2PQ_KERNEL_H
#define OPER_AB2PQ_KERNEL_H

#include "oper_ab2pq_util.h"

namespace ctns{

   // non-su2 case
   template <typename Qm, typename Tm>
      void oper_a2p(const comb<Qm,Tm>& icomb,
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
            std::cout << "ctns::oper_a2p ifab=" << ifab << " ifkr=" << ifkr << std::endl;
         }
         if(ifkr and ifab){
            tools::exit("error: oper_a2p does not support ifkr=true and ifab=true [cNK] yet!");
         }
         double tinit = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();

         if(alg_a2p == 0){

            // loop over all A
            auto aindex = oper_index_opA(qops.cindex, qops.ifkr, qops.isym);
            auto pindex = qops2.oper_index_op('P');
            for(const auto& isr : aindex){
               auto iproc = distribute2('A',ifkr,size,isr,sorb);
               auto sr = oper_unpack(isr);
               int s = sr.first, ks = s/2;
               int r = sr.second, kr = r/2;
               // bcast A to all processors
               qtensor2<ifab,Tm> opCrs;               
               if(iproc == rank){
                  auto t0 = tools::get_time();
                  auto optmp = qops('A').at(isr).H(true); 
                  opCrs.init(optmp.info);
                  linalg::xcopy(optmp.size(), optmp.data(), opCrs.data());
                  auto t1 = tools::get_time();
                  tadjt += tools::get_duration(t1-t0);
               }
#ifndef SERIAL
               if(size > 1){
                  auto t0 = tools::get_time();
                  mpi_wrapper::broadcast(icomb.world, opCrs, iproc);
                  auto t1 = tools::get_time();
                  tcomm += tools::get_duration(t1-t0);
               }
#endif
               // ZL@2024/11/26: this can happen for small bond dimension,
               // where no symmetry sectors are connected by deltaN=-2
               if(opCrs.size() == 0) continue;
               // loop over all opP indices via openmp
               auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int i=0; i<pindex.size(); i++){
                  auto ipq = pindex[i]; 
                  auto pq = oper_unpack(ipq);
                  int p = pq.first, kp = p/2;
                  int q = pq.second, kq = q/2;
                  auto& opP = qops2('P')[ipq];
                  if(opCrs.info.sym != opP.info.sym) continue;
                  if(ifab){
                     linalg::xaxpy(opP.size(), int2e.get(p,q,s,r), opCrs.data(), opP.data()); 
                  }else{
                     int ts = opP.info.sym.ts();
                     Tm fac = get_xint2e_su2(int2e,ts,kp,kq,ks,kr);
                     linalg::xaxpy(opP.size(), fac, opCrs.data(), opP.data()); 
                  }
               }
               auto t1 = tools::get_time();
               tcomp += tools::get_duration(t1-t0);
            } // isr

         }else if(alg_a2p == 1){

            // loop over rank
            for(int iproc=0; iproc<size; iproc++){
               auto aindex_iproc = oper_index_opA_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb);
               if(aindex_iproc.size() == 0) continue;
               // broadcast {opCrs} for given sym from iproc
               qoper_dict<ifab,Tm> qops_tmp;
               qops_tmp.sorb = qops.sorb;
               qops_tmp.isym = qops.isym;
               qops_tmp.ifkr = qops.ifkr;
               qops_tmp.cindex = qops.cindex;
               qops_tmp.krest = qops.krest;
               qops_tmp.qbra = qops.qbra;
               qops_tmp.qket = qops.qket;
               qops_tmp.oplist = "M"; // ZL2024/11/25: just us M[sr] to store A[sr]^H
               qops_tmp.mpisize = size;
               qops_tmp.mpirank = iproc; // not rank
               qops_tmp.ifdist2 = true;
               qops_tmp.init();
               if(qops_tmp.size() == 0) continue;

               if(iproc == rank){
                  auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<aindex_iproc.size(); idx++){
                     auto isr = aindex_iproc[idx];
                     auto optmp = qops('A').at(isr).H(true);
                     auto& opAbar = qops_tmp('M')[isr]; // M[sr] = A[sr]^H
                     assert(optmp.size() == opAbar.size()); 
                     linalg::xcopy(optmp.size(), optmp.data(), opAbar.data());
                  }
                  auto t1 = tools::get_time();
                  tadjt += tools::get_duration(t1-t0);
               }
#ifndef SERIAL
               if(size > 1){
                  auto t0 = tools::get_time();
                  size_t data_size = qops_tmp.size();
                  mpi_wrapper::broadcast(icomb.world, qops_tmp._data, data_size, iproc);
                  auto t1 = tools::get_time();
                  double dtb = tools::get_duration(t1-t0);
                  tcomm += dtb;
                  if(rank == iproc){
                     std::cout << " iproc=" << iproc << " rank=" << rank 
                        << " size(opA.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                        << " T(bcast)=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                        << std::endl;
                  }
               }
#endif
               auto t0 = tools::get_time();
               // only perform calculation if opP is exist on the current process
               if(qops2.num_ops('P') > 0){
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
                     // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                     int rows = aindex.size();
                     int cols = pindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opM = qops_tmp('M').at(aindex[0]).data();
                     Tm* ptr_opP = qops2('P')[pindex[0]].data(); 
                     linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                           ptr_opM, opsize, cmat.data(), rows, beta,
                           ptr_opP, opsize);
                  } // amap
                  auto t1 = tools::get_time();
                  tcomp += tools::get_duration(t1-t0);
               } // p
            } // iproc

         }else if(alg_a2p == 2){

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
               qops_tmp.init();
               if(qops_tmp.size() == 0) continue;
               auto t1i = tools::get_time();
               double dti = tools::get_duration(t1i-t0i);
               tinit += dti;
               if(rank == iproc){
                  std::cout << "iproc=" << iproc << std::endl;
                  qops_tmp.print("qops_tmp");
                  std::cout << "   init qops_tmp: dt=" << dti << " tinit=" << tinit << std::endl;
               }

               // convert opA to opA.H()
               auto t0h = tools::get_time();
               if(iproc == rank){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<aindex_iproc.size(); idx++){
                     auto isr = aindex_iproc[idx];
                     HermitianConjugate(qops('A').at(isr), qops_tmp('M')[isr], true);
                  }
               }
               auto t1h = tools::get_time();
               double dth = tools::get_duration(t1h-t0h); 
               tadjt += dth;
               if(rank == iproc) std::cout << "   from opA to opA.H(): dt=" <<  dth
                  << " tadjt=" << tadjt << std::endl;

#ifndef SERIAL
               // broadcast opA.H()
               auto t0b = tools::get_time();
               if(size > 1){
                  mpi_wrapper::broadcast(icomb.world, qops_tmp.ptr_ops('M'), qops_tmp.size_ops('M'), iproc);
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
                     // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                     int rows = aindex.size();
                     int cols = pindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opM = qops_tmp('M').at(aindex[0]).data();
                     Tm* ptr_opP = qops2('P')[pindex[0]].data();
                     linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                           ptr_opM, opsize, cmat.data(), rows, beta,
                           ptr_opP, opsize);
                  } // amap
                  auto t1c = tools::get_time();
                  double dtc = tools::get_duration(t1c-t0c);
                  tcomp += dtc;
                  if(rank == iproc) std::cout << "   compute opP from opA: dt=" << dtc
                     << " tcomp=" << tcomp << std::endl;
               }
            } // iproc

         }else if(alg_a2p >= 3){
            std::cout << "error: alg_a2p=3 should be used with alg_renorm>10 and ifnccl=true!" << std::endl;
            exit(1);  
         } // alg_a2p 
         auto t_end = tools::get_time();

         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2p : " << t_tot << " S"
               << " T(init/adjt+bcast/comp/rest)=" << tinit << ","
               << tadjt+tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

   template <typename Qm, typename Tm>
      void oper_b2q(const comb<Qm,Tm>& icomb,
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
            std::cout << "ctns::oper_b2q ifab=" << ifab << " ifkr=" << ifkr << std::endl;
         }
         if(ifkr and ifab){
            tools::exit("error: oper_b2q does not support ifkr=true and ifab=true [cNK] yet!");
         }
         assert(qops.ifhermi);
         double tinit = 0.0, tcopy = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();

         if(alg_b2q == 0){

            // loop over all B
            auto bindex = oper_index_opB(qops.cindex, qops.ifkr, qops.isym, qops.ifhermi);
            auto qindex = qops2.oper_index_op('Q');
            for(const auto& iqr : bindex){
               auto iproc = distribute2('B',ifkr,size,iqr,sorb);
               auto qr = oper_unpack(iqr);
               int q = qr.first, kq = q/2;
               int r = qr.second, kr = r/2;
               // bcast B to all processors
               qtensor2<ifab,Tm> opBqr, opBrq;
               if(iproc == rank){
                  auto t0 = tools::get_time();
                  auto optmp1 = qops('B').at(iqr);
                  auto optmp2 = qops('B').at(iqr).H(true);
                  opBqr.init(optmp1.info);
                  opBrq.init(optmp2.info);
                  linalg::xcopy(optmp1.size(), optmp1.data(), opBqr.data());
                  linalg::xcopy(optmp2.size(), optmp2.data(), opBrq.data());
                  auto t1 = tools::get_time();
                  tadjt += tools::get_duration(t1-t0);
               }
#ifndef SERIAL
               if(size > 1){
                  auto t0 = tools::get_time();
                  mpi_wrapper::broadcast(icomb.world, opBqr, iproc);
                  mpi_wrapper::broadcast(icomb.world, opBrq, iproc);
                  auto t1 = tools::get_time();
                  tcomm += tools::get_duration(t1-t0);
               }
#endif
               if(opBqr.size() == 0) continue;
               // loop over all opQ indices via openmp
               auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int i=0; i<qindex.size(); i++){
                  auto ips = qindex[i]; 
                  auto ps = oper_unpack(ips);
                  int p = ps.first, kp = p/2;
                  int s = ps.second, ks = s/2;
                  auto& opQ = qops2('Q')[ips];
                  if(ifab){
                     if(opBqr.info.sym == opQ.info.sym){
                        linalg::xaxpy(opQ.size(), int2e.get(p,q,s,r), opBqr.data(), opQ.data());
                     }
                     if(opBrq.info.sym == opQ.info.sym and q != r){
                        linalg::xaxpy(opQ.size(), int2e.get(p,r,s,q), opBrq.data(), opQ.data());
                     }
                  }else{
                     int ts = opQ.info.sym.ts();
                     if(opBqr.info.sym == opQ.info.sym){
                        Tm fac = get_vint2e_su2(int2e,ts,kp,kq,ks,kr);
                        linalg::xaxpy(opQ.size(), fac, opBqr.data(), opQ.data());
                     }
                     if(opBrq.info.sym == opQ.info.sym and kq != kr){
                        Tm fac = get_vint2e_su2(int2e,ts,kp,kr,ks,kq); 
                        if(ts == 2) fac = -fac; // (-1)^k in my note for Qps^k
                        linalg::xaxpy(opQ.size(), fac, opBrq.data(), opQ.data());
                     }
                  }
               }
               auto t1 = tools::get_time();
               tcomp += tools::get_duration(t1-t0);
            } // iqr

         }else if(alg_b2q == 1){

            // loop over rank
            for(int iproc=0; iproc<size; iproc++){
               auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb);
               // broadcast {opBqr} for given sym from iproc
               if(bindex_iproc.size() > 0){
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
                  qops_tmp.init();
                  if(qops_tmp.size() == 0) continue;  
                  
                  if(iproc == rank){
                     auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                     for(int idx=0; idx<bindex_iproc.size(); idx++){
                        auto iqr = bindex_iproc[idx];
                        auto optmp = qops('B').at(iqr);
                        auto& opBqr = qops_tmp('B')[iqr]; 
                        assert(optmp.size() == opBqr.size()); 
                        linalg::xcopy(optmp.size(), optmp.data(), opBqr.data());
                     }
                     auto t1 = tools::get_time();
                     tadjt += tools::get_duration(t1-t0);
                  }
#ifndef SERIAL
                  if(size > 1){
                     auto t0 = tools::get_time();
                     size_t data_size = qops_tmp.size();
                     mpi_wrapper::broadcast(icomb.world, qops_tmp._data, data_size, iproc);
                     auto t1 = tools::get_time();
                     double dtb = tools::get_duration(t1-t0);
                     tcomm += dtb;
                     if(rank == iproc){
                        std::cout << " iproc=" << iproc << " rank=" << rank 
                           << " size(opB)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " T(bcast)=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                           << std::endl;
                     }
                  }
#endif
                  // only perform calculation if opQ is exist on the current process
                  if(qops2.num_ops('Q') > 0){
                     auto t0c = tools::get_time();
                     const auto& qmap = qops2.indexmap('Q');
                     const auto& bmap = qops_tmp.indexmap('B');
                     for(const auto& pr : bmap){
                        const auto& symQ = pr.first;
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
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        int rows = bindex.size();
                        int cols = qindex.size();
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('B').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, cmat.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1c = tools::get_time();
                     tcomp += tools::get_duration(t1c-t0c);
                  } // q
               } // b

               // broadcast {opBqr^H} for given sym from iproc
               if(bindex_iproc.size() > 0){
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
                  qops_tmp.init();
                  if(qops_tmp.size() == 0) continue;

                  if(iproc == rank){
                     auto t0 = tools::get_time();
                     auto bindex = qops.oper_index_op('B');
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                     for(int idx=0; idx<bindex.size(); idx++){
                        auto iqr = bindex[idx];
                        auto optmp = qops('B').at(iqr).H(true);
                        auto& opBqrH = qops_tmp('N')[iqr];
                        assert(optmp.size() == opBqrH.size()); 
                        linalg::xcopy(optmp.size(), optmp.data(), opBqrH.data());
                     }
                     auto t1 = tools::get_time();
                     tadjt += tools::get_duration(t1-t0);
                  }
#ifndef SERIAL
                  if(size > 1){
                     auto t0 = tools::get_time();
                     size_t data_size = qops_tmp.size();
                     mpi_wrapper::broadcast(icomb.world, qops_tmp._data, data_size, iproc);
                     auto t1 = tools::get_time();
                     double dtb = tools::get_duration(t1-t0);
                     tcomm += dtb;
                     if(rank == iproc){
                        std::cout << " iproc=" << iproc << " rank=" << rank 
                           << " size(opB.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " T(bcast)=" << dtb << " speed=" << tools::sizeGB<Tm>(data_size)/dtb << "GB/s"
                           << std::endl;
                     }
                  }
#endif
                  if(qops2.num_ops('Q') > 0){
                     auto t0c = tools::get_time();
                     const auto& qmap = qops2.indexmap('Q');
                     const auto& bmap = qops_tmp.indexmap('N');
                     for(const auto& pr : bmap){
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
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        int rows = bindex.size();
                        int cols = qindex.size();
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('N').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, cmat.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1c = tools::get_time();
                     tcomp += tools::get_duration(t1c-t0c);
                  } // q
               } // b

            } // iproc

         // keep both BN such that only one bcast is need 
         }else if(alg_b2q == 2){

            // loop over rank
            for(int iproc=0; iproc<size; iproc++){
               auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb); 
               if(bindex_iproc.size() == 0) continue;
               // broadcast {opBqr} for given sym from iproc
               auto t0i = tools::get_time();
               qoper_dict<ifab,Tm> qops_tmp;
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
               auto t1i = tools::get_time();
               double dti = tools::get_duration(t1i-t0i);
               tinit += dti;
               if(rank == iproc){
                  std::cout << "iproc=" << iproc << std::endl;
                  qops_tmp.print("qops_tmp");
                  std::cout << "   init qops_tmp: dt=" << dti << " tinit=" << tinit << std::endl;
               }

               // copy opB on CPU
               auto t0c = tools::get_time();
               if(iproc == rank){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<bindex_iproc.size(); idx++){
                     auto iqr = bindex_iproc[idx];
                     auto optmp = qops('B').at(iqr);
                     auto& opBqr = qops_tmp('B')[iqr]; 
                     assert(optmp.size() == opBqr.size()); 
                     linalg::xcopy(optmp.size(), optmp.data(), opBqr.data());
                  }
               }
               auto t1c = tools::get_time();
               double dtc = tools::get_duration(t1c-t0c);
               tcopy += dtc;
               if(rank == iproc) std::cout << "   copy opB to qops_tmps: dt=" << dtc 
                  << " tcopy=" << tcopy << std::endl;

#ifndef SERIAL
               // broadcast opB
               auto t0b = tools::get_time();
               if(size > 1){
                  mpi_wrapper::broadcast(icomb.world, qops_tmp.ptr_ops('B'), qops_tmp.size_ops('B'), iproc);
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

               // convert opB to opB.H()
               auto t0h = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int idx=0; idx<bindex_iproc.size(); idx++){
                  auto iqr = bindex_iproc[idx];
                  HermitianConjugate(qops_tmp('B').at(iqr), qops_tmp('N')[iqr], true);
               }
               auto t1h = tools::get_time();
               double dth = tools::get_duration(t1h-t0h);
               tadjt += dth; 
               if(rank == iproc) std::cout << "   from opB to opB.H(): dt=" << dth
                  << " tadjt=" << tadjt << std::endl;

               // only perform calculation if opQ is exist on the current process
               if(qops2.num_ops('Q') > 0){
                  auto t0c = tools::get_time();
                  const auto& qmap = qops2.indexmap('Q');
                  // Qps = wqr*vpqsr*Bqr
                  const auto& bmap = qops_tmp.indexmap('B');
                  for(const auto& pr : bmap){
                     const auto& symQ = pr.first;
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
                     // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                     int rows = bindex.size();
                     int cols = qindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opB = qops_tmp('B').at(bindex[0]).data();
                     Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                     linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                           ptr_opB, opsize, cmat.data(), rows, beta,
                           ptr_opQ, opsize);
                  } // bmap
                  // Qps = (-1)^k*wqr*vprsq*Bqr.H()
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
                     // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                     int rows = bindex.size();
                     int cols = qindex.size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opB = qops_tmp('N').at(bindex[0]).data();
                     Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                     linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                           ptr_opB, opsize, cmat.data(), rows, beta,
                           ptr_opQ, opsize);
                  } // nmap
                  auto t1c = tools::get_time();
                  dtc = tools::get_duration(t1c-t0c);
                  tcomp += dtc;
                  if(rank == iproc) std::cout << "   compute opQ from opB: dt=" << dtc
                     << " tcomp=" << tcomp << std::endl;
               }
            } // iproc

         }else if(alg_b2q >= 3){
            std::cout << "error: alg_b2q=3 should be used with alg_renorm>10 and ifnccl=true!" << std::endl;
            exit(1);  
         } // alg_b2q 
         auto t_end = tools::get_time();

         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tinit - tcopy - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2q : " << t_tot << " S"
               << " T(init/copy/adjt+bcast/comp/rest)=" << tinit << "," << tcopy << ","
               << tadjt+tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif
