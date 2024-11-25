#ifndef OPER_AB2PQ_KERNEL_H
#define OPER_AB2PQ_KERNEL_H

namespace ctns{

   // non-su2 case
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_a2p(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            const int alg_ab2pq){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2p does not support ifkr=true!");
         }else{
            if(alg_ab2pq == 0){

               // loop over all A
               auto aindex = oper_index_opA(qops.cindex, qops.ifkr, qops.isym);
               auto pindex = qops2.oper_index_op('P');
               for(const auto& isr : aindex){
                  auto iproc = distribute2('A',ifkr,size,isr,sorb);
                  auto sr = oper_unpack(isr);
                  int s = sr.first;
                  int r = sr.second;
                  // bcast A to all processors
                  stensor2<Tm> opCrs;               
                  if(iproc == rank){
                     auto t0 = tools::get_time();
                     auto optmp = qops('A').at(isr).H(); 
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
                  // loop over all opP indices via openmp
                  auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int i=0; i<pindex.size(); i++){
                     auto ipq = pindex[i]; 
                     auto pq = oper_unpack(ipq);
                     int p = pq.first;
                     int q = pq.second;
                     auto& opP = qops2('P')[ipq];
                     if(opCrs.info.sym != opP.info.sym) continue;
                     linalg::xaxpy(opP.size(), int2e.get(p,q,s,r), opCrs.data(), opP.data()); 
                  }
                  auto t1 = tools::get_time();
                  tcomp += tools::get_duration(t1-t0);
               } // isr

            }else if(alg_ab2pq == 1){

               // construct qmap for {opPpq} on current process
               const auto& pmap = qops2.get_qindexmap('P');
               // loop over rank
               for(int iproc=0; iproc<size; iproc++){
                  // broadcast {opCrs} for given sym from iproc
                  qoper_dict<Qm::ifabelian,Tm> qops_tmp;
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
                  qops_tmp.init(true);
                  if(iproc == rank){
                     auto t0 = tools::get_time();
                     auto aindex = qops.oper_index_op('A');
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                     for(int idx=0; idx<aindex.size(); idx++){
                        auto isr = aindex[idx];
                        auto optmp = qops('A').at(isr).H();
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
                     double tbcast = tools::get_duration(t1-t0);
                     tcomm += tbcast;
                     std::cout << "rank=" << rank 
                        << " size(opA.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                        << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                        << std::endl;
                  }
#endif
                  auto t0 = tools::get_time();
                  const auto& amap = qops_tmp.get_qindexmap('M');
                  for(const auto& pr : amap){
                     const auto& symP = pr.first;
                     const auto& aindex = pr.second;
                     if(pmap.find(symP) == pmap.end()) continue;
                     const auto& pindex = pmap.at(symP);
                     // construct coefficient matrix
                     size_t rows = aindex.size();
                     size_t cols = pindex.size();
                     linalg::matrix<Tm> coeff(rows,cols);
                     for(int icol=0; icol<cols; icol++){
                        int ipq = pindex[icol];
                        auto pq = oper_unpack(ipq);
                        int p = pq.first;
                        int q = pq.second;
                        for(int irow=0; irow<rows; irow++){
                           int isr = aindex[irow];
                           auto sr = oper_unpack(isr);
                           int s = sr.first;
                           int r = sr.second;
                           coeff(irow,icol) = int2e.get(p,q,s,r);
                        } // irow
                     } // icol
                     // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                     size_t opsize = qops2('P').at(pindex[0]).size();
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     const Tm* ptr_opM = qops_tmp('M').at(aindex[0]).data();
                     Tm* ptr_opP = qops2('P')[pindex[0]].data(); 
                     linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                           ptr_opM, opsize, coeff.data(), rows, beta,
                           ptr_opP, opsize);
                  } // amap
                  auto t1 = tools::get_time();
                  tcomp += tools::get_duration(t1-t0);
               } // iproc

            } // alg_ab2pq
            auto t_end = tools::get_time();
            if(rank == 0){
               double t_tot = tools::get_duration(t_end-t_start);
               double trest = t_tot - tadjt - tcomm - tcomp;
               std::cout << "----- TIMING FOR oper_a2p : " << t_tot << " S"
                  << " T(adjt/bcast/comp/rest)=" << tadjt << "," << tcomm << "," 
                  << tcomp << "," << trest << " -----"  
                  << std::endl;
            }
         } // ifkr
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_b2q(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            const int alg_ab2pq){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2p does not support ifkr=true!");
         }else{
            assert(qops.ifhermi);
            if(alg_ab2pq == 0){

               // loop over all B
               auto bindex = oper_index_opB(qops.cindex, qops.ifkr, qops.isym, qops.ifhermi);
               auto qindex = qops2.oper_index_op('Q');
               for(const auto& iqr : bindex){
                  auto iproc = distribute2('B',ifkr,size,iqr,sorb);
                  auto qr = oper_unpack(iqr);
                  int q = qr.first;
                  int r = qr.second;
                  // bcast B to all processors
                  stensor2<Tm> opBqr, opBrq;
                  if(iproc == rank){
                     auto t0 = tools::get_time();
                     auto optmp1 = qops('B').at(iqr);
                     auto optmp2 = qops('B').at(iqr).H();
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
                  // loop over all opQ indices via openmp
                  auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int i=0; i<qindex.size(); i++){
                     auto ips = qindex[i]; 
                     auto ps = oper_unpack(ips);
                     int p = ps.first;
                     int s = ps.second;
                     auto& opQ = qops2('Q')[ips];
                     if(opBqr.info.sym == opQ.info.sym){
                        linalg::xaxpy(opQ.size(), int2e.get(p,q,s,r), opBqr.data(), opQ.data());
                     }
                     if(opBrq.info.sym == opQ.info.sym and q != r){
                        linalg::xaxpy(opQ.size(), int2e.get(p,r,s,q), opBrq.data(), opQ.data());
                     }
                  }
                  auto t1 = tools::get_time();
                  tcomp += tools::get_duration(t1-t0);
               } // iqr

            }else if(alg_ab2pq == 1){

               // construct qmap for {opQps} on current process
               const auto& qmap = qops2.get_qindexmap('Q');
               // loop over rank
               for(int iproc=0; iproc<size; iproc++){

                  // broadcast {opBqr} for given sym from iproc
                  {
                     qoper_dict<Qm::ifabelian,Tm> qops_tmp;
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
                     qops_tmp.init(true);
                     if(iproc == rank){
                        auto t0 = tools::get_time();
                        auto bindex = qops.oper_index_op('B');
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                        for(int idx=0; idx<bindex.size(); idx++){
                           auto iqr = bindex[idx];
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
                        double tbcast = tools::get_duration(t1-t0);
                        tcomm += tbcast;
                        std::cout << "rank=" << rank 
                           << " size(opB.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                           << std::endl;
                     }
#endif
                     auto t0 = tools::get_time();
                     const auto& bmap = qops_tmp.get_qindexmap('B');
                     for(const auto& pr : bmap){
                        const auto& symQ = pr.first;
                        const auto& bindex = pr.second;
                        if(qmap.find(symQ) == qmap.end()) continue;
                        const auto& qindex = qmap.at(symQ);
                        // construct coefficient matrix
                        size_t rows = bindex.size();
                        size_t cols = qindex.size();
                        linalg::matrix<Tm> coeff(rows,cols);
                        for(int icol=0; icol<cols; icol++){
                           int ips = qindex[icol];
                           auto ps = oper_unpack(ips);
                           int p = ps.first;
                           int s = ps.second;
                           for(int irow=0; irow<rows; irow++){
                              int iqr = bindex[irow];
                              auto qr = oper_unpack(iqr);
                              int q = qr.first;
                              int r = qr.second;
                              double wqr = (q==r)? 0.5 : 1.0;
                              coeff(irow,icol) = wqr*int2e.get(p,q,s,r);
                           } // irow
                        } // icol
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        size_t opsize = qops2('Q').at(qindex[0]).size();
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('B').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, coeff.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1 = tools::get_time();
                     tcomp += tools::get_duration(t1-t0);
                  }

                  // broadcast {opBqr^H} for given sym from iproc
                  {
                     qoper_dict<Qm::ifabelian,Tm> qops_tmp;
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
                     qops_tmp.init(true);
                     if(iproc == rank){
                        auto t0 = tools::get_time();
                        auto bindex = qops.oper_index_op('B');
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                        for(int idx=0; idx<bindex.size(); idx++){
                           auto iqr = bindex[idx];
                           auto optmp = qops('B').at(iqr).H();
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
                        double tbcast = tools::get_duration(t1-t0);
                        tcomm += tbcast;
                        std::cout << "rank=" << rank 
                           << " size(opB.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                           << std::endl;
                     }
#endif
                     auto t0 = tools::get_time();
                     const auto& bmap = qops_tmp.get_qindexmap('N');
                     for(const auto& pr : bmap){
                        const auto& symQ = pr.first;
                        const auto& bindex = pr.second;
                        if(qmap.find(symQ) == qmap.end()) continue;
                        const auto& qindex = qmap.at(symQ);
                        // construct coefficient matrix
                        size_t rows = bindex.size();
                        size_t cols = qindex.size();
                        linalg::matrix<Tm> coeff(rows,cols);
                        for(int icol=0; icol<cols; icol++){
                           int ips = qindex[icol];
                           auto ps = oper_unpack(ips);
                           int p = ps.first;
                           int s = ps.second;
                           for(int irow=0; irow<rows; irow++){
                              int iqr = bindex[irow];
                              auto qr = oper_unpack(iqr);
                              int q = qr.first;
                              int r = qr.second;
                              double wqr = (q==r)? 0.5 : 1.0;
                              coeff(irow,icol) = wqr*int2e.get(p,r,s,q);
                           } // irow
                        } // icol
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        size_t opsize = qops2('Q').at(qindex[0]).size();
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('N').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, coeff.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1 = tools::get_time();
                     tcomp += tools::get_duration(t1-t0);
                  }

               } // iproc

            } // alg_ab2pq
            auto t_end = tools::get_time();
            if(rank == 0){
               double t_tot = tools::get_duration(t_end-t_start);
               double trest = t_tot - tadjt - tcomm - tcomp;
               std::cout << "----- TIMING FOR oper_b2q : " << t_tot << " S"
                  << " T(adjt/bcast/comp)=" << tadjt << "," << tcomm << "," 
                  << tcomp << "," << trest << " -----"  
                  << std::endl;
            }
         } // ifkr
      }

} // ctns

#endif
