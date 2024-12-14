#ifndef OPER_AB2PQ_KERNEL_SU2_H
#define OPER_AB2PQ_KERNEL_SU2_H

#include "symbolic_compxwf_su2.h"

namespace ctns{

   template <typename Tm>
      linalg::matrix<Tm> get_A2Pmat_su2(const std::vector<int>& aindex, 
            const std::vector<int>& pindex, 
            const int ts, 
            const integral::two_body<Tm>& int2e){
         int rows = aindex.size();
         int cols = pindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ipq = pindex[icol];
            auto pq = oper_unpack(ipq);
            int p2 = pq.first, kp = p2/2;
            int q2 = pq.second, kq = q2/2;
            for(int irow=0; irow<rows; irow++){
               int isr = aindex[irow];
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               cmat(irow,icol) = get_xint2e_su2(int2e,ts,kp,kq,ks,kr);
            } // irow
         } // icol
         return cmat;
      } 

   template <typename Tm>
      linalg::matrix<Tm> get_B2Qmat_su2(const std::vector<int>& bindex, 
            const std::vector<int>& qindex, 
            const int ts, 
            const integral::two_body<Tm>& int2e,
            const bool swap_qr){
         int rows = bindex.size();
         int cols = qindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ips = qindex[icol];
            auto ps = oper_unpack(ips);
            int p2 = ps.first, kp = p2/2;
            int s2 = ps.second, ks = s2/2;
            for(int irow=0; irow<rows; irow++){
               int iqr = bindex[irow];
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq = q2/2;
               int r2 = qr.second, kr = r2/2;
               double wqr = (kq==kr)? 0.5 : 1.0;
               if(!swap_qr){
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kq,ks,kr);
               }else{
                  if(ts == 2) wqr = -wqr; // (-1)^k in my note for Qps^k
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kr,ks,kq);
               }
            } // irow
         } // icol
         return cmat;
      }

   // su2 case
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
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
         auto t_start = tools::get_time();
         double tinit = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0, taccum = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         if(alg_a2p == 0){

            // loop over all A
            auto aindex = oper_index_opA(qops.cindex, qops.ifkr, qops.isym);
            auto pindex = qops2.oper_index_op('P');
            for(const auto& isr : aindex){
               auto iproc = distribute2('A',ifkr,size,isr,sorb);
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               // bcast A to all processors
               stensor2su2<Tm> opCrs;
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
               if(opCrs.size() == 0) continue;
               // loop over all opP indices via openmp
               auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int i=0; i<pindex.size(); i++){
                  auto ipq = pindex[i]; 
                  auto pq = oper_unpack(ipq);
                  int p2 = pq.first, kp = p2/2;
                  int q2 = pq.second, kq = q2/2;
                  auto& opP = qops2('P')[ipq];
                  if(opCrs.info.sym != opP.info.sym) continue;
                  int ts = opP.info.sym.ts();
                  Tm fac = get_xint2e_su2(int2e,ts,kp,kq,ks,kr);
                  linalg::xaxpy(opP.size(), fac, opCrs.data(), opP.data()); 
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
               if(iproc == rank){
                  auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<aindex_iproc.size(); idx++){
                     auto isr = aindex_iproc[idx];
                     auto optmp = qops('A').at(isr).H(true);
                     auto& opAbar = qops_tmp('M')[isr];
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
                  if(rank == 0){
                     std::cout << " iproc=" << iproc << " rank=" << rank 
                        << " size(opA.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                        << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
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
                     int ts = symP.ts();
                     const auto& pindex = pmap.at(symP);
                     size_t opsize = qops2('P').at(pindex[0]).size();
                     if(opsize == 0) continue;
                     // construct coefficient matrix
                     int rows = aindex.size();
                     int cols = pindex.size();
                     auto cmat = get_A2Pmat_su2(aindex, pindex, ts, int2e);
                     // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                     const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                     // NOTE: ptr is not the starting ptr of 'M', because
                     // 'M' have several classes of different symmetries! 
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
               qops_tmp.oplist = "M";
               qops_tmp.mpisize = size;
               qops_tmp.mpirank = iproc; // not rank
               qops_tmp.ifdist2 = true;
               qops_tmp.init();
               if(qops_tmp.size() == 0) continue;
	       icomb.world.barrier();
	       auto t1i = tools::get_time();
	       tinit += tools::get_duration(t1i-t0i);
	       taccum += tools::get_duration(t1i-t0i);
	       if(rank == 0){
	          std::cout << "iproc=" << iproc << std::endl;
		  std::cout << "   init qops_tmp: t=" << tools::get_duration(t1i-t0i)
		      << " tinit=" << tinit << " taccum=" << taccum << std::endl;
	       }

               // convert opA to opA.H()
               auto t0x = tools::get_time();
               if(iproc == rank){
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                  for(int idx=0; idx<aindex_iproc.size(); idx++){
                     auto isr = aindex_iproc[idx];
                     HermitianConjugate(qops('A').at(isr), qops_tmp('M')[isr], true);
                  }
               }
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
                  mpi_wrapper::broadcast(icomb.world, qops_tmp.ptr_ops('M'), qops_tmp.size_ops('M'), iproc);
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
                  // contract opP(dat,pq) = opCrs(dat,rs)*x(rs,pq)
                  const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                  const Tm* ptr_opM = qops_tmp('M').at(aindex[0]).data();
                  Tm* ptr_opP = qops2('P')[pindex[0]].data();
                  linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                        ptr_opM, opsize, cmat.data(), rows, beta,
                        ptr_opP, opsize);
               } // amap
               auto t1z = tools::get_time();
               tcomp += tools::get_duration(t1z-t0z);
               taccum += tools::get_duration(t1z-t0z);
	       if(rank == 0) std::cout << "   compute opP from opA: t=" << tools::get_duration(t1z-t0z) 
	   	       << " tcomp=" << tcomp << " taccum=" << taccum << std::endl;
            } // iproc

         }else if(alg_a2p == 3){
            std::cout << "error: alg_a2p=3 should be used with alg_renorm>10 and ifnccl=true!" << std::endl;
            exit(1);  
         }else{
            std::cout << "error: no such option for alg_a2p=" << alg_a2p << std::endl;
            exit(1);
         } // alg_a2p 
         auto t_end = tools::get_time();

         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2p(su2) : " << t_tot << " S"
               << " T(init/adjt/bcast/comp/rest)=" << tinit << ","
	       << tadjt << "," << tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
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
         auto t_start = tools::get_time();
         double tinit = 0.0, tcopy = 0.0, tadjt = 0.0, tcomm = 0.0, tcomp = 0.0, taccum = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(qops.ifhermi);
         if(alg_b2q == 0){

            // loop over all B
            auto bindex = oper_index_opB(qops.cindex, qops.ifkr, qops.isym, qops.ifhermi);
            auto qindex = qops2.oper_index_op('Q');
            for(const auto& iqr : bindex){
               auto iproc = distribute2('B',ifkr,size,iqr,sorb);
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq = q2/2;
               int r2 = qr.second, kr = r2/2;
               // bcast B to all processors
               stensor2su2<Tm> opBqr, opBrq;
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
                  int p2 = ps.first, kp = p2/2;
                  int s2 = ps.second, ks = s2/2;
                  auto& opQ = qops2('Q')[ips];
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
               auto t1 = tools::get_time();
               tcomp += tools::get_duration(t1-t0);
            } // iqr

         }else if(alg_b2q == 1){

            // loop over rank
            for(int iproc=0; iproc<size; iproc++){
               auto bindex_iproc = oper_index_opB_dist(qops.cindex, qops.ifkr, qops.isym, size, iproc, qops.sorb); 
               // broadcast {opBqr} for given sym from iproc
               if(bindex_iproc.size() > 0){
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
                     double tbcast = tools::get_duration(t1-t0);
                     tcomm += tbcast;
                     if(rank == 0){
                        std::cout << " iproc=" << iproc << " rank=" << rank 
                           << " size(opB)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                           << std::endl;
                     }
                  }
#endif
                  auto t0 = tools::get_time();
                  // only perform calculation if opQ is exist on the current process
                  if(qops2.num_ops('Q') > 0){ 
                     const auto& qmap = qops2.indexmap('Q');
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
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('B').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, cmat.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1 = tools::get_time();
                     tcomp += tools::get_duration(t1-t0);
                  } // q
               } // b

               // broadcast {opBqr^H} for given sym from iproc
               if(bindex_iproc.size() > 0){
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
                  qops_tmp.init();
                  if(qops_tmp.size() == 0) continue;  
                  if(iproc == rank){
                     auto t0 = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                     for(int idx=0; idx<bindex_iproc.size(); idx++){
                        auto iqr = bindex_iproc[idx];
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
                     double tbcast = tools::get_duration(t1-t0);
                     tcomm += tbcast;
                     if(rank == 0){
                        std::cout << " iproc=" << iproc << " rank=" << rank 
                           << " size(opB.H)=" << data_size << ":" << tools::sizeGB<Tm>(data_size) << "GB" 
                           << " t(bcast)=" << tbcast << " speed=" << tools::sizeGB<Tm>(data_size)/tbcast << "GB/s"
                           << std::endl;
                     }
                  }
#endif
                  auto t0 = tools::get_time();
                  if(qops2.num_ops('Q') > 0){
                     const auto& qmap = qops2.indexmap('Q');
                     const auto& bmap = qops_tmp.indexmap('N');
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
                        auto cmat = get_B2Qmat_su2(bindex, qindex, ts, int2e, true);
                        // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                        const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                        const Tm* ptr_opB = qops_tmp('N').at(bindex[0]).data();
                        Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                        linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                              ptr_opB, opsize, cmat.data(), rows, beta,
                              ptr_opQ, opsize);
                     } // bmap
                     auto t1 = tools::get_time();
                     tcomp += tools::get_duration(t1-t0);
                  } // q
               } // b
            } // iproc

         }else if(alg_b2q == 2){

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
	       icomb.world.barrier();
	       auto t1i = tools::get_time();
	       tinit += tools::get_duration(t1i-t0i);
	       taccum += tools::get_duration(t1i-t0i);
	       if(rank == 0){
	          std::cout << "iproc=" << iproc << " init qops_tmp: t=" << tools::get_duration(t1i-t0i)
	              << " tinit=" << tinit << " taccum=" << taccum << std::endl;
	       } 

               auto t0x = tools::get_time();
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
                  mpi_wrapper::broadcast(icomb.world, qops_tmp.ptr_ops('B'), qops_tmp.size_ops('B'), iproc);
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

               // convert opB to opB.H()
               auto t0y = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int idx=0; idx<bindex_iproc.size(); idx++){
                  auto iqr = bindex_iproc[idx];
                  HermitianConjugate(qops_tmp('B').at(iqr), qops_tmp('N')[iqr], true);
               }
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
                  // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
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
                  int ts = symQ.ts();
                  const auto& qindex = qmap.at(symQ);
                  size_t opsize = qops2('Q').at(qindex[0]).size();
                  if(opsize == 0) continue;
                  // construct coefficient matrix
                  int rows = bindex.size();
                  int cols = qindex.size();
                  auto cmat = get_B2Qmat_su2(bindex, qindex, ts, int2e, true);
                  // contract opQ(dat,ps) = opBqr(dat,qr)*x(ps,qr)
                  const Tm alpha = 1.0, beta = 1.0; // accumulation from different processes
                  const Tm* ptr_opB = qops_tmp('N').at(bindex[0]).data();
                  Tm* ptr_opQ = qops2('Q')[qindex[0]].data(); 
                  linalg::xgemm("N", "N", opsize, cols, rows, alpha,
                        ptr_opB, opsize, cmat.data(), rows, beta,
                        ptr_opQ, opsize);
               } // nmap
               auto t1z = tools::get_time();
               tcomp += tools::get_duration(t1z-t0z);
               taccum += tools::get_duration(t1z-t0z);
   	       if(rank == 0) std::cout << "   compute opQ from opB: t=" << tools::get_duration(t1z-t0z) 
	   	       << " tcomp=" << tcomp << " taccum=" << taccum << std::endl;
            } // iproc

         }else if(alg_b2q == 3){
            std::cout << "error: alg_b2q=3 should be used with alg_renorm>10 and ifnccl=true!" << std::endl;
            exit(1);  
         }else{
            std::cout << "error: no such option for alg_b2q=" << alg_b2q << std::endl;
            exit(1);
         } // alg_b2q 
         auto t_end = tools::get_time();

         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2q(su2) : " << t_tot << " S"
               << " T(init/copy/adjt/bcast/comp/rest)=" << tinit << "," << tcopy << ","
	       << tadjt << "," << tcomm << "," << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif
