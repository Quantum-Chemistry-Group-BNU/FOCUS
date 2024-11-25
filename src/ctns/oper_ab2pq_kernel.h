#ifndef OPER_AB2PQ_KERNEL_H
#define OPER_AB2PQ_KERNEL_H

namespace ctns{

   // non-su2 case
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_a2p(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp,
            const int alg_ab2pq=1){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         tcomm = 0.0;
         tcomp = 0.0;
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
                     auto optmp = qops('A').at(isr).H(); 
                     opCrs.init(optmp.info);
                     linalg::xcopy(optmp.size(), optmp.data(), opCrs.data());
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
               qops2.print("qops2",2);
               const auto& pmap = qops2.get_qindexmap('P');
               for(const auto& pr : pmap){
                  std::cout << "pr.sym=" << pr.first << " size=" << pr.second.size() << std::endl;
                  tools::print_vector(pr.second,"index");
               }
               exit(1);

               tools::exit("not implemented yet");

            } // alg_ab2pq
         } // ifkr
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_b2q(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp,
            const int alg_ab2pq=0){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         tcomm = 0.0;
         tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2p does not support ifkr=true!");
         }else{
            if(alg_ab2pq == 0){
               assert(qops.ifhermi);
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
                     auto optmp1 = qops('B').at(iqr);
                     auto optmp2 = qops('B').at(iqr).H();
                     opBqr.init(optmp1.info);
                     opBrq.init(optmp2.info);
                     linalg::xcopy(optmp1.size(), optmp1.data(), opBqr.data());
                     linalg::xcopy(optmp2.size(), optmp2.data(), opBrq.data());
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

               tools::exit("not implemented yet");

            }
         } // ifkr
      }

} // ctns

#endif
