#ifndef OPER_AB2PQ_H
#define OPER_AB2PQ_H

#include "sadmrg/symbolic_compxwf_su2.h"

namespace ctns{

   // determine switch point from ab2pq
   /*
   (6+1)/2=3
      0   1   2   3   4   5
      *---*---*---X---*---*
     rPQ rPQ rPQ rPQ rAB rAB [backward]
     lAB lAB lPQ lPQ lPQ lPQ [forward]
   3-1=2
      0   1   2   3   4   5
      *---*---X---X---*---*
     rPQ rPQ rPQ rPQ rAB rAB
     lAB lPQ lPQ lPQ lPQ lPQ
   3-2=1
   (7+1)/2=4
      0   1   2   3   4   5   6
      *---*---*---X---*---*---*
     rPQ rPQ rPQ rPQ rPQ rAB rAB
     lAB lAB lAB lPQ lPQ lPQ lPQ
   4-1=3
      0   1   2   3   4   5   6
      *---*---X---X---*---*---*
     rPQ rPQ rPQ rPQ rPQ rAB rAB
     lAB lAB lPQ lPQ lPQ lPQ lPQ
   4-2=2
   */
   inline int get_ab2pq_pos(const int nsite){
      return (nsite+1)/2;
   }

   // determine which set of renormalize operators is to be used
   template <typename Qm, typename Tm>
      std::string oper_renorm_oplist(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const input::schedule& schd,
            const int ndots=2){
         std::string oplist = "CSH";
         if(!schd.ctns.ifab2pq){
            oplist += "ABPQ";
         }else{
            assert(icomb.topo.ifmps);
            int nsite = icomb.get_nphysical();
            int psite = pcoord.first;
            int pos = get_ab2pq_pos(nsite);
            bool ifAB = (superblock=="cr" and psite>=pos) or
                        (superblock=="lc" and psite<=pos-ndots);
            oplist += (ifAB? "AB" : "PQ");
         }
         return oplist;
      }

   // non-su2 case
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_a2p(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp){
         tcomm = 0.0;
         tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int size = qops.mpisize;
         const int rank = qops.mpirank;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2p does not support ifkr=true!");
         }else{
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
         } // ifkr
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void oper_b2q(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp){
         tcomm = 0.0;
         tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int size = qops.mpisize;
         const int rank = qops.mpirank;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2p does not support ifkr=true!");
         }else{
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
         } // ifkr
      }

   // su2 case
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void oper_a2p(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp){
         tcomm = 0.0;
         tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int size = qops.mpisize;
         const int rank = qops.mpirank;
         const int sorb = qops.sorb;
         assert(ifkr);
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
               auto optmp = qops('A').at(isr).H(true);
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
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void oper_b2q(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            qoper_dict<Qm::ifabelian,Tm>& qops2,
            double& tcomm,
            double& tcomp){
         tcomm = 0.0;
         tcomp = 0.0;
         const bool ifkr = Qm::ifkr;
         const int size = qops.mpisize;
         const int rank = qops.mpirank;
         const int sorb = qops.sorb;
         assert(ifkr);
         assert(qops.ifhermi);
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
               auto optmp1 = qops('B').at(iqr);
               auto optmp2 = qops('B').at(iqr).H(true);
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
      }

   template <typename Qm, typename Tm>
      void oper_ab2pq(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int ndots=2){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         assert(icomb.topo.ifmps);
         int nsite = icomb.get_nphysical();
         int psite = pcoord.first;
         int pos = get_ab2pq_pos(nsite);
         bool ab2pq = (superblock=="cr" and psite==pos) or // determine switch point
                      (superblock=="lc" and psite==pos-ndots); // -2 for twodot case 
         int alg_renorm = schd.ctns.alg_renorm;
         const bool debug = (rank == 0);
         if(debug and schd.ctns.verbose>0){
            std::cout << "ctns::oper_ab2pq coord=" << pcoord
               << " superblock=" << superblock
               << " ab2pq=" << ab2pq
               << " alg_renorm=" << alg_renorm
               << std::endl;
         }
         if(!ab2pq) return;
         auto t0 = tools::get_time();

         // 0. initialization: for simplicity, we perform the transformation on CPU
         qoper_dict<Qm::ifabelian,Tm> qops2;
         qops2.sorb = qops.sorb;
         qops2.isym = qops.isym;
         qops2.ifkr = qops.ifkr;
         qops2.cindex = qops.cindex;
         qops2.krest = qops.krest;
         qops2.qbra = qops.qbra;
         qops2.qket = qops.qket;
         qops2.oplist = "CSHPQ";
         qops2.mpisize = size;
         qops2.mpirank = rank;
         qops2.ifdist2 = true;
         qops2.init(true);
         auto ta = tools::get_time();

         // 1. copy CSH
         for(const auto key : "CSH"){
            size_t totsize = 0, offset = 0, idx = 0;
            for(int p : qops.oper_index_op(key)){
               totsize += qops(key).at(p).size();
               if(idx == 0) offset = qops._offset.at(std::make_pair(key,p));
               idx += 1; 
            }
            linalg::xcopy(totsize, qops._data+offset, qops2._data+offset);
         }
         auto tb = tools::get_time();

         // 2. transform A to P
         double tp_comm, tp_comp;
         oper_a2p(icomb, int2e, qops, qops2, tp_comm, tp_comp);
         auto tc = tools::get_time();

         // 3. transform B to Q
         double tq_comm, tq_comp;
         oper_b2q(icomb, int2e, qops, qops2, tq_comm, tq_comp);
         auto td = tools::get_time();

         // 4. to gpu (if necessary)
#ifdef GPU
         if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){
            // deallocate qops on GPU
            qops.clear_gpu();
            // allocate qops on GPU
            qops2.allocate_gpu();
            qops2.to_gpu();
         }
#endif
         auto te = tools::get_time();

         // 5. move
         qops = std::move(qops2);

         if(debug){
            auto t1 = tools::get_time();
            double tinit = tools::get_duration(ta-t0);
            double tcomm = tools::get_duration(tb-ta);
            double tp = tools::get_duration(tc-tb);
            double tq = tools::get_duration(td-tc);
            double tgpu = tools::get_duration(te-td);
            double tmove = tools::get_duration(t1-te);
            std::cout << "----- TIMING FOR oper_ab2pq : " << tools::get_duration(t1-t0) << " S"
               << " T(init/copyCSH/opP/opQ/to_gpu/move)=" << tinit << "," 
               << tcomm << "," << tp << "," << tq << "," << tgpu << "," << tmove << " -----"
               << std::endl;
            double tp_rest = tp - tp_comm - tp_comp;
            std::cout << "tp[tot,comm,comp,rest]=" << tp << ","
               << tp_comm << "," << tp_comp << tp_rest 
               << std::endl;
            double tq_rest = tq - tq_comm - tq_comp;
            std::cout << "tq[tot,comm,comp,rest]=" << tq << ","
               << tq_comm << "," << tq_comp << tq_rest
               << std::endl;
         }
      }

} // ctns

#endif
