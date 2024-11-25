#ifndef OPER_AB2PQ_H
#define OPER_AB2PQ_H

#include "oper_ab2pq_kernel.h"
#include "sadmrg/oper_ab2pq_kernel_su2.h"

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
            std::cout << "tp[tot]=" << tp << " t[comm,comp,rest]="
               << tp_comm << "," << tp_comp << "," << tp_rest 
               << std::endl;
            double tq_rest = tq - tq_comm - tq_comp;
            std::cout << "tq[tot]=" << tq << " t[comm,comp,rest]="
               << tq_comm << "," << tq_comp << "," << tq_rest
               << std::endl;
         }
      }

} // ctns

#endif
