#ifndef OPER_AB2PQ_H
#define OPER_AB2PQ_H

#include "oper_ab2pq_kernel.h"
#include "sadmrg/oper_ab2pq_kernel_su2.h"
#ifdef GPU
#include "../gpu/gpu_blas.h"
#include "oper_ab2pq_kernelGPU.h"
#include "sadmrg/oper_ab2pq_kernelGPU_su2.h"
#endif

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
   inline std::string oper_renorm_oplist(const std::string superblock,
         const bool ifmps,
         const int nsite,
         const comb_coord& pcoord,
         const bool ifab2pq,
         const int ndots=2){
      std::string oplist = "CSH";
      if(!ifab2pq){
         oplist += "ABPQ";
      }else{
         assert(ifmps);
         int psite = pcoord.first;
         int pos = get_ab2pq_pos(nsite);
         bool ifAB = (superblock=="cr" and psite>=pos) or
            (superblock=="lc" and psite<=pos-ndots);
         oplist += (ifAB? "AB" : "PQ");
      }
      return oplist;
   }

   inline bool get_ab2pq_current(const std::string superblock,
         const bool ifmps,
         const int nsite,
         const comb_coord& pcoord,
         const bool ifab2pq,
         const int ndots){
      int psite = pcoord.first;
      int pos = get_ab2pq_pos(nsite);
      bool ab2pq = (superblock=="cr" and psite==pos) or // determine switch point
         (superblock=="lc" and psite==pos-ndots); // -2 for twodot case 
      return ifab2pq and ifmps and ab2pq;
   }

   template <typename Qm, typename Tm>
      void oper_ab2pq(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            qoper_dict<Qm::ifabelian,Tm>& qops){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool ifab2pq_gpunccl = schd.ctns.alg_renorm>10 and schd.ctns.ifnccl and
            schd.ctns.alg_a2p==3 and schd.ctns.alg_b2q==3; 
         const bool debug = (rank == 0);
         if(debug and schd.ctns.verbose>0){
            std::cout << "ctns::oper_ab2pq coord=" << pcoord
               << " superblock=" << superblock
               << " alg_renorm=" << alg_renorm
               << " alg_a2p=" << schd.ctns.alg_a2p
               << " alg_b2q=" << schd.ctns.alg_b2q
               << " ifab2pq_gpunccl=" << ifab2pq_gpunccl
               << std::endl;
         }
         auto t0 = tools::get_time();

         double tinit = 0.0, tcopy = 0.0;
         double ta2p = 0.0, tb2q = 0.0;
         double t2gpu = 0.0, t2cpu = 0.0, tmove = 0.0;

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
         qops2.init();
         if(debug){
            qops2.print("qops2", schd.ctns.verbose-1);
            get_mem_status(rank);
         }

         if(!ifab2pq_gpunccl){
         
            // 0. initialization
            memset(qops2._data, 0, qops2._size*sizeof(Tm));
            auto ta = tools::get_time();
            tinit = tools::get_duration(ta-t0);

            // 1. copy CSH
            std::string opseq = "CSH";
            for(const auto& key : opseq){
               linalg::xcopy(qops.size_ops(key), qops.ptr_ops(key), qops2.ptr_ops(key));
            }
            auto tb = tools::get_time();
            tcopy = tools::get_duration(tb-ta);

            // 2. transform A to P
            oper_a2p(icomb, int2e, qops, qops2, schd.ctns.alg_a2p);
            auto tc = tools::get_time();
            ta2p = tools::get_duration(tc-tb);

            // 3. transform B to Q
            oper_b2q(icomb, int2e, qops, qops2, schd.ctns.alg_b2q);
            auto td = tools::get_time();
            tb2q = tools::get_duration(td-tc);

#ifdef GPU
            // 4. to gpu (if necessary)
            if(alg_renorm > 10){
	       if(debug) get_mem_status(rank, 0, "before qops.clear_gpu");
	       qops.clear_gpu(); // deallocate qops on GPU
	       if(debug) get_mem_status(rank, 0, "before qops2.allocate_gpu");
	       qops2.allocate_gpu(); // allocate qops on GPU
	       if(debug) get_mem_status(rank, 0, "after qops2.allocate_gpu");
               qops2.to_gpu();
            }
#endif
            auto te = tools::get_time();
            t2gpu = tools::get_duration(te-td);

            // 5. move
	    if(debug) get_mem_status(rank, 0, "before move");
	    qops = std::move(qops2);
	    if(debug) get_mem_status(rank, 0, "after move");
            auto tf = tools::get_time();
            tmove = tools::get_duration(tf-te);

         }else{

#if defined(GPU) && defined(NCCL)
            // 0. initialization of qops on gpu
            qops2.allocate_gpu(true);
            auto ta = tools::get_time();
            tinit = tools::get_duration(ta-t0);
           
            // 1. copy CSH
            std::string opseq = "CSH";
            for(const auto& key : opseq){
               linalg::xcopy_gpu(qops.size_ops(key), qops.ptr_ops_gpu(key), qops2.ptr_ops_gpu(key));
            }
            auto tb = tools::get_time();
            tcopy = tools::get_duration(tb-ta);

            // 2. transform A to P
            oper_a2pGPU(icomb, int2e, qops, qops2, schd.ctns.alg_a2p);
            auto tc = tools::get_time();
            ta2p = tools::get_duration(tc-tb);

            // 3. transform B to Q
            oper_b2qGPU(icomb, int2e, qops, qops2, schd.ctns.alg_b2q);
            auto td = tools::get_time();
            tb2q = tools::get_duration(td-tc);
            
            // 4. to cpu
            qops2.to_cpu();
            auto te = tools::get_time();
            t2cpu = tools::get_duration(te-td);

            // 5. move
            qops = std::move(qops2);
            auto tf = tools::get_time();
            tmove = tools::get_duration(tf-te);
#else
            tools::exit("error: gpu-nccl version is not enabled!");
#endif
         } // ifab2pq_gpunccl

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_ab2pq : " << tools::get_duration(t1-t0) << " S"
               << " T(init/copyCSH/opP/opQ/to_gpu/to_cpu/move)=" << tinit << "," 
               << tcopy << "," << ta2p << "," << tb2q << "," << t2gpu << "," << t2cpu << "," << tmove << " -----"
               << std::endl;
	    get_mem_status(rank);
         }
      }

} // ctns

#endif
