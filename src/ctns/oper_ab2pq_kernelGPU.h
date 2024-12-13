#if defined(GPU) && defined(NCCL)

#ifndef OPER_AB2PQ_KERNELGPU_H
#define OPER_AB2PQ_KERNELGPU_H

namespace ctns{

   // non-su2 case
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
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
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_a2pGPU does not support ifkr=true!");
         }
         assert(alg_a2p == 3);
         std::cout << "a2pGPU not implemented yet!" << std::endl;
         exit(1);
         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_a2pGPU : " << t_tot << " S"
               << " T(adjt/bcast/comp/rest)=" << tadjt << "," << tcomm << "," 
               << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
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
         double tadjt = 0.0, tcomm = 0.0, tcomp = 0.0;
         auto t_start = tools::get_time();
         const bool ifkr = Qm::ifkr;
         const int sorb = qops.sorb;
         if(ifkr){
            tools::exit("error: oper_b2qGPU does not support ifkr=true!");
         }
         assert(alg_b2q == 3);
         assert(qops.ifhermi);
         std::cout << "b2qGPU not implemented yet!" << std::endl;
         exit(1);
         auto t_end = tools::get_time();
         if(rank == 0){
            double t_tot = tools::get_duration(t_end-t_start);
            double trest = t_tot - tadjt - tcomm - tcomp;
            std::cout << "----- TIMING FOR oper_b2qGPU : " << t_tot << " S"
               << " T(adjt/bcast/comp)=" << tadjt << "," << tcomm << "," 
               << tcomp << "," << trest << " -----"  
               << std::endl;
         }
      }

} // ctns

#endif

#endif
