#ifndef OPER_AB2PQ_H
#define OPER_AB2PQ_H

namespace ctns{

   // renormalize operators
   template <typename Qm, typename Tm>
      std::string oper_renorm_oplist(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const input::schedule& schd){
         std::string oplist = "CSH";
         if(!schd.ctns.ifab2pq){
            oplist += "ABPQ";
         }else{
            assert(icomb.topo.ifmps);
            int nsite = icomb.get_nphysical();
            int psite = pcoord.first;
            //  *---*---*---*---*---*
            // PQ  PQ  PQ  AB  AB  AB [backword:cr]
            // AB  AB  AB  PQ  PQ  PQ [forward:lc]
            bool ifAB = (superblock=="cr" and psite>=nsite/2) or
                        (superblock=="lc" and psite<nsite/2);
            oplist += (ifAB? "AB" : "PQ");
         }
         return oplist;
      }

   template <typename Qm, typename Tm>
      void oper_ab2pq(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const comb_coord& pcoord,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            qoper_pool<Qm::ifabelian,Tm>& qops_pool,
            const std::string& frop){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool debug = (rank == 0);
         if(debug and schd.ctns.verbose>0){
            std::cout << "ctns::oper_ab2pq coord=" << pcoord
               << " superblock=" << superblock
               << std::endl;
         }
         exit(1);
      }

} // ctns

#endif
