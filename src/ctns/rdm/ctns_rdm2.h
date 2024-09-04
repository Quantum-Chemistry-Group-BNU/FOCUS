#ifndef CTNS_RDM2_H
#define CTNS_RDM2_H

namespace ctns{

   template <typename Qm, typename Tm>
      void get_rdm2(const bool is_same,
            const comb<Qm,Tm>& icomb,
            const comb<Qm,Tm>& icomb2,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm2){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool ifab = Qm::ifabelian;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "\nctns::get_rdm2 ifab=" << ifab
               << std::endl;
         }
         auto t0 = tools::get_time();


         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::get_rdm2", t0, t1);
         }
      }

} // ctns

#endif
