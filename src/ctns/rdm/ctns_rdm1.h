#ifndef CTNS_RDM1_H
#define CTNS_RDM1_H

namespace ctns{

   template <typename Qm, typename Tm>
      void get_rdm1(const comb<Qm,Tm>& icomb,
            const int isite,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<type_pattern>& patterns,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm1){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool ifab = Qm::ifabelian;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "\nctns::get_rdm1 ifab=" << ifab
               << std::endl;
         }
         auto t0 = tools::get_time();

         display_patterns(patterns);
         // assemble rdms
         // spin-recoupling
         // reorder indices to physical
         // Compute RDMs

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::get_rdm1", t0, t1);
         }
      }

} // ctns

#endif
