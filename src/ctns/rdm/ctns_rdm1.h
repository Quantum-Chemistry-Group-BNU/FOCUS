#ifndef CTNS_RDM1_H
#define CTNS_RDM1_H

namespace ctns{

   template <typename Qm, typename Tm>
      void rdm_compute(const int order,
            const comb<Qm,Tm>& icomb,
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
         const int alg_rdm = schd.ctns.alg_rdm;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "ctns::rdm_compute"
               << " order=" << order
               << " ifab=" << ifab
               << " alg_rdm=" << alg_rdm
               << " isite=" << isite
               << std::endl;
         }
         auto t0 = tools::get_time();

         display_patterns(patterns);


         const std::map<std::string,std::pair<char,bool>> str2optype = {
            {"",{'I',0}},
            {"+",{'C',0}},
            {"-",{'C',1}},
            {"++",{'A',0}},
            {"--",{'A',1}},
            {"+-",{'B',0}},
            {"-+",{'B',1}},
            // coper
            {"++-",{'T',0}},
            {"+--",{'W',0}},
            {"++--",{'U',0}}
         };
         
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         lqops.print("lqops");
         rqops.print("rqops");
         cqops.print("cqops");

         const auto& lop = lqops('I').at(0);
         const auto& cop = cqops('I').at(0);
         const auto& rop = rqops('I').at(0);
         auto ldiff = linalg::deviationFromIdentity(lop.to_matrix());
         auto cdiff = linalg::deviationFromIdentity(cop.to_matrix());
         auto rdiff = linalg::deviationFromIdentity(rop.to_matrix());
         std::cout << "ldiff,cdiff,rdiff=" << ldiff << ","
            << cdiff << "," << rdiff << "," << std::endl;
         assert(ldiff < 1.e-10);
         assert(cdiff < 1.e-10);
         assert(rdiff < 1.e-10);

         // compute RDMs by pattern
         for(int i=0; i<patterns.size(); i++){
            const auto& pattern = patterns[i];
            const auto& loptype = str2optype.at(pattern.left);
            const auto& coptype = str2optype.at(pattern.center);
            const auto& roptype = str2optype.at(pattern.right);
            const auto& lkey = loptype.first; 
            const auto& ldagger = loptype.second;
            const auto& ckey = coptype.first; 
            const auto& cdagger = coptype.second;
            const auto& rkey = roptype.first; 
            const auto& rdagger = roptype.second;
            std::cout << "i=" << i 
               << " pattern=" << pattern.to_string() 
               << " opkey=" << lkey << ldagger 
               << ":" << ckey << cdagger 
               << ":" << rkey << rdagger 
               << std::endl;
            const auto& lops = lqops(lkey);
            std::cout << "lops=" << lops.size() << std::endl;
            const auto& cops = cqops(ckey);
            std::cout << "cops=" << cops.size() << std::endl;
            const auto& rops = rqops(rkey);
            std::cout << "rops=" << rops.size() << std::endl;
            //const auto& lops = lqops.at("");

            if(alg_rdm == 0){

            
            }
            
            // assemble rdms
            // spin-recoupling
            // reorder indices to physical
         }

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::rdm_compute", t0, t1);
         }
      }

} // ctns

#endif
