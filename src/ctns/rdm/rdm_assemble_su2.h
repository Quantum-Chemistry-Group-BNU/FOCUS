#ifndef RDM_ASSEMBLE_SU2_H
#define RDM_ASSEMBLE_SU2_H

namespace ctns{

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rdm_assemble(const int order,
            const bool is_same,
            const comb<Qm,Tm>& icomb,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<type_pattern>& allpatterns,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm,
            const linalg::matrix<Tm>& tdm){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool ifab = Qm::ifabelian;
         const int alg_rdm = schd.ctns.alg_rdm;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "ctns::rdm_assemble(su2)"
               << " order=" << order
               << " ifab=" << ifab
               << " alg_rdm=" << alg_rdm
               << std::endl;
         }
         auto t0 = tools::get_time();

         exit(1);

/*
         const auto& str2optype = is_same? str2optype_same : str2optype_diff;
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         // Check identity operator
         if(is_same){
            const auto& lop = lqops('I').at(0);
            const auto& cop = cqops('I').at(0);
            const auto& rop = rqops('I').at(0);
            auto ldiff = linalg::check_identityMatrix(lop.to_matrix());
            auto cdiff = linalg::check_identityMatrix(cop.to_matrix());
            auto rdiff = linalg::check_identityMatrix(rop.to_matrix());
            if(debug){
               std::cout << "rank=" << rank << " ldiff,cdiff,rdiff=" 
                  << ldiff << "," << cdiff << "," << rdiff << std::endl;
            }
            assert(ldiff < 1.e-8);
            assert(cdiff < 1.e-8);
            assert(rdiff < 1.e-8);
         }

         // assemble RDMs by pattern
         for(int i=0; i<allpatterns.size(); i++){
            const auto& pattern = allpatterns[i];
            const auto& loptype = str2optype.at(pattern.left);
            const auto& coptype = str2optype.at(pattern.center);
            const auto& roptype = str2optype.at(pattern.right);
            const auto& lkey = loptype.first; 
            const auto& ldagger = loptype.second;
            const auto& ckey = coptype.first; 
            const auto& cdagger = coptype.second;
            const auto& rkey = roptype.first; 
            const auto& rdagger = roptype.second;
            const auto& lops = lqops(lkey);
            const auto& cops = cqops(ckey);
            const auto& rops = rqops(rkey);
            if(debug){
               std::cout << " i=" << i 
                  << " pattern=" << pattern.to_string() 
                  << " opkey=" << lkey << ldagger 
                  << ":" << ckey << cdagger 
                  << ":" << rkey << rdagger
                  << " sizes=" << lops.size()
                  << ":" << cops.size()  
                  << ":" << rops.size();
               if(alg_rdm == 0) std::cout << std::endl;
            }

            // for parallel computation
            std::vector<int> reval;
            std::vector<int> leval;
            const bool ifkr = Qm::ifkr;
            int sorb = icomb.get_nphysical()*2;
            setup_evalmap(pattern.num_string(), ifkr, sorb, lops, rops, leval, reval, size, rank);

            auto ti = tools::get_time();
            if(alg_rdm == 0){

               rdm_assemble_simple(is_same, pattern, lkey, ckey, rkey, ldagger, cdagger, rdagger, lops, cops, rops, 
                     wf3bra, wf3ket, leval, reval, rdm, tdm, rank);

            }else if(alg_rdm == 1){

               rdm_assemble_simple_parallel(is_same, lkey, ckey, rkey, ldagger, cdagger, rdagger, lops, cops, rops, 
                     wf3bra, wf3ket, leval, reval, rdm);

            }else{
               tools::exit("error: no such option for alg_rdm");
            } // alg_rdm
            if(debug){
               auto tf = tools::get_time();
               double dt = tools::get_duration(tf-ti);
               std::cout << " TIMING=" << dt << " S" << std::endl;
            }
         } // pattern
*/

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::rdm_assemble(su2)", t0, t1);
         }
      }

} // ctns

#endif
