#ifndef RDM_COMPUTE_H
#define RDM_COMPUTE_H

#include "rdm_string.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void rdm_compute(const int order,
            const bool is_same,
            const comb<Qm,Tm>& icomb,
            const int isite,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<type_pattern>& patterns,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm1,
            const linalg::matrix<Tm>& tdm1){
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

         const std::map<std::string,std::pair<char,bool>> str2optype_same = {
            {"",{'I',0}},
            {"+",{'C',0}},
            {"-",{'C',1}},
            {"++",{'A',0}},
            {"--",{'A',1}},
            {"+-",{'B',0}},
            {"-+",{'B',1}},
            // only needed for dot operators: cop, lop[fpattern], rop[lpattern]
            {"+--",{'T',0}},
            {"++-",{'T',1}},
            {"++--",{'F',0}}
         };
         const std::map<std::string,std::pair<char,bool>> str2optype_diff = {
            {"",{'I',0}},
            {"+",{'C',0}},
            {"-",{'D',0}},
            {"++",{'A',0}},
            {"--",{'M',0}},
            {"+-",{'B',0}},
            // only needed for dot operators: cop, lop[fpattern], rop[lpattern]
            {"+--",{'T',0}},
            {"++-",{'T',1}},
            {"++--",{'F',0}}
         };
         auto& str2optype = is_same? str2optype_same : str2optype_diff;

         const std::map<char,int> op2parity = {
            {'I',0},
            {'C',1},{'D',1},
            {'A',0},{'B',0},{'M',0},
            {'T',1},{'F',0}
         };

         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         lqops.print("lqops");
         rqops.print("rqops");
         cqops.print("cqops");

         if(is_same){
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
         }

         // compute RDMs by pattern
         for(int i=0; i<patterns.size(); i++){
            const auto& pattern = patterns[i];
            std::cout << "\ni=" << i 
               << " pattern=" << pattern.to_string()
               << std::endl;
            const auto& loptype = str2optype.at(pattern.left);
            const auto& coptype = str2optype.at(pattern.center);
            const auto& roptype = str2optype.at(pattern.right);
            const auto& lkey = loptype.first; 
            const auto& ldagger = loptype.second;
            const auto& ckey = coptype.first; 
            const auto& cdagger = coptype.second;
            const auto& rkey = roptype.first; 
            const auto& rdagger = roptype.second;
            std::cout << "\ni=" << i 
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
            int lparity = op2parity.at(lkey);
            int cparity = op2parity.at(ckey);
            int rparity = op2parity.at(rkey);

            if(alg_rdm == 0){

               // assemble rdms
               for(const auto& rpr : rops){
                  const auto& rdx = rpr.first;
                  const auto& rop = rpr.second;
                  std::cout << "rop: key=" << rkey << " rdx=" << rdx << " normF()=" << rop.normF() << std::endl;
                  auto rstr = get_calst(rkey, rdx, rdagger);
                  auto opxwf1 = oper_kernel_IOwf("cr", wf3ket, rop, rparity, rdagger);
                  for(const auto& cpr : cops){
                     const auto& cdx = cpr.first;
                     const auto& cop = cpr.second;
                     std::cout << "cop: key=" << ckey << " cdx=" << cdx << " normF()=" << cop.normF() << std::endl;
                     auto cstr = get_calst(ckey, cdx, cdagger);
                     auto opxwf2 = oper_kernel_OIwf("cr", opxwf1, cop, cdagger);
                     if((cparity+rparity)%2 == 1) opxwf2.row_signed();
                     auto op2 = contract_qt3_qt3("cr", wf3bra, opxwf2); 
                     for(const auto& lpr : lops){
                        const auto& ldx = lpr.first;
                        auto lop = ldagger? lpr.second.H() : lpr.second;
                        std::cout << "lop: key=" << lkey << " ldx=" << ldx << " normF()=" << lop.normF() << std::endl;
                        auto lstr = get_calst(lkey, ldx, ldagger);
                        if(tools::is_complex<Tm>()) lop = lop.conj();
                        Tm val = contract_qt2_qt2_full(lop, op2); 
                        rdmstring rdmstr(lstr, cstr, rstr);
                        auto rdmstr2 = rdmstr;
                        Tm sgn = rdmstr2.sort();
                        auto ijdx = rdmstr2.get_ijdx();
                        size_t idx = ijdx.first;
                        size_t jdx = ijdx.second;
                        rdm1(idx,jdx) = sgn*val;
                        double diff = std::abs(sgn*val - tdm1(idx,jdx));
                        std::cout << "ldx,cdx,rdx=" << ldx << "," << cdx << "," << rdx 
                           << " rdmstr=" << rdmstr.to_string()
                           << " rdmstr2=" << rdmstr2.to_string()
                           << " sgn=" << sgn 
                           << " val=" << std::setprecision(schd.ctns.outprec) << sgn*val
                           << " idx,jdx=" << idx << "," << jdx
                           << " tdm=" << tdm1(idx,jdx)
                           << " diff=" << diff
                           << std::endl;
                        assert(diff < 1.e-10);
                        if(is_same) rdm1(jdx,idx) = tools::conjugate(rdm1(idx,jdx));
                     }
                  }
               }

               // spin-recoupling for su2 case

               // reorder indices to physical

            } // alg_rdm
         } // pattern

         //exit(1);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::rdm_compute", t0, t1);
         }
      }

} // ctns

#endif
