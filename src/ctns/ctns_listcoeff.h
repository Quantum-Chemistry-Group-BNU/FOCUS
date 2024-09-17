#ifndef CTNS_LISTCOEFF_H
#define CTNS_LISTCOEFF_H

namespace ctns{

   // list all states with |coeff|>thresh_cabs
   // code structure is similar to rcanon_random in ctns_rcandom.h
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rcanon_listcoeff(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_cabs){
         auto t0 = tools::get_time();
         std::cout << "\nctns::rcanon_listcoeff:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot
            << " thresh_cabs=" << thresh_cabs
            << std::endl;
         
         assert(icomb.topo.ifmps);
         int ks = icomb.get_nphysical();
         int k = 2*ks;
         std::vector<std::pair<fock::onstate,stensor2<Tm>>> space(1);
         fock::onstate state(k);
         auto wf = icomb.rwfuns[iroot];
         space[0] = std::make_pair(state,wf);
         
         // start iteration
         std::cout << "breadth first search for the configuration tree" << std::endl;
         const auto& rindex = icomb.topo.rindex;
         for(int i=0; i<ks; i++){
            if(space.size() == 0) break; // no acceptable state
            std::vector<std::pair<fock::onstate,stensor2<Tm>>> space_new;
            for(int j=0; j<space.size(); j++){
               const auto& state = space[j].first;
               const auto& wf = space[j].second;
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // compute probability for physical index
               for(int ndx=0; ndx<4; ndx++){
                  auto qt2n = qt3.fix_mid( idx2mdx(Qm::isym, ndx) );
                  // \sum_a |psi[n,a]|^2
                  auto psi2 = qt2n.dot(qt2n.H());
                  double weight = std::real(psi2(0,0)(0,0));
                  if(std::sqrt(weight) < thresh_cabs){
                     continue;
                  }else{
                     auto state_new = state; 
                     if(ndx == 1){
                        state_new[2*i] = 1;
                        state_new[2*i+1] = 1;
                     }else if(ndx == 2){
                        state_new[2*i] = 1;
                     }else if(ndx == 3){
                        state_new[2*i+1] = 1;
                     }
                     space_new.push_back(std::make_pair(state_new,qt2n));
                  }
               }
            }
            space = std::move(space_new);
            std::cout << " isite=" << i << " space.size()=" << space.size() << std::endl;
         }
         
         // sort by |coeff|
         std::stable_sort(space.begin(), space.end(),
               [](const auto& pr1, const auto& pr2){
               return std::abs(pr1.second(0,0)(0,0)) > std::abs(pr2.second(0,0)(0,0));
               });
         
         // print find results
         std::cout << "find " << space.size() << " states with |coeff|>thresh_cabs=" << thresh_cabs << std::endl;
         double psum = 0.0;
         for(int i=0; i<space.size(); i++){
            const auto& state = space[i].first;
            const auto& wf = space[i].second;
            const auto& coeff = wf(0,0)(0,0);
            psum += std::norm(coeff);
            std::cout << " i=" << i << " state=" << state << " coeff=" << coeff << " psum[accum]=" << psum << std::endl; 
         }
         std::cout << "psum=" << psum << std::endl;
         
         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_listcoeff", t0, t1);
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rcanon_listcoeff(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_cabs){
         auto t0 = tools::get_time();
         std::cout << "\nctns::rcanon_listcoeff:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot
            << " thresh_cabs=" << thresh_cabs
            << std::endl;
        
         assert(icomb.topo.ifmps);
         int ks = icomb.get_nphysical();
         int k = 2*ks;
         std::vector<std::pair<fock::csfstate,stensor2su2<Tm>>> space(1);
         fock::csfstate state(k);
         auto wf = icomb.rwfuns[iroot];
         space[0] = std::make_pair(state,wf);
        
         // start iteration
         std::cout << "breadth first search for the configuration tree" << std::endl;
         const auto& rindex = icomb.topo.rindex;
         auto sym = icomb.get_qsym_state();
         int ne = sym.ne();
         int ts = sym.ts();
         for(int i=0; i<ks; i++){
            if(space.size() == 0) break; // no acceptable state
            std::vector<std::pair<fock::csfstate,stensor2su2<Tm>>> space_new;
            for(int j=0; j<space.size(); j++){
               const auto& state = space[j].first;
               const auto& wf = space[j].second;
               /*
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // compute probability for physical index
               for(int ndx=0; ndx<4; ndx++){
                  auto qt2n = qt3.fix_mid( idx2mdx(Qm::isym, ndx) );
                  // \sum_a |psi[n,a]|^2
                  auto psi2 = qt2n.dot(qt2n.H());
                  double weight = std::real(psi2(0,0)(0,0));
                  if(std::sqrt(weight) < thresh_cabs){
                     continue;
                  }else{
                     auto state_new = state; 
                     if(ndx == 1){
                        state_new[2*i] = 1;
                        state_new[2*i+1] = 1;
                     }else if(ndx == 2){
                        state_new[2*i] = 1;
                     }else if(ndx == 3){
                        state_new[2*i+1] = 1;
                     }
                     space_new.push_back(std::make_pair(state_new,qt2n));
                  }
               }
               */
            }
            space = std::move(space_new);
            std::cout << " isite=" << i << " space.size()=" << space.size() << std::endl;
         }

         // sort by |coeff|
         std::stable_sort(space.begin(), space.end(),
               [](const auto& pr1, const auto& pr2){
               return std::abs(pr1.second(0,0)(0,0)) > std::abs(pr2.second(0,0)(0,0));
               });

         // print find results
         std::cout << "find " << space.size() << " states with |coeff|>thresh_cabs=" << thresh_cabs << std::endl;
         double psum = 0.0;
         for(int i=0; i<space.size(); i++){
            const auto& state = space[i].first;
            const auto& wf = space[i].second;
            const auto& coeff = wf(0,0)(0,0);
            psum += std::norm(coeff);
            std::cout << " i=" << i << " state=" << state << " coeff=" << coeff << " psum[accum]=" << psum << std::endl; 
         }
         std::cout << "psum=" << psum << std::endl;

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_listcoeff", t0, t1);
      }

} // ctns

#endif
