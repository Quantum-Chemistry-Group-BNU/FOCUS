#ifndef CTNS_LISTCOEFF_H
#define CTNS_LISTCOEFF_H

namespace ctns{

   template <typename Tm>
      void update_space_new(const int i,
            const qsym& sym_state,
            const fock::onstate& state, 
            const stensor3<Tm>& qt3, 
            const double thresh_cabs, 
            std::vector<std::pair<fock::onstate, stensor2<Tm>>>& space_new){
         // compute probability for physical index
         for(int ndx=0; ndx<4; ndx++){
            auto qt2n = qt3.fix_mid( idx2mdx(sym_state.isym(), ndx) );
            // \sum_a |psi[n,a]|^2
            auto psi2 = qt2n.dot(qt2n.H());
            double weight = std::real(psi2(0,0)(0,0));
            if(std::sqrt(weight) < thresh_cabs) continue;
            auto state_new = state; 
            idx2occ(state_new, i, ndx);
            space_new.push_back(std::make_pair(state_new,qt2n));
         }
      }

   template <typename Tm>
      void update_space_new(const int i,
            const qsym& sym_state,
            const fock::csfstate& state, 
            const stensor3su2<Tm>& qt3, 
            const double thresh_cabs, 
            std::vector<std::pair<fock::csfstate, stensor2su2<Tm>>>& space_new){
         // 1. compute weights for (bc,bm) following ctns_random.h
         const auto& qrow = qt3.info.qrow;
         const auto& qcol = qt3.info.qcol;
         const auto& qmid = qt3.info.qmid;
         assert(qrow.size() == 1);
         for(int bm=0; bm<qmid.size(); bm++){
            for(int bc=0; bc<qcol.size(); bc++){
               auto qr = qrow.get_sym(0);
               auto qc = qcol.get_sym(bc);
               auto qm = qmid.get_sym(bm);
               auto blk3 = qt3.get_rcf_symblk(qr,qc,qm);
               if(blk3.empty()) continue;
               double weight = linalg::xnrm2(blk3.size(), blk3.data());
               if(weight < thresh_cabs) continue;
               // 2. construct wf
               qbond qleft({{qcol.get_sym(bc),1}}); // because dr*dm=1
               stensor2su2<Tm> qt2n(qsym(3,0,0),qleft,qcol,{1-std::get<1>(qt3.info.dir),std::get<1>(qt3.info.dir)});
               //auto blk3 = qt3(0,bc,bm,qrow.get_sym(0).ts()); // CRcouple
               auto blk2 = qt2n(0,bc);
               assert(!blk3.empty() && !blk2.empty() && blk3.size()==blk2.size());
               linalg::xcopy(blk3.size(), blk3.data(), blk2.data());
               // 3. setup state 
               auto sym = qcol.get_sym(bc);
               int dne = (sym_state.ne() - state.nelec()) - sym.ne();
               int dts = (sym_state.ts() - state.twos()) - sym.ts();
               auto state_new = state;
               state_new.setocc(i, dne, dts);
               space_new.push_back(std::make_pair(state_new,qt2n));
            } // bc
         } // bm
      }

   // list all states with |coeff|>thresh_cabs
   // code structure is similar to rcanon_random in ctns_rcandom.h
   template <typename Qm, typename Tm>
      void rcanon_listcoeff(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_cabs,
            const std::string saveconfs=""){
         if(thresh_cabs > 1.0) return;
         using statetype = typename std::conditional<Qm::ifabelian, fock::onstate, fock::csfstate>::type;
         const bool ifab = Qm::ifabelian;
         std::cout << "\nctns::rcanon_listcoeff:"
            << " ifab=" << ifab
            << " iroot=" << iroot
            << " thresh_cabs=" << thresh_cabs
            << " saveconfs=" << saveconfs
            << std::endl;
         auto t0 = tools::get_time();

         assert(icomb.topo.ifmps);
         int ks = icomb.get_nphysical();
         int k = 2*ks;
         std::vector<std::pair<statetype, qtensor2<ifab,Tm>>> space(1);
         statetype state((ifab? k : ks));
         auto wf = icomb.rwfuns[iroot];
         space[0] = std::make_pair(state,wf);

         // start iteration
         std::cout << "breadth first search for the configuration tree" << std::endl;
         const auto& rindex = icomb.topo.rindex;
         auto sym_state = icomb.get_qsym_state();
         for(int i=0; i<ks; i++){
            if(space.size() == 0) break; // no acceptable state
            std::vector<std::pair<statetype, qtensor2<ifab,Tm>>> space_new;
#ifdef SERIAL
            for(int j=0; j<space.size(); j++){
               const auto& state = space[j].first;
               const auto& wf = space[j].second;
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               update_space_new(i, sym_state, state, qt3, thresh_cabs, space_new);
            } // j
#else            
            // openmp version
            #pragma omp parallel
            {
               std::vector<std::pair<statetype, qtensor2<ifab,Tm>>> space_local;
               #pragma omp for schedule(dynamic) nowait
               for(int j=0; j<space.size(); j++){
                  const auto& state = space[j].first;
                  const auto& wf = space[j].second;
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
                  auto qt3 = contract_qt3_qt2("l",site,wf);
                  update_space_new(i, sym_state, state, qt3, thresh_cabs, space_local);
               } // j
               #pragma omp critical
               {
                  std::copy(space_local.begin(), space_local.end(), std::back_inserter(space_new)); 
               }
            }
#endif
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
            std::cout << " i=" << i << " state=" << state 
               << " coeff=" << std::setw(9) << std::fixed << std::setprecision(6) << coeff 
               << " psum[accum]=" << psum 
               << std::endl; 
         }
         std::cout << "psum=" << psum << std::endl;

         // save configurations to text file
         if(!saveconfs.empty()){
            std::cout << "save to file " << saveconfs << "_" << "list.txt" << std::endl;
            std::ofstream file(saveconfs+"_list.txt");
            file << std::fixed << std::setprecision(12);
            file << "size= " << space.size() << " psum= " << psum << std::endl;
            for(int i=0; i<space.size(); i++){
               const auto& state = space[i].first;
               const auto& wf = space[i].second;
               const auto& coeff = wf(0,0)(0,0);
               file << state << " " << coeff << std::endl;
            }
            file.close(); 
         }

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_listcoeff", t0, t1);
      }

} // ctns

#endif
