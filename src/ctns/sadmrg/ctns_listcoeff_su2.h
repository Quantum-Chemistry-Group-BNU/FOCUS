#ifndef CTNS_LISTCOEFF_SU2_H
#define CTNS_LISTCOEFF_SU2_H

namespace ctns{
        
   template <typename Tm>
      using onstate_data = std::tuple<fock::onstate, qbond, std::vector<linalg::matrix<Tm>>>; 

   template <typename Tm>
      void update_space_new(const int i,
            const qsym& sym_state,
            const onstate_data<Tm>& space_j,
            const stensor3su2<Tm>& site,
            const double thresh_cabs, 
            std::vector<onstate_data<Tm>>& space_new){
         // 1. compute weights for bm following ctns_random_su2.h
         const auto& state = std::get<0>(space_j);
         const auto& qleft = std::get<1>(space_j);
         const auto& bmats = std::get<2>(space_j);
         const auto& qcol = site.info.qcol;
         auto qphys = get_qbond_phys(2); // {0,2,a,b} 
         for(int ndx=0; ndx<4; ndx++){
            auto qc2 = qphys.get_sym(ndx);
            int ne_i = qc2.ne(), tm_i = qc2.tm(), na_i = (ne_i+tm_i)/2, nb_i = ne_i-na_i;
            qsym qc(3,ne_i,na_i+nb_i==1);
            // construct Bnew[r] = sum_l B[l]*A[l,r] for each ndx
            double weight = 0.0;
            onstate_data<Tm> item;
            for(int r=0; r<qcol.size(); r++){
               auto qr = qcol.get_sym(r);
               int dr = qcol.get_dim(r);
               linalg::matrix<Tm> bmat(1,dr);
               bool ifexist = false;
               for(int l=0; l<qleft.size(); l++){
                  auto ql = qleft.get_sym(l);
                  int dl = qleft.get_dim(l);
                  auto blk3 = site.get_rcf_symblk(ql,qr,qc);
                  if(blk3.empty()) continue;
                  // perform contraction Bnew[r] = sum_r B[l]*A[l,r]
                  ifexist = true;
                  linalg::matrix<Tm> tmp(dl,dr,blk3.data());
                  auto tmp2 = linalg::xgemm("N","N",bmats[l],tmp);
                  assert(tmp2.size() == bmat.size());
                  // <s[i]m[i]S[i-1]M[i-1]|S[i]M[i]> [consistent with csf.cpp]
                  int tm = sym_state.ts(); // high-spin
                  int tm_l = tm - state.twom();
                  int tm_r = tm_l  - tm_i;
                  Tm cg = fock::cgcoeff(qc.ts(),qr.ts(),ql.ts(),na_i-nb_i,tm_r,tm_l);
                  linalg::xaxpy(tmp2.size(), cg, tmp2.data(), bmat.data());
               } // l
               if(ifexist){
                  std::get<1>(item).dims.push_back(std::make_pair(qr,dr));
                  std::get<2>(item).push_back(bmat); 
                  weight += std::norm(linalg::xnrm2(bmat.size(), bmat.data()));
               }
            } // r
            if(std::sqrt(weight) < thresh_cabs) continue; 
            // setup state 
            auto state_new = state;
            idx2occ(state_new, i, ndx);
            std::get<0>(item) = std::move(state_new);
            space_new.push_back(item);
         } // ndx
      }

   // list all states with |coeff|>thresh_cabs
   // code structure is similar to rcanon_random in ctns_rcandom.h
   // ZL@2024/10/29: the essential idea is to use weights to do the screening! 
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rcanon_listcoeff_det(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_cabs,
            const std::string saveconfs=""){
         if(thresh_cabs > 1.0) return;
         const bool ifab = Qm::ifabelian;
         std::cout << "\nctns::rcanon_listcoeff_det:"
            << " ifab=" << ifab
            << " iroot=" << iroot
            << " thresh_cabs=" << thresh_cabs
            << " saveconfs=" << saveconfs
            << std::endl;
         auto t0 = tools::get_time();

         assert(icomb.topo.ifmps);
         int ks = icomb.get_nphysical();
         int k = 2*ks;
         // ZL@2024/10/30: the following part copied from ctns_random_su2.h
         auto sym_state = icomb.get_qsym_state();
         int ne = sym_state.ne(), ts = sym_state.ts(), tm = ts; // high-spin component
         // initialize boundary wf for i-th state
         auto wf2 = icomb.rwfuns[iroot];
         int bc = wf2.info.qcol.existQ(sym_state);
         assert(bc != -1);
         qbond qleft;
         qleft.dims.resize(1);
         qleft.dims[0] = wf2.info.qcol.dims[bc];
         std::vector<linalg::matrix<Tm>> bmats(1);
         auto wf2blk = wf2(0,bc);
         bmats[0] = wf2blk.to_matrix();
         std::vector<onstate_data<Tm>> space(1);
         fock::onstate state(k);
         std::get<0>(space[0]) = std::move(state);
         std::get<1>(space[0]) = std::move(qleft);
         std::get<2>(space[0]) = std::move(bmats);
         // start iteration
         std::cout << "breadth first search for the configuration tree:"
           << " depth=" << ks 
           << std::endl;
         const auto& rindex = icomb.topo.rindex;
         for(int i=0; i<ks; i++){
            if(space.size() == 0) break; // no acceptable state
            const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
            std::vector<onstate_data<Tm>> space_new;
            auto ti = tools::get_time();
#ifdef SERIAL
            for(int j=0; j<space.size(); j++){
               update_space_new(i, sym_state, space[j], site, thresh_cabs, space_new);
            } // j
#else            
            // openmp version
            #pragma omp parallel
            {
               std::vector<onstate_data<Tm>> space_local;
               #pragma omp for schedule(static) nowait
               for(int j=0; j<space.size(); j++){
                  update_space_new(i, sym_state, space[j], site, thresh_cabs, space_local);
               } // j
               #pragma omp critical
               std::copy(space_local.begin(), space_local.end(), std::back_inserter(space_new)); 
            }
#endif
            space = std::move(space_new);
            auto tf = tools::get_time();
            std::cout << " isite=" << i << " space.size()=" << space.size() 
               << " TIMING=" << tools::get_duration(tf-ti) << " S" 
               << std::endl;
         }

         // cleanup  
         size_t dim = space.size();         
         std::vector<std::pair<fock::onstate, Tm>> space_final(dim);
         std::transform(space.begin(), space.end(), space_final.begin(),
               [](const auto& x){ return std::make_pair(std::get<0>(x),std::get<2>(x)[0](0,0)); });
 
         print_listcoeff(space_final, thresh_cabs, saveconfs);

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_listcoeff_det", t0, t1);
      }

} // ctns

#endif
