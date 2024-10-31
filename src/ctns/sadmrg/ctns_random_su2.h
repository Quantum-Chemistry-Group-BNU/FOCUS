#ifndef CTNS_RANDOM_SU2_H
#define CTNS_RANDOM_SU2_H

/*
   Algorithms for CTNS:

   3. rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
*/

#include "../../core/onspace.h"
#include "../../core/analysis.h"
#include "../ctns_comb.h"

namespace ctns{

   // --- Non-Abelian MPS ---

   // Sampling SA-MPS to get {|csf>,coeff(csf)=<csf|Psi[i]>} 
   // In case that CTNS is unnormalized, p(csf) is also unnormalized. 
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::csfstate,Tm> rcanon_random(const comb<Qm,Tm>& icomb, 
            const int iroot,
            const bool debug=false){
         // only correct for MPS, because csf is linearly coupled.
         assert(icomb.topo.ifmps);
         if(debug) std::cout << "\nctns::rcanon_random iroot=" << iroot << std::endl; 
         const int nsite = icomb.get_nphysical();
         fock::csfstate state(nsite);
         // initialize boundary wf for i-th state
         auto wf = icomb.rwfuns[iroot];
         const auto& nodes = icomb.topo.nodes; 
         const auto& rindex = icomb.topo.rindex;
         auto sym_state = icomb.get_qsym_state();
         int ne = sym_state.ne(), ts = sym_state.ts();
         // loop from left to right
         for(int i=0; i<icomb.topo.nbackbone; i++){
            const auto& node = nodes[i][0];
            assert(node.lindex == i); 
            int tp = node.type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               
               // 0. setup probability for (bc,bm)
               auto qt3 = contract_qt3_qt2("l",site,wf);
               const auto& qrow = qt3.info.qrow;
               const auto& qcol = qt3.info.qcol;
               const auto& qmid = qt3.info.qmid;
               assert(qrow.size() == 1);
               int size = qmid.size()*qcol.size();
               std::vector<double> weights(size,0.0);
               std::vector<std::pair<int,int>> indices(size);
               for(int bm=0; bm<qmid.size(); bm++){
                  for(int bc=0; bc<qcol.size(); bc++){
                     int iaddr = bm*qcol.size()+bc;
                     indices[iaddr] = std::make_pair(bc,bm);
                     auto qr = qrow.get_sym(0);
                     auto qc = qcol.get_sym(bc);
                     auto qm = qmid.get_sym(bm);
                     auto blk3 = qt3.get_rcf_symblk(qr,qc,qm);
                     if(blk3.empty()) continue;
                     weights[iaddr] = std::pow(linalg::xnrm2(blk3.size(), blk3.data()),2);
                  }
               }

               // 1. sample
               std::discrete_distribution<> dist(weights.begin(), weights.end());
               int idx = dist(tools::generator);
               auto key = indices[idx];
               int bc = key.first;
               int bm = key.second;
               //std::cout << "\nsite i=" << i << " bc=" << bc << " bm=" << bm << std::endl;
               //qt3.print("qt3",2);
               
               // 2. construct wf
               qbond qleft({{qcol.get_sym(bc),1}}); // because dr*dm=1
               wf.init(qsym(3,0,0),qleft,qcol,{1-std::get<1>(qt3.info.dir),std::get<1>(qt3.info.dir)});
               auto blk3 = qt3(0,bc,bm,qrow.get_sym(0).ts());
               auto blk2 = wf(0,bc);
               //wf.print("wf",2);
               assert(!blk3.empty() && !blk2.empty() && blk3.size()==blk2.size());
               linalg::xcopy(blk3.size(), blk3.data(), blk2.data());
               
               // 3. setup state
               auto sym = qcol.get_sym(bc);
               int dne = ne - sym.ne();
               int dts = ts - sym.ts();
               state.setocc(i, dne, dts);
               ne = sym.ne();
               ts = sym.ts();
            } // tp
         }
         // finally wf should be the corresponding CI coefficients
         assert(wf.rows() == 1 && wf.cols() == 1); 
         Tm coeff0 = wf(0,0)(0,0);
         if(debug){
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0=" << coeff0 
               << " coeff1=" << coeff1 
               << " diff=" << coeff0-coeff1 
               << std::endl;
            assert(std::abs(coeff0-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

   // Sampling SA-MPS to get {|det>,coeff(det)=<det|Psi[i]>} 
   // In case that CTNS is unnormalized, p(det) is also unnormalized. 
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::onstate,Tm> rcanon_random_det(const comb<Qm,Tm>& icomb, 
            const int iroot,
            const bool debug=false){
         // only correct for MPS, because csf is linearly coupled.
         assert(icomb.topo.ifmps);
         if(debug) std::cout << "\nctns::rcanon_random_det iroot=" << iroot << std::endl;
         const int nsite = icomb.get_nphysical();
         fock::onstate state(2*nsite);
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
         if(debug) wf2blk.print("wf2blk");
         bmats[0] = wf2blk.to_matrix(); 
         // loop from left to right
         auto qphys = get_qbond_phys(2); // {0,2,a,b} 
         const auto& nodes = icomb.topo.nodes; 
         const auto& rindex = icomb.topo.rindex;
         for(int i=0; i<icomb.topo.nbackbone; i++){
            const auto& node = nodes[i][0];
            assert(node.lindex == i); 
            int tp = node.type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               if(debug) std::cout << "isite=" << i << std::endl; 
               // 0. setup probablity for bm
               const auto& qcol = site.info.qcol;
               std::vector<qbond> qrights(4);
               std::vector<std::vector<linalg::matrix<Tm>>> bmats_vec(4);
               std::vector<double> weights(4,0.0);
               for(int ndx=0; ndx<4; ndx++){
                  auto qc2 = qphys.get_sym(ndx);
                  int ne_i = qc2.ne(), tm_i = qc2.tm(), na_i = (ne_i+tm_i)/2, nb_i = ne_i-na_i;
                  qsym qc(3,ne_i,na_i+nb_i==1);
                  if(debug) std::cout << " ndx=" << ndx << " qc2=" << qc2
                     << " (na_ia,nb_i)=" << na_i << "," << nb_i
                     << " qc=" << qc
                     << std::endl;
                  // construct Bnew[r] = sum_l B[l]*A[l,r] for each ndx
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
                        int tm_r = tm-tm_i;
                        int tm_l = tm;
                        Tm cg = fock::cgcoeff(qc.ts(),qr.ts(),ql.ts(),na_i-nb_i,tm_r,tm_l);
                        linalg::xaxpy(tmp2.size(), cg, tmp2.data(), bmat.data());
                     } // l
                     if(ifexist){
                        qrights[ndx].dims.push_back(std::make_pair(qr,dr));
                        bmats_vec[ndx].push_back(bmat); 
                        weights[ndx] += std::norm(linalg::xnrm2(bmat.size(), bmat.data())); 
                     }
                  } // r
               } // ndx
               // 1. sample physical index
               std::discrete_distribution<> dist(weights.begin(),weights.end());
               int k = dist(tools::generator);
               idx2occ(state, node.lindex, k);
               // 2. update qleft and bmats
               qleft = std::move(qrights[k]);
               bmats = std::move(bmats_vec[k]);
               tm -= qphys.get_sym(k).tm();
            } // tp
         } // i
         // finally bmats should be the corresponding CI coefficients
         assert(bmats.size() == 1 and bmats[0].size() == 1);
         Tm coeff0 = bmats[0](0,0);
         if(debug){
            qleft.print("qleft");
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0=" << coeff0 
               << " coeff1=" << coeff1 
               << " diff=" << coeff0-coeff1 
               << std::endl;
            assert(std::abs(coeff0-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

} // ctns

#endif
