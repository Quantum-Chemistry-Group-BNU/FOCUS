#ifndef CTNS_ALG_H
#define CTNS_ALG_H

/*
   Algorithms for CTNS:

   0. rcanon_check: check sites in right canonical form (RCF)
   1. get_Smat: <CTNS[i]|CTNS[j]> 
   2. rcanon_CIcoeff: <n|CTNS>
      rcanon_CIovlp: <CI|CTNS>
      rcanon_CIcoeff_check
   3. rcanon_Sdiag_exact:
      rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
      rcanon_Sdiag_sample: compute Sdiag via sampling
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"

namespace ctns{

   // Algorithm 0:
   // Check right canonical form
   template <typename Km>
      void rcanon_check(const comb<Km>& icomb,
            const double thresh_ortho,
            const bool ifortho=true){
         std::cout << "\nctns::rcanon_check thresh_ortho=" 
            << std::scientific << std::setprecision(2) << thresh_ortho 
            << std::endl;
         // loop over all sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto p = icomb.topo.rcoord[idx];
            // check right canonical form -> A*[l'cr]A[lcr] = w[l'l] = Id
            auto qt2 = contract_qt3_qt3("cr", icomb.sites[idx], icomb.sites[idx]);
            double maxdiff = qt2.check_identityMatrix(thresh_ortho, false);
            int Dtot = qt2.info.qrow.get_dimAll();
            std::cout << " idx=" << idx << " node=" << p << " Dtot=" << Dtot 
               << " maxdiff=" << std::scientific << maxdiff << std::endl;
            if((ifortho || (!ifortho && idx != icomb.topo.ntotal-1)) && (maxdiff>thresh_ortho)){
               tools::exit("error: deviate from identity matrix!");
            }
         } // idx
      }

   // Algorithm 1:
   // <CTNS[i]|CTNS[j]>: compute by a typical loop for right canonical form 
   template <typename Km>
      linalg::matrix<typename Km::dtype> get_Smat(const comb<Km>& icomb){ 
         // loop over sites on backbone
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         stensor2<typename Km::dtype> qt2_r, qt2_u;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
            int tp = node.type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               if(i == icomb.topo.nbackbone-1){
                  qt2_r = contract_qt3_qt3("cr",site,site);
               }else{
                  auto qtmp = contract_qt3_qt2("r",site,qt2_r);
                  qt2_r = contract_qt3_qt3("cr",site,qtmp);
               }
            }else if(tp == 3){
               for(int j=nodes[i].size()-1; j>=1; j--){
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,j))];
                  if(j == nodes[i].size()-1){
                     qt2_u = contract_qt3_qt3("cr",site,site);
                  }else{
                     auto qtmp = contract_qt3_qt2("r",site,qt2_u);
                     qt2_u = contract_qt3_qt3("cr",site,qtmp);
                  }
               } // j
               // internal site without physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qtmp = contract_qt3_qt2("r",site,qt2_r); // ket
               qtmp = contract_qt3_qt2("c",qtmp,qt2_u); // upper branch
               qt2_r = contract_qt3_qt3("cr",site,qtmp); // bra
            }
         } // i
         auto Smat = qt2_r.to_matrix();
         return qt2_r.to_matrix();
      }

   // Algorithm 2:
   // <n|CTNS[i]> by contracting the CTNS
   template <typename Km>
      std::vector<typename Km::dtype> rcanon_CIcoeff(const comb<Km>& icomb,
            const fock::onstate& state){
         using Tm = typename Km::dtype;
         // compute <n|CTNS> by contracting all sites
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         stensor2<Tm> qt2_r, qt2_u;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
            int tp = node.type;
            if(tp == 0 || tp == 1){
               // site on backbone with physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               // given occ pattern, extract the corresponding qblock
               auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, node.pindex) ); 
               if(i == icomb.topo.nbackbone-1){
                  qt2_r = std::move(qt2);
               }else{
                  qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
               }
            }else if(tp == 3){
               // propogate symmetry from leaves down to backbone
               for(int j=nodes[i].size()-1; j>=1; j--){
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,j))];
                  const auto& nodej = nodes[i][j];
                  auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, nodej.pindex) );
                  if(j == nodes[i].size()-1){
                     qt2_u = std::move(qt2);
                  }else{
                     qt2_u = qt2.dot(qt2_u);
                  }
               } // j
               // internal site without physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               // contract upper matrix: permute row and col for contract_qt3_qt2_c
               auto qt3 = contract_qt3_qt2("c",site,qt2_u.T());
               auto qt2 = qt3.fix_mid( std::make_pair(0,0) );
               qt2_r = qt2.dot(qt2_r); // contract right matrix
            } // tp
         } // i
         const auto& wfcoeff = qt2_r; 
         assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
         // finally return coeff = <n|CTNS[i]> as a vector 
         int n = icomb.get_nroots(); 
         std::vector<Tm> coeff(n,0.0);
         // in case this CTNS does not encode this det, no such block 
         const auto blk2 = wfcoeff(0,0);
         if(blk2.empty()) return coeff; 
         assert(blk2.size() == n);
         // compute fermionic sign changes to match ordering of orbitals
         double sgn = state.permute_sgn(icomb.topo.image2);
         linalg::xaxpy(n, sgn, blk2.data(), coeff.data());
         return coeff;
      }

   // check rcanon_CIcoeff
   template <typename Km>
      int rcanon_CIcoeff_check(const comb<Km>& icomb,
            const fock::onspace& space,
            const linalg::matrix<typename Km::dtype>& vs,
            const double thresh=1.e-8){
         std::cout << "\nctns::rcanon_CIcoeff_check" << std::endl;
         int n = icomb.get_nroots(); 
         int dim = space.size();
         double maxdiff = -1.e10;
         // cmat[j,i] = <D[i]|CTNS[j]>
         for(int i=0; i<dim; i++){
            auto coeff = rcanon_CIcoeff(icomb, space[i]);
            std::cout << " i=" << i << " state=" << space[i] << std::endl;
            for(int j=0; j<n; j++){
               auto diff = std::abs(coeff[j] - vs(i,j));
               std::cout << "   j=" << j << " <n|CTNS[j]>=" << coeff[j] 
                  << " <n|CI[j]>=" << vs(i,j)
                  << " diff=" << diff << std::endl;
               maxdiff = std::max(maxdiff, diff);
            }
         }
         std::cout << "maxdiff = " << maxdiff << " thresh=" << thresh << std::endl;
         if(maxdiff > thresh) tools::exit("error: too large maxdiff in rcanon_CIcoeff_check!");
         return 0;
      }

   // ovlp[i,n] = <SCI[i]|CTNS[n]>
   template <typename Km>
      linalg::matrix<typename Km::dtype> rcanon_CIovlp(const comb<Km>& icomb,
            const fock::onspace& space,
            const linalg::matrix<typename Km::dtype>& vs){
         using Tm = typename Km::dtype;
         std::cout << "\nctns::rcanon_CIovlp" << std::endl;
         int n = icomb.get_nroots(); 
         int dim = space.size();
         // cmat[n,i] = <D[i]|CTNS[n]>
         linalg::matrix<Tm> cmat(n,dim);
         for(int i=0; i<dim; i++){
            auto coeff = rcanon_CIcoeff(icomb, space[i]);
            linalg::xcopy(n, coeff.data(), cmat.col(i));
         }
         // ovlp[i,n] = vs*[k,i] cmat[n,k]
         auto ovlp = linalg::xgemm("C","T",vs,cmat);
         return ovlp;
      }

   // Algorithm 3:
   // exact computation of Sdiag, only for small system
   template <typename Km>
      double rcanon_Sdiag_exact(const comb<Km>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         using Tm = typename Km::dtype; 
         std::cout << "\nctns::rcanon_Sdiag_exact iroot=" << iroot
            << " thresh_print=" << thresh_print << std::endl;
         // setup FCI space
         qsym sym_state = icomb.get_sym_state();
         int ne = sym_state.ne(); 
         int ks = icomb.get_nphysical();
         fock::onspace fci_space;
         if(Km::isym == 0){
            fci_space = fock::get_fci_space(2*ks);
         }else if(Km::isym == 1){
            fci_space = fock::get_fci_space(2*ks,ne);
         }else if(Km::isym == 2){
            int tm = sym_state.tm(); 
            int na = (ne+tm)/2, nb = ne-na;
            fci_space = fock::get_fci_space(ks,na,nb); 
         }
         int dim = fci_space.size();
         std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI=" << dim << std::endl;

         // brute-force computation of exact coefficients <n|CTNS>
         std::vector<Tm> coeff(dim,0.0);
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
         }
         double Sdiag = fock::coeff_entropy(coeff);
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2); 
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;

         // check: computation by sampling CI vector
         std::vector<double> weights(dim,0.0);
         std::transform(coeff.begin(), coeff.end(), weights.begin(),
               [](const Tm& x){ return std::norm(x); });
         std::discrete_distribution<> dist(weights.begin(),weights.end());
         const int nsample = 1e6;
         int noff = nsample/10;
         const double cutoff = 1.e-12;
         double Sd = 0.0, Sd2 = 0.0, std = 0.0;
         for(int i=0; i<nsample; i++){
            int idx = dist(tools::generator);
            auto ci2 = weights[idx];
            double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
            double fac = 1.0/(i+1.0);
            Sd = (Sd*i + s)*fac;
            Sd2 = (Sd2*i + s*s)*fac;
            if((i+1)%noff == 0){
               std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
               std::cout << " i=" << i << " Sd=" << Sd << " std=" << std 
                  << " range=(" << Sd-std << "," << Sd+std << ")"
                  << std::endl;
            }
         }
         return Sdiag;
      }

   // Sampling CTNS to get {|det>,p(det)=|<det|Psi[i]>|^2} 
   // In case that CTNS is unnormalized, p(det) is also unnormalized. 
   template <typename Km>
      std::pair<fock::onstate,double> rcanon_random(const comb<Km>& icomb, 
            const int iroot,
            const bool debug=false){
         if(debug) std::cout << "\nctns::rcanon_random iroot=" << iroot << std::endl; 
         using Tm = typename Km::dtype; 
         fock::onstate state(2*icomb.get_nphysical());
         // initialize boundary wf for i-th state
         auto wf = icomb.get_rwfun(iroot); 
         const auto& nodes = icomb.topo.nodes; 
         const auto& rindex = icomb.topo.rindex;
         // loop from left to right
         for(int i=0; i<icomb.topo.nbackbone; i++){
            int tp = nodes[i][0].type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // compute probability for physical index
               std::vector<stensor2<Tm>> qt2n(4);
               std::vector<double> weights(4);
               for(int idx=0; idx<4; idx++){
                  qt2n[idx] = qt3.fix_mid( idx2mdx(Km::isym, idx) );
                  // \sum_a |psi[n,a]|^2
                  auto psi2 = qt2n[idx].dot(qt2n[idx].H()); 
                  weights[idx] = std::real(psi2(0,0)(0,0));
               }
               std::discrete_distribution<> dist(weights.begin(),weights.end());
               int idx = dist(tools::generator);
               idx2occ(state, nodes[i][0].pindex, idx);
               wf = std::move(qt2n[idx]);
            }else if(tp == 3){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // propogate upwards
               for(int j=1; j<nodes[i].size(); j++){
                  const auto& sitej = icomb.sites[rindex.at(std::make_pair(i,j))];
                  // compute probability for physical index
                  std::vector<stensor3<Tm>> qt3n(4);
                  std::vector<double> weights(4);
                  for(int idx=0; idx<4; idx++){
                     auto qt2 = sitej.fix_mid( idx2mdx(Km::isym, idx) );
                     // purely change direction
                     qt3n[idx] = contract_qt3_qt2("c",qt3,qt2.T()); 
                     // \sum_ab |psi[n,a,b]|^2
                     auto psi2 = contract_qt3_qt3("cr",qt3n[idx],qt3n[idx]); 
                     weights[idx] = std::real(psi2(0,0)(0,0));
                  }
                  std::discrete_distribution<> dist(weights.begin(),weights.end());
                  int idx = dist(tools::generator);
                  idx2occ(state, nodes[i][j].pindex, idx);
                  qt3 = std::move(qt3n[idx]);
               } // j
               wf = qt3.fix_mid(std::make_pair(0,0));
            }
         }
         // finally wf should be the corresponding CI coefficients: coeff0*sgn = coeff1
         auto coeff0 = wf(0,0)(0,0);
         if(debug){
            double sgn = state.permute_sgn(icomb.topo.image2); // from orbital ordering
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0,sgn=" << coeff0 << "," << sgn
               << " coeff1=" << coeff1 
               << " diff=" << coeff0*sgn-coeff1 << std::endl;
            assert(std::abs(coeff0*sgn-coeff1)<1.e-10);
         }
         double prob = std::norm(coeff0);
         return std::make_pair(state,prob);
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]log2p[i] = - (sum_i p[i]) <log2p[i] > = -<psi|psi>*<log2p[i]>
   template <typename Km>
      double rcanon_Sdiag_sample(const comb<Km>& icomb,
            const int iroot,
            const int nsample,  
            const int nprt=10){ // no. of largest states to be printed
         const double cutoff = 1.e-12;
         std::cout << "\nctns::rcanon_Sdiag_sample iroot=" << iroot 
            << " nsample=" << nsample 
            << " nprt=" << nprt << std::endl;
         auto t0 = tools::get_time();
         const int noff = nsample/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
         std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
         // start sampling
         double Sd = 0.0, Sd2 = 0.0, std = 0.0;
         std::map<fock::onstate,int> pop;
         for(int i=0; i<nsample; i++){
            auto pr = rcanon_random(icomb,iroot);
            auto state = pr.first;
            auto ci2 = pr.second;
            // statistical analysis
            pop[state] += 1;
            double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
            double fac = 1.0/(i+1.0);
            Sd = (Sd*i + s)*fac;
            Sd2 = (Sd2*i + s*s)*fac;
            if((i+1)%noff == 0){
               std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
               auto t1 = tools::get_time();
               double dt = tools::get_duration(t1-t0);
               std::cout << " i=" << i << " Sd=" << Sd << " std=" << std
                  << " timing=" << dt << " s" << std::endl;	      
               t0 = tools::get_time();
            }
         }
         // print important determinants
         if(nprt > 0){
            int size = pop.size();
            std::cout << "sampled important determinants: pop.size=" << size << std::endl; 
            std::vector<fock::onstate> states(size);
            std::vector<int> counts(size);
            int i = 0;
            for(const auto& pr : pop){
               states[i] = pr.first;
               counts[i] = pr.second;
               i++;
            }
            auto indx = tools::sort_index(counts,1);
            // compare the first n important dets by counts
            int sum = 0;
            for(int i=0; i<std::min(size,nprt); i++){
               int idx = indx[i];
               fock::onstate state = states[idx];
               auto ci = rcanon_CIcoeff(icomb, state)[iroot];
               sum += counts[idx];
               std::cout << " i=" << i << " " << state
                  << " counts=" << counts[idx] 
                  << " p_i(sample)=" << counts[idx]/(1.0*nsample)
                  << " p_i(exact)=" << std::norm(ci)/ovlp 
                  << " c_i(exact)=" << ci/std::sqrt(ovlp)
                  << std::endl;
            }
            std::cout << "accumulated counts=" << sum 
               << " nsample=" << nsample 
               << " per=" << 1.0*sum/nsample << std::endl;
         }
         return Sd;
      }

} // ctns

#endif
