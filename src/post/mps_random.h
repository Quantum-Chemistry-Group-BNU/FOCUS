#ifndef MPS_RANDOM_H
#define MPS_RANDOM_H
#include "mps.h"

namespace ctns{

   // Sampling CTNS to get {|det>,coeff(det)=<det|Psi[i]>} 
   // In case that CTNS is unnormalized, p(det) is also unnormalized. 
   template <typename Qm, typename Tm>
      std::pair<fock::onstate,Tm> mps_random(const mps<Qm,Tm>& imps, 
            const int iroot, const bool debug=false){
         if(debug) std::cout << "\nctns::mps_random iroot=" << iroot << std::endl; 
         fock::onstate state(2*imps.nphysical);
         // initialize boundary wf for i-th state
         auto wf = imps.rwfuns[iroot];
         // loop from left to right
         for(int i=0; i<imps.nphysical; i++){
            const auto& site = imps.sites[i];
            auto qt3 = contract_qt3_qt2("l",site,wf);
            // compute probability for physical index
            std::vector<stensor2<Tm>> qt2n(4);
            std::vector<double> weights(4);
            for(int idx=0; idx<4; idx++){
               qt2n[idx] = qt3.fix_mid( idx2mdx(Qm::isym, idx) );
               // \sum_a |psi[n,a]|^2
               auto psi2 = qt2n[idx].dot(qt2n[idx].H()); 
               weights[idx] = std::real(psi2(0,0)(0,0));
            }
            std::discrete_distribution<> dist(weights.begin(),weights.end());
            int idx = dist(tools::generator);
            idx2occ(state, imps.get_pindex(i), idx);
            wf = std::move(qt2n[idx]);
         }
         // finally wf should be the corresponding CI coefficients: coeff0*sgn = coeff1
         auto coeff0 = wf(0,0)(0,0);
         if(debug){
            double sgn = state.permute_sgn(imps.image2); // from orbital ordering
            auto coeff1 = mps_CIcoeff(imps, iroot, state);
            std::cout << " state=" << state 
               << " coeff0,sgn=" << coeff0 << "," << sgn
               << " coeff1=" << coeff1 
               << " diff=" << coeff0*sgn-coeff1 << std::endl;
            assert(std::abs(coeff0*sgn-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]log2p[i] = - (sum_i p[i]) <log2p[i] > = -<psi|psi>*<log2p[i]>
   template <typename Qm, typename Tm>
      double mps_sdiag_sample(const mps<Qm,Tm>& imps,
            const int iroot,
            const int nsample,  
            const int nprt, // no. of largest states to be printed
            const double cutoff = 1.e-12){
         auto ti = tools::get_time();
         std::cout << "\nctns::mps_Sdiag_sample iroot=" << iroot 
            << " nsample=" << nsample 
            << " nprt=" << nprt << std::endl;
         const int noff = nsample/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(imps,imps)(iroot,iroot));
         std::cout << "<MPS[i]|MPS[i]> = " << ovlp << std::endl; 
         // start sampling
         double Sd = 0.0, Sd2 = 0.0, std = 0.0;
         std::map<fock::onstate,int> pop;
         auto t0 = tools::get_time();
         for(int i=0; i<nsample; i++){
            auto pr = mps_random(imps,iroot);
            auto state = pr.first;
            auto ci2 = std::norm(pr.second);
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
               std::cout << " i=" << i 
                  << "  Sd=" << std::setprecision(6) << Sd
                  << "  std=" << std
                  << "  timing=" << dt << " S" << std::endl;	      
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
               auto ci = mps_CIcoeff(imps, iroot, state);
               sum += counts[idx];
               std::cout << " i=" << i << " " << state
                  << " counts=" << counts[idx]
                  << std::scientific << std::setprecision(4)
                  << " p_i(sample)=" << counts[idx]/(1.0*nsample)
                  << " {p_i(exact)=" << std::norm(ci)/ovlp 
                  << " c_i(exact)="  << ci/std::sqrt(ovlp) << "}"
                  << std::endl;
            }
            std::cout << "accumulated counts=" << sum 
               << " nsample=" << nsample 
               << " per=" << 1.0*sum/nsample << std::endl;
         }
         auto tf = tools::get_time();
         tools::timing("ctns::mps_diag_sample", ti, tf);
         return Sd;
      }

   template <typename Qm, typename Tm>
      void mps_sdiag(const input::schedule& schd){
         std::cout << "\nctns::mps_sdiag" << std::endl;
         topology topo;
         topo.read(schd.post.topology_file);
         //topo.print();
         int nket = schd.post.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Qm,Tm> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.post.ket[j])+".info"; 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            // compute sdiag via sampling
            mps_sdiag_sample(kmps, schd.post.iroot, 
                  schd.post.nsample, 
                  schd.post.ndetprt);
         }
      }

} // ctns

#endif
