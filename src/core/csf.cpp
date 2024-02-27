#include "csf.h"
#include "spin.h"
#include "tools.h"
#include "analysis.h"

using namespace std;
using namespace fock;

ostream& fock::operator <<(ostream& os, const csfstate& state){
   os << state.to_string();
   return os;
}

// right canonical form
std::vector<int> csfstate::intermediate_narray() const{
   int k = repr.size()/2;
   std::vector<int> ninter(k);
   for(int i=k-1; i>=0; i--){
      int ndelta = repr[2*i]+repr[2*i+1];
      if(i==k-1){
         ninter[i] = ndelta;
      }else{
         ninter[i] += ninter[i+1]+ndelta;
      }
   }
   return ninter;
}

std::vector<int> csfstate::intermediate_tsarray() const{
   int k = repr.size()/2;
   std::vector<int> tsinter(k);
   for(int i=k-1; i>=0; i--){
      int tsdelta = repr[2*i]-repr[2*i+1];
      if(i==k-1){
         tsinter[i] = tsdelta;
      }else{
         tsinter[i] += tsinter[i+1]+tsdelta;
      }
   }
   return tsinter;
}

double csfstate::det_coeff(const onstate& state) const{
   assert(repr.size() == state.size());
   int ks = state.size()/2; 
   double coeff = 1.0;
   int tsin = 0, tsout = 0;
   int tmin = 0, tmout = 0;
   for(int i=ks-1; i>=0; i--){
      int apos = 2*i;
      int bpos = 2*i+1;
      int dval = dvec(i);
      if(dval == 0 || dval == 3){
         if(state[bpos]*2+state[apos] != dval){
            coeff = 0.0;
            break;
         }
      }else{
         // open-shell case: dval=1/2
         if(state[bpos]+state[apos] != 1){
            coeff = 0.0;
            break;
         }
         int tsdelta = repr[apos]-repr[bpos];
         int tmdelta = state[apos]-state[bpos];
         tsout = tsin + tsdelta;
         tmout = tmin + tmdelta;
         // we impose that only the high-spin det is generated.
         if(i == ks-1) tmout = twos(); 
         if(abs(tmout)<=tsout){
            coeff *= cgcoeff(1,tsin,tsout,tmdelta,tmin,tmout); // <s[i]m[i]S[i-1]M[i-1]|S[i]M[i]>
         }else{
            coeff = 0.0;
            break;
         }
      }
      tsin = tsout;
      tmin = tmout;
   } // spatial orb
   return coeff;
}

std::pair<onspace,std::vector<double>> csfstate::to_det() const{
   onspace dets;
   std::vector<double> coeff;
   if(norb_single() == 0){
      dets.push_back(repr);
      coeff.resize(1);
      coeff[0] = 0;
   }else{
      int ne_os = norb_single();
      int ts = twos();
      int na_os = (ne_os+ts)/2;
      onspace dets_os = get_fci_space_single(norb_single(), na_os);
      size_t dim = dets_os.size();
      dets.resize(dim);
      coeff.resize(dim);
      auto orbs_os = orbs_single();
      for(size_t j=0; j<dim; j++){
         const auto& det = dets_os[j];
         auto state = repr;
         for(int i=0; i<orbs_os.size(); i++){
            int ios = orbs_os[i]; 
            state[2*ios] = det[2*i];
            state[2*ios+1] = det[2*i+1];
         }
         dets[j] = state;
         coeff[j] = det_coeff(state); 
      }
   }
   return std::make_pair(dets,coeff);
}

std::pair<onstate,double> csfstate::sample() const{
   int ks = norb();
   onstate state(2*ks);
   double coeff = 1.0;
   int tsin = 0, tsout = 0;
   int tmin = 0, tmout = 0;
   // sample must be carried out from left as in MPS
   for(int i=0; i<ks; i++){
   /*
      int apos = 2*i;
      int bpos = 2*i+1;
      int dval = dvec(i);
      if(dval == 0 || dval == 3){
         state[apos] = repr[apos];
         state[bpos] = repr[bpos];
      }else{
         // open-shell case: sample a/b
         int tsdelta = repr[apos]-repr[bpos];
         tsout = tsin + tsdelta;
         std::vector<double> weights(2);
         // alpha
         tmout = tmin + 1;
         double ca = 0.0;
         if(abs(tmout)<=tsout) ca = cgcoeff(1,tsin,tsout,1,tmin,tmout);
         weights[0] = ca*ca;                                    
         // beta                                                
         tmout = tmin - 1;                                      
         double cb = 0.0;                                       
         if(abs(tmout)<=tsout) cb = cgcoeff(1,tsin,tsout,-1,tmin,tmout);
         weights[1] = cb*cb;
         // sample 
         std::discrete_distribution<> dist(weights.begin(),weights.end());
         int idx = dist(tools::generator);
         state[apos] = 1-idx;
         state[bpos] = idx;
         tmout = tmin + (1-2*idx);
      }
      tsin = tsout;
      tmin = tmout;
   */
      exit(1);
   } // i
   return std::make_pair(state,coeff);
}

double csfstate::Sdiag_sample(const int nsample, const int nprt) const{
   const double cutoff = 1.e-12;
   std::cout << "\ncsfstate::Sdiag_sample nsample=" << nsample << std::endl;
   auto t0 = tools::get_time();
   const int noff = nsample/10;
   // start sampling
   double Sd = 0.0, Sd2 = 0.0, std = 0.0;
   std::map<fock::onstate,int> pop;
   for(int i=0; i<nsample; i++){
      auto pr = sample();
      auto state = pr.first;
      auto ci2 = std::norm(pr.second);
      // statistical analysis
      pop[state] += 1;
      double s = (ci2 < cutoff)? 0.0 : -log2(ci2);
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
         auto ci = det_coeff(state);
         sum += counts[idx];
         std::cout << " i=" << i << " " << state
            << " counts=" << counts[idx] 
            << " p_i(sample)=" << counts[idx]/(1.0*nsample)
            << " p_i(exact)=" << std::norm(ci)
            << " c_i(exact)=" << ci
            << std::endl;
      }
      std::cout << "accumulated counts=" << sum 
         << " nsample=" << nsample 
         << " per=" << 1.0*sum/nsample << std::endl;
   } // nprt
   return Sd;
}

double csfstate::Sdiag_exact() const{
   std::cout << "\ncsfstate::Sdiag_exact csf=" << (*this) << std::endl;
   auto detexpansion = to_det();
   size_t dim = detexpansion.first.size();
   std::vector<double> coeff(dim,0.0);
   for(size_t i=0; i<dim; i++){
      coeff[i] = detexpansion.second[i];
      if(abs(coeff[i])<1.e-10) continue;
      std::cout << " i=" << i << " det=" << detexpansion.first[i]
         << " coeff=" << coeff[i] << std::endl;
   }
   double Sdiag = fock::coeff_entropy(coeff);
   double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2); 
   std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;
   return Sdiag;
}

csfspace fock::get_csf_space(const int k, const int n, const int ts){
   const bool debug = false;
   std::cout << "fock::get_csf_space (k,n,ts)=" << k << "," << n << "," << ts << std::endl;
   assert(k <= 32); // no. of spatial orbitals should be smaller than 32
   csfspace space;
   csfstate vacuum(k);
   space.push_back(vacuum);
   // gradually construct FCI space
   for(int i=k-1; i>=0; i--){
      if(debug) std::cout << "\ni=" << i << std::endl;
      csfspace space_new;
      int kres = i;
      size_t idx = 0;
      for(const auto& state : space){
         if(debug) std::cout << " idx=" << idx << " state=" << state.repr << std::endl;
         idx += 1;
         // produce new state
         for(int d=0; d<4; d++){
            auto state0 = state;
            state0.repr[2*i] = d%2;
            state0.repr[2*i+1] = d/2;
            int nelec = state0.nelec();
            int twos = state0.twos();
            // check whether state is acceptible
            if(twos >= 0 && (nelec <= n && nelec+2*kres >= n)){
               int nres = n - nelec;
               int tsmin = (nres%2==0)? 0 : 1;
               int tsmax = (nres<=kres)? nres : 2*kres-nres; 
               if(debug){ 
                  std::cout << "  state0=" << state0
                     << " twos,tsmin,tsmax,ts="
                     << twos << "," << tsmin << "," 
                     << tsmax << "," << ts 
                     << std::endl;
               }
               // twos + {tsmin,...,tsmax} => ts
               for(int tsval=tsmin; tsval<=tsmax; tsval+=2){
                  if(debug){
                     std::cout << "    tsval=" << tsval
                        << " spin_triangle(twos,tsval,ts)=" << spin_triangle(twos,tsval,ts)
                        << std::endl;
                  }
                  if(spin_triangle(twos,tsval,ts)){ 
                     space_new.push_back(state0);
                     break;
                  }
               } // tsval
            } // ifsym
         } // d
      } // iorb
      space = std::move(space_new);
   } 
   // check dimension of the CSF space
   assert(space.size() == dim_csf_space(k,n,ts)); 
   return space;
}
