#ifndef SCI_UTIL_H
#define SCI_UTIL_H

#include <unordered_set>
#include <functional>
#include <vector>
#include <map>
#include "../core/analysis.h"
#include "../core/hamiltonian.h"
#include "../core/integral.h"
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"  // get_ortho_basis
#include "../io/input.h" // schedule
#include "fci_util.h"

namespace sci{

struct heatbath_table{
public: 
   template <typename Tm>
   heatbath_table(const integral::two_body<Tm>& int2e,
		  const integral::one_body<Tm>& int1e){
      const bool debug = false;
      auto t0 = tools::get_time();
      std::cout << "\nheatbath_table::heatbath_table" << std::endl;
      int k = int2e.sorb;
      sorb = k;
      eri4.resize(k*(k-1)/2);
      for(int i=0; i<k; i++){
         for(int j=0; j<i; j++){
            int ij = i*(i-1)/2+j;
            for(int p=0; p<k; p++){
               if(p == i || p == j) continue; // guarantee true double excitations
               for(int q=0; q<p; q++){
                  if(q == i || q == j) continue;
                  double mag = abs(int2e.get(i,j,p,q)); // |<ij||pq>|
                  if(mag > thresh) eri4[ij].insert(std::make_pair(mag,p*(p-1)/2+q));
               } // q
            } // p
         } // j
      } // i
      eri3.resize(k*(k+1)/2);
      for(int i=0; i<k; i++){
         for(int j=0; j<=i; j++){
            int ij = i*(i+1)/2+j;
            eri3[ij].resize(k+1);
            for(int p=0; p<k; p++){
               // |<ip||jp>| 
               eri3[ij][p] = abs(int2e.get(i,p,j,p));
            } // p
            eri3[ij][k] = abs(int1e.get(i,j));
         } // j
      } // i
      if(debug){
	 std::cout << std::defaultfloat << std::setprecision(12);
         for(int ij=0; ij<k*(k-1)/2; ij++){
            auto pr = tools::inverse_pair0(ij);
            int i = pr.first, j = pr.second;
            std::cout << "ij=" << ij << " i,j=" << i << "," << j 
                      << " eri4[ij] size : " << eri4[ij].size() << std::endl;
            for(const auto& p : eri4[ij]){
               if(p.first > 1.e-2){
                  auto pq = tools::inverse_pair0(p.second);
        	  std::cout << "   val=" << p.first 
           	            << " -> p,q=" << pq.first << "," << pq.second 
           	            << std::endl;
               }
            }
         }
      } // debug
      auto t1 = tools::get_time();
      std::cout << "timing for heatbath_table::heatbath_table : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
public:
   int sorb;
   // cut-off value 
   double thresh = 1.e-14; 
   // sorted by magnitude Iij[kl]=<ij||kl> (i>j,k>l)
   std::vector<std::multimap<float,int,std::greater<float>>> eri4; 
   // Iik[j]={<ij||kj>(i>=k),hik} for fast estimation of singles
   std::vector<std::vector<float>> eri3; 
};

// expand variational subspace
void expand_varSpace(fock::onspace& space, 
		     std::unordered_set<fock::onstate>& varSpace, 
		     const heatbath_table& hbtab, 
		     const std::vector<double>& cmax, 
		     const double eps1,
		     const bool flip);

// prepare intial solution
template <typename Tm>
void get_initial(std::vector<double>& es,
		 linalg::matrix<Tm>& vs,
		 fock::onspace& space,
	         std::unordered_set<fock::onstate>& varSpace,
		 const heatbath_table& hbtab, 
		 const input::schedule& schd, 
	         const integral::two_body<Tm>& int2e,
	         const integral::one_body<Tm>& int1e,
	         const double ecore){
   std::cout << "\nsci::get_initial" << std::endl;
   // space = {|Di>}
   int k = int1e.sorb;
   for(const auto& det : schd.det_seeds){
      fock::onstate state(k); // convert det to onstate
      for(int i : det) state[i] = 1;
      // search first
      auto search = varSpace.find(state);
      if(search == varSpace.end()){
	 varSpace.insert(state);
	 space.push_back(state);
      }
      // flip determinant 
      if(schd.flip){
         auto state1 = state.flip();
         auto search1 = varSpace.find(state1);
         if(search1 == varSpace.end()){
            space.push_back(state1);
            varSpace.insert(state1);
         }
      }
   }
   // print
   std::cout << "energies for reference states:" << std::endl;
   std::cout << std::defaultfloat << std::setprecision(12);
   int nsub = space.size();
   for(int i=0; i<nsub; i++){
      std::cout << "i = " << i << " state = " << space[i]
	   << " e = " << fock::get_Hii(space[i],int2e,int1e)+ecore 
	   << std::endl;
   }
   // selected CISD space
   double eps1 = schd.eps0;
   std::vector<double> cmax(nsub,1.0);
   expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.flip);
   nsub = space.size();
   // set up initial states
   if(schd.nroots > nsub){
      std::cout << "error in sci::ci_solver: subspace is too small!" << std::endl;
      exit(1);
   }
   linalg::matrix<Tm> H = fock::get_Ham(space, int2e, int1e, ecore);
   std::vector<double> esol(nsub);
   linalg::matrix<Tm> vsol;
   eig_solver(H, esol, vsol);
   // save
   int neig = schd.nroots;
   es.resize(neig);
   vs.resize(nsub, neig);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nsub; i++){
	 vs(i,j) = vsol(i,j);
      }
      es[j] = esol[j];
   }
   // print
   std::cout << std::setprecision(12);
   for(int i=0; i<neig; i++){
      std::cout << "i = " << i << " e = " << es[i] << std::endl; 
   }
}

// truncate CI coefficients for later use in initializing TNS
template <typename Tm>
void ci_truncate(fock::onspace& space,
		 std::vector<std::vector<Tm>>& vs,
		 const int maxdets,
		 const bool ifortho=false){
   const bool debug = true;
   std::cout << "\nsci::ci_truncate maxdets=" << maxdets 
	     << " ifortho=" << ifortho << std::endl;
   int nsub = space.size();
   int neig = vs.size();
   int nred = std::min(nsub,maxdets);
   std::cout << "reduction from " << nsub << " to " << nred << " dets" << std::endl;
   // select important basis
   std::vector<double> cmax(nsub,0.0);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nsub; i++){
	 cmax[i] += std::norm(vs[j][i]);
      }
   }
   auto index = tools::sort_index(cmax, 1); 
   // orthogonalization if required
   std::vector<Tm> vtmp(nred*neig);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nred; i++){
	 vtmp[i+nred*j] = vs[j][index[i]];
      }
   }
   if(ifortho){
      int nindp = linalg::get_ortho_basis(nred,neig,vtmp);
      if(nindp != neig){
	 std::cout << "error: thresh is too large for ci_truncate!" << std::endl;
         std::cout << "nindp,neig=" << nindp << "," << neig << std::endl;
         exit(1);
      }
   }
   // copy basis and coefficients
   fock::onspace space2(nred);
   for(int i=0; i<nred; i++){
      space2[i] = space[index[i]];	
   }
   std::vector<std::vector<Tm>> vs2(neig);
   for(int j=0; j<neig; j++){
      vs2[j].resize(nred);
      std::copy(&vtmp[nred*j],&vtmp[nred*j]+nred,vs2[j].begin());
   }
   // check the quality of truncated states
   if(debug){
      for(int j=0; j<neig; j++){
         std::vector<Tm> vec(nred);
         for(int i=0; i<nred; i++){
            vec[i] = vs[j][index[i]];
         }
	 auto ova = linalg::xdot(nred, vs2[j].data(), vec.data());
         std::cout << "iroot=" << j << " ova=" 
                   << std::setprecision(12) << ova << std::endl;
      }
   }
   // replace (space,vs) by the reduced counterpart 
   space = space2;
   vs = vs2;
}

} // sci

#endif
