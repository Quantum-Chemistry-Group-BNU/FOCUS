#include "../settings/global.h"
#include "../core/linalg.h"
#include "tns_pspace.h"
#include <cassert>
#include <cmath>
#include <map>

using namespace std;
using namespace fock;
using namespace linalg;
using namespace tns;

// renormalized states from determinants
void renorm_sector::print(const string msg, const int level){
   cout << "renorm_sector: " << msg << " qsym=" << sym 
        << " shape=" << coeff.rows() << "," << coeff.cols() << endl; 
   if(level >= 1){
      for(int i=0; i<space.size(); i++){
         cout << " idx=" << i << " state=" << space[i].to_string2() << endl;
      }
   }
   if(level >= 2) coeff.print("coeff");
}

// rbasis for leaves
renorm_basis tns::get_rbasis_phys(){
   renorm_basis rbasis(4);
   for(int i=0; i<4; i++){
      rbasis[i].sym = phys_sym[i]; 
      rbasis[i].space.push_back(phys_space[i]);
      rbasis[i].coeff = linalg::identity_matrix(1);
   }
   return rbasis;
}

// factorize the space into products of two spaces 
void product_space::get_pspace(const onspace& space, const int n){
   bool debug = false;
   if(debug) cout << "\nproduct_space::get_pspace" << endl;
   int udxA = 0, udxB = 0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   dim = space.size();
   for(int i=0; i<dim; i++){
      onstate strA = space[i].get_before(n);
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
	 spaceA.push_back(strA);
         auto pr = umapA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 rowA.resize(udxA); // reserve additional space for the new row
      };
      // odd bits
      onstate strB = space[i].get_after(n);
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
	 spaceB.push_back(strB);
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 colB.resize(udxB);
      }
      rowA[itA->second].emplace_back(itB->second,i);
      colB[itB->second].emplace_back(itA->second,i);
   }
   assert(udxA == spaceA.size());
   assert(udxB == spaceB.size());
   dimA = udxA;
   dimB = udxB;
   if(debug){
      cout << "dim=" << dim << " dimA=" << dimA << " dimB=" << dimB << endl;
      for(int i=0; i<dimA; i++){
         cout << "ia=" << i << " : " << spaceA[i].to_string2() << endl;
      }
      for(int i=0; i<dimB; i++){
         cout << "ib=" << i << " : " << spaceB[i].to_string2() << endl;
      }
   }
}

// left projection
pair<int,double> product_space::projection(const vector<vector<double>>& vs,
				           const double thresh){
   double thresh_vcoeff = 1.e-2;
   bool debug = true;
   if(debug) cout << "\nproduct_space::projection thresh="
	          << thresh << endl;
   // collect states with the same symmetry (N,NA)
   map<qsym,vector<int>> qsecA; 
   for(int i=0; i<dimA; i++){
      int ne = spaceA[i].nelec();
      int ne_a = spaceA[i].nelec_a();
      qsecA[qsym(ne,ne_a)].push_back(i); 
   }
   // loop over symmetry sectors
   int idx = 0;
   int dimAc = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(auto it = qsecA.cbegin(); it != qsecA.cend(); ++it){
      auto& sym = it->first;
      auto& idxA = it->second;
      int dimAs = idxA.size();
      if(debug){
         cout << "\nidx=" << idx << " symA(Ne,Na)=" << sym
              << " dim=" << dimAs << endl;
	 for(int i=0; i<dimAs; i++){
	    cout << " ia=" << i << " : " << spaceA[idxA[i]].to_string2() << endl;
	    /*
	    // complementary part
	    for(auto pr : rowA[idxA[i]]){
	       int j = pr.first;
	       cout << "  ib=" << j << " : " << spaceB[j] << endl;
	    }
	    */
	 }
      }
      // build reduced density matrix
      matrix rhol(dimAs,dimAs);
      int nroots = vs.size();
      for(int iroot = 0; iroot<nroots; iroot++){
         // vlr for sym sector
         matrix vlr(dimAs,dimB);
         for(int ia=0; ia<dimAs; ia++){
            for(const auto& pib : rowA[idxA[ia]]){
               int ib = pib.first;
               int id = pib.second;
               vlr(ia,ib) = vs[iroot][id];
            }
         }
         rhol += dgemm("N","N",vlr,vlr.T());
      }
      rhol *= 1.0/nroots;
      vector<double> eig(dimAs);
      eigen_solver(rhol,eig,1);
      // compute entropy
      int dimAi = 0;
      double sumi = 0.0;
      for(int i=0; i<dimAs; i++){ 
	 if(eig[i]>thresh){
            if(debug){ 
	       cout << " i=" << i
		    << " eig=" << scientific << eig[i] << endl;
	       for(int j=0; j<dimAs; j++){
		  if(abs(rhol(j,i))>thresh_vcoeff){
		     cout << "     " << j << " " << spaceA[idxA[j]].to_string2() 
			  << " : " << rhol(j,i) << endl; 
		  }
	       }
	    }
 	    SvN += -eig[i]*log2(eig[i]);
            sumi += eig[i];
	    dimAi += 1;
	 }
      }
      sum += sumi;
      dimAc += dimAi;
      idx++;
      if(debug) cout << " dimAs=" << dimAs 
	       	     << " sumi=" << defaultfloat << sumi
	             << " dimAi=" << dimAi 
		     << " sum=" << sum
	             << " dimAc=" << dimAc 
		     << endl;
   } // sym sectors
   if(!debug){
      cout << "\ndim=" << dim << " dimA=" << dimA << " dimB=" << dimB
           << " thresh=" << thresh << " dimAc=" << dimAc 
           << " SvN=" << SvN << endl;
   }
   if(debug){
      cout << endl;
      // also check qsecB
      map<qsym,vector<int>> qsecB; 
      for(int i=0; i<dimB; i++){
         int ne = spaceB[i].nelec();
         int ne_a = spaceB[i].nelec_a();
         qsecB[qsym(ne,ne_a)].push_back(i); 
      }
      for(auto& pr : qsecB){
         auto& sym = pr.first;
         auto& idxB = pr.second;
         int dimBs = idxB.size();
         cout << "symB=" << sym << " dim=" << dimBs << endl;
      }
   }
   return make_pair(dimAc,SvN); 
}
     
// right projection
renorm_basis product_space::right_projection(const vector<vector<double>>& vs,
				 	     const double thresh,
					     const bool debug){
   auto t0 = global::get_time();
   int debug_level = debug;
   if(debug_level) cout << "\nproduct_space::right_projection thresh="
                        << scientific << thresh << endl;
   renorm_basis rbasis;				     
   // 1. collect states with the same symmetry (N,NA)
   map<qsym,vector<int>> qsecB; // sym -> indices in spaceB
   map<qsym,map<int,int>> qmapA; // index in spaceA to idxA
   map<qsym,vector<tuple<int,int,int>>> qlst;
   for(int ib=0; ib<dimB; ib++){
      int ne = spaceB[ib].nelec();
      int ne_a = spaceB[ib].nelec_a();
      qsym symB(ne,ne_a);
      qsecB[symB].push_back(ib);
      for(const auto& pia : colB[ib]){
	 int ia = pia.first;
	 int idet = pia.second;
	 // search unique
	 auto it = qmapA[symB].find(ia);
         if(it == qmapA[symB].end()){
            qmapA[symB].insert({ia,qmapA[symB].size()});
         };
	 int idxB = qsecB[symB].size()-1;
	 int idxA = qmapA[symB][ia];
	 qlst[symB].push_back(make_tuple(idxB,idxA,idet));
      }
   }
   // 2. loop over symmetry sectors to compute renormalized states
   int idx = 0, dimBc = 0, dB = 0, dA = 0;
   double sum = 0.0, SvN = 0.0;
   for(auto it = qsecB.cbegin(); it != qsecB.cend(); ++it){
      auto& symB = it->first;
      auto& idxB = it->second;
      int dimBs = idxB.size(); 
      int dimAs = qmapA[symB].size();
      dB += dimBs;
      dA += dimAs;
      if(debug_level){
         cout << "idx=" << idx << " symB(Ne,Na)=(" << symB 
              << " dimBs=" << dimBs
	      << " dimAs=" << qmapA[symB].size() 
	      << endl;
	 if(debug_level>1){
	    for(int i=0; i<dimBs; i++){
	       cout << " ib=" << i << " : " 
	            << spaceB[idxB[i]].to_string2() << endl;
	    }
	 }
      }
      int nroots = vs.size();
      vector<double> sig;
      matrix vrl, u, vt;
      if(dimBs > dimAs*nroots){
         // compute renormalized basis using SVD
	 sig.resize(dimAs*nroots);
         vrl.resize(dimBs,dimAs*nroots);
         for(int iroot = 0; iroot<nroots; iroot++){
            for(const auto& t : qlst[symB]){
               int ib = get<0>(t);
               int ia = get<1>(t);
               int id = get<2>(t);
               vrl(ib,ia+dimAs*iroot) = vs[iroot][id];
            }
         }
         vrl *= 1.0/sqrt(nroots);
         svd_solver(vrl,sig,u,vt,1);
         transform(sig.begin(),sig.end(),sig.begin(),
		   [](const double& x){ return x*x; });
      }else{
	 // compute renormalized basis using eigen decomposition
	 sig.resize(dimBs);
	 vrl.resize(dimBs,dimAs);
	 u.resize(dimBs,dimBs);
         for(int iroot = 0; iroot<nroots; iroot++){
            for(const auto& t : qlst[symB]){
               int ib = get<0>(t);
               int ia = get<1>(t);
               int id = get<2>(t);
               vrl(ib,ia) = vs[iroot][id];
            }
	    u += dgemm("N","N",vrl,vrl.T()); 
         }
         u *= 1.0/nroots;
         eigen_solver(u,sig,1);
      } 
      // select important renormalized states
      int dimBi = 0;
      double sumi = 0.0;
      for(int i=0; i<sig.size(); i++){
	 if(sig[i]>thresh){
	    dimBi += 1;
            sumi += sig[i];
 	    SvN += -sig[i]*log2(sig[i]); // compute entanglement entropy
            if(debug_level>1){ 
	       cout << " i=" << i
		    << " sig2=" << scientific << sig[i] << endl;
	       double thresh_coeff = -1.e-2;
	       for(int j=0; j<dimBs; j++){
		  if(abs(u(j,i))>thresh_coeff){
		     cout << "     " << j << " " << spaceB[idxB[j]].to_string2() 
			  << " : " << u(j,i) << endl; 
		  }
	       }
	    }
	 }
      } // i
      dimBc += dimBi;
      sum += sumi;
      if(debug_level) cout << " dimBs=" << dimBs << " dimBi=" << dimBi 
		           << " sumi=" << sumi << " sum=" << sum << endl;
      // save sites
      if(dimBi > 0){
	 renorm_sector rsec;
	 rsec.sym = symB;
	 rsec.space.resize(dimBs);
	 for(int i=0; i<dimBs; i++){
            rsec.space[i] = spaceB[idxB[i]];
	 }
	 rsec.coeff.resize(dimBs,dimBi);
         copy(u.data(),u.data()+dimBs*dimBi,rsec.coeff.data());
	 rbasis.push_back(rsec);
      }
      idx++;
   } // sym sectors
   assert(dA == dimA && dB == dimB);
   if(debug_level){
      cout << "dim=" << dim << " dimA=" << dimA << " dimB=" << dimB
           << " dimBc=" << dimBc << " sum=" << sum << " SvN=" << SvN << endl;
      auto t1 = global::get_time();
      cout << "timing for product_space::right_projection : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
   return rbasis;
}
