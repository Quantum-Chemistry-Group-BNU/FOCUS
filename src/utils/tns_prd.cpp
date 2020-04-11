#include "../core/linalg.h"
#include "tns.h"
#include <map>
#include <tuple>
#include <cmath>

using namespace std;
using namespace fock;
using namespace linalg;
using namespace tns;

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
      
pair<int,double> product_space::projection(const vector<vector<double>>& vs,
				           const double thresh){
   bool debug = true;
   if(debug) cout << "\nproduct_space::projection thresh="
	          << thresh << endl;
   // collect states with the same symmetry (N,NA)
   map<pair<int,int>,vector<int>> qsecA; 
   for(int i=0; i<dimA; i++){
      int ne = spaceA[i].nelec();
      int ne_a = spaceA[i].nelec_a();
      qsecA[make_pair(ne,ne_a)].push_back(i); 
   }
   // loop over symmetry sectors
   int idx = 0;
   int dimAr = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(auto& pr : qsecA){
      const pair<int,int>& sym = pr.first;
      vector<int>& idxA = pr.second;
      int dimAs = idxA.size();
      if(debug){
         cout << "idx=" << idx << " symA=" << sym.first << ":" << sym.second 
              << " dim=" << dimAs << endl;
	 for(int i=0; i<dimAs; i++){
	    cout << " i=" << i << " : " << spaceA[idxA[i]].to_string2() << endl;
	    /*
	    // complementary part
	    for(auto pr : rowA[idxA[i]]){
	       int j = pr.first;
	       cout << "  j=" << j << " : " << spaceB[j] << endl;
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
               int i = pib.second;
               vlr(ia,ib) = vs[iroot][i];
            }
         }
         rhol += dgemm("N","N",vlr,vlr.transpose());
      }
      rhol *= 1.0/nroots;
      vector<double> eig(dimAs);
      eigen_solver(rhol,eig,1);
      // compute entropy
      for(int i=0; i<dimAs; i++){ 
	 if(eig[i]>thresh){
            if(debug){ 
	       cout << " i=" << i << " eig=" << eig[i] << endl;
	       cout << "      v=";
	       for(int j=0; j<dimAs; j++) cout << rhol(j,i) << " ";
	       cout << endl;
	    }
 	    SvN += -eig[i]*log2(eig[i]);
            sum += eig[i];
	    dimAr += 1;
	 }
      }
      idx++;
      if(debug) cout << " sum=" << sum << " dimAr=" << dimAr << endl;
   } // sym sectors

   if(debug){
      cout << "dim=" << dim << " dimA=" << dimA << " dimB=" << dimB
           << " thresh=" << thresh << " dimAr=" << dimAr 
           << " SvN=" << SvN << endl;
      // also check qsecB
      map<pair<int,int>,vector<int>> qsecB; 
      for(int i=0; i<dimB; i++){
         int ne = spaceB[i].nelec();
         int ne_a = spaceB[i].nelec_a();
         qsecB[make_pair(ne,ne_a)].push_back(i); 
      }
      for(auto& pr : qsecB){
         const pair<int,int>& sym = pr.first;
         vector<int>& idxB = pr.second;
         int dimBs = idxB.size();
         cout << "symB=" << sym.first << ":" << sym.second 
              << " dim=" << dimBs << endl;
      }
   }
   return make_pair(dimAr,SvN); 
}
