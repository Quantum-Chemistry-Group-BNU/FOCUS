#include "../core/linalg.h"
#include "tns_pspace.h"
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
   double thresh_vcoeff = 1.e-2;
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
   /* 
   // print only
   int idx = 0;
   for(auto it = qsecA.crbegin(); it != qsecA.crend(); ++it){
      const pair<int,int>& sym = it->first;
      const vector<int>& idxA = it->second;
      int dimAs = idxA.size();
      if(debug){
         cout << "\nidx=" << idx << " symA(Ne,Na)=(" 
	      << sym.first << "," << sym.second << ")"
              << " dim=" << dimAs << endl;
      }
      cout << " ia=0 : " << spaceA[idxA[0]].to_string2() << endl;
      idx++;
   }
   */
   // loop over symmetry sectors
   int idx = 0;
   int dimAc = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(auto it = qsecA.crbegin(); it != qsecA.crend(); ++it){
      const pair<int,int>& sym = it->first;
      const vector<int>& idxA = it->second;
      int dimAs = idxA.size();
      if(debug){
         cout << "\nidx=" << idx << " symA(Ne,Na)=(" 
	      << sym.first << "," << sym.second << ")"
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
   return make_pair(dimAc,SvN); 
}
