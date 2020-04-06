#include "../core/tools.h"
#include "tns.h"
#include <numeric>

using namespace std;
using namespace fock;
using namespace linalg;
using namespace tns;

// brute-force  
void tns::ordering_bf(const onspace& space, 
		      const vector<vector<double>>& vs,
		      vector<int>& order){
   cout << "\ntns::ordering_bf" << endl;
   int k = space[0].size();
   vector<int> s(k/2);
   iota(s.begin(),s.end(),0);
   int idx = 0;
   double Smin = 1.e10;
   do{
       // generate permutation of spatial sites
       tools::perm pm(k/2);
       for(int i=0; i<pm.size; i++){
	  pm.image[i] = s[i];
       }
       auto image2 = pm.to_image2();
       // update basis vector and signs 
       onspace sci_space2;
       vector<int> sgns;
       for(const auto& state : space){
          sci_space2.push_back(state.permute(image2));
          sgns.push_back(state.permute_sgn(image2));
       }
       int dim = space.size();
       int nroot = vs.size();
       vector<vector<double>> vs2(nroot);
       for(int i=0; i<nroot; i++){
	  vs2[i].resize(dim);
          transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
    	            [](const double& x, const int& y){ return x*y; });
       }
       // bipartition 
       vector<int> bdims; 
       double SvN = 0.0;
       for(int pos=1; pos<pm.size; pos++){
          tns::product_space pspace2;
          pspace2.get_pspace(sci_space2, 2*pos);
          auto pr = pspace2.projection(vs2);
	  SvN += pr.second;
	  bdims.push_back(pr.first);
       }
       // check
       cout << "idx=" << idx << " : ";	   
       for(int i : s) cout << i << " ";
       cout << "SvN=" << defaultfloat << setprecision(12) << SvN 
	    << " bdims=";
       for(int i : bdims) cout << i << ",";  
       cout << endl;
       // save data
       if(SvN < Smin){
	  Smin = SvN;
	  order = s;
       }
       idx++;
       exit(1);
   }while(next_permutation(s.begin(), s.end()));

}

void tns::ordering_ga(const onspace& space, 
		      const vector<vector<double>>& vs,
		      vector<int>& order){
   cout << "\ntns::ordering_ga" << endl;

   ordering_bf(space, vs, order);
   for(int i : order) cout << i << " ";
   cout << endl;

/*
   int k = space[0].size();
   int dim = space.size();
   
   // bipartition 
   int nroot = vs.size();

   vector<int> s(k/2);
   iota(s.begin(),s.end(),0);
   int idx = 0;
   do{
       tools::perm pm(k/2);
       for(int i=0; i<pm.size; i++){
	  pm.image[i] = s[i];
       }
       auto image2 = pm.to_image2();
       
       onspace sci_space2;
       vector<int> sgns;
       for(const auto& state : space){
          sci_space2.push_back(state.permute(image2));
          sgns.push_back(state.permute_sgn(image2));
       }
    
       vector<vector<double>> vs2(nroot);
       for(int i=0; i<nroot; i++){
	  vs2[i].resize(dim);
          transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
    	            [](const double& x, const int& y){ return x*y; });
       }
      
       vector<int> bdims; 
       double SvN = 0.0;
       for(int pos=1; pos<pm.size; pos++){
          tns::product_space pspace2;
          pspace2.get_pspace(sci_space2, 2*pos);
          auto pr = pspace2.projection(vs2,1.e-6);
	  SvN += pr.second;
	  bdims.push_back(pr.first);
       }

       // check
       cout << "idx=" << idx << " : ";	   
       for(int i : s) cout << i << " ";
       cout << "SvN=" << defaultfloat << setprecision(12) << SvN 
	    << " bdims=";
       for(int i : bdims) cout << i << ",";  
       cout << endl;
       idx++;
   }while(next_permutation(s.begin(), s.end()));

*/
/*
   // cutoff to managable size to increase efficiency 

   tns::product_space pspace;
   pspace.get_pspace(sci_space, pos);
   pspace.projection(vs);
  
   cout << pm << endl;
   pm.shuffle();
   cout << pm << endl;

*/

}
