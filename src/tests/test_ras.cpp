#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../settings/global.h"
#include "../io/input.h"
#include "../utils/fci.h"
#include "../utils/fci_rdm.h"
#include "../utils/sci.h"
#include "../utils/tns.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

onspace direct_product(const onspace& space1,
		       const onspace& space2){
   int dim1 = space1.size();
   int dim2 = space2.size();
   onspace space12;
   for(int i=0; i<dim1; i++){
      for(int j=0; j<dim2; j++){
         space12.push_back(space1[i].join(space2[j]));
      }
   }
   return space12;
}

using qspace = map<pair<int,int>,onspace>;
qspace space_combine(const qspace& qspace1,
			const qspace& qspace2){
   qspace qspace12;
   for(const auto& p1 : qspace1){
      auto sym1 = p1.first;
      auto space1 = p1.second; 
      for(const auto& p2 : qspace2){
         auto sym2 = p2.first;
         auto space2 = p2.second;
	 // combine
	 int ne = sym1.first+sym2.first;
	 int na = sym1.second+sym2.second;
	 auto sym12 = make_pair(ne,na);
         auto space12 = direct_product(space1,space2);
	 auto search = qspace12.find(sym12);
	 if(search == qspace12.end()){
	    qspace12.insert({sym12,space12});
	 }else{
	    std::copy(space12.begin(), space12.end(), 
		      std::back_inserter(search->second));
	 }
      }
   }
   return qspace12;
}

int tests::test_ras(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_ras" << endl;
   cout << global::line_separator << endl;	
  
   // read input
   string fname = "input.dat";
   input::schedule schd;
   input::read_input(schd,fname);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, 
		   	  schd.integral_file,
		    	  schd.integral_type);
   
   vector<int> orderK;
   tns::ordering_fiedler(int2e.K, orderK);

   // generate restricted active space for Fe
   int ks = 5;
   qspace space_Fe;
   //vector<pair<int,int>> sym_Fe({{5,5},{5,0}});
   /*
   vector<pair<int,int>> sym_Fe({{5,5},{5,0},	// 0p Fe(III)
		   		 {5,4},{5,1},
				 {5,3},{5,2}});
   */
   // local fock space for Fe
   vector<pair<int,int>> sym_Fe;
   for(int ne=0; ne<=2*ks; ne++){
      for(int na=max(0,ne-ks); na<=min(ks,ne); na++){
	 //if(ne <= 3 || ne >= 7) continue;
	 if(ne != 5) continue;
         sym_Fe.push_back(make_pair(ne,na));
      }
   }
   /*
   vector<pair<int,int>> sym_Fe({{5,5},{5,0},	// 0p Fe(III
	   			 {5,4},{5,1},
        	   	         {6,5},{6,1}});	// 1p Fe(II)
   */
   /*
   vector<pair<int,int>> sym_Fe({{5,5},{5,0},	// 0p Fe(III
        	   	         {6,5},{6,1},	// 1p Fe(II)
        		         {7,5},{7,2}}); // 2p Fe(I)
   */ 
   int nstate = 0;
   for(const auto& sym : sym_Fe){
      int ne = sym.first;
      int na = sym.second;
      auto space = get_fci_space(ks,na,ne-na);
      nstate += space.size();
      space_Fe.insert({sym,space});
      cout << "SYM=(" << ne << "," << na << ")" 
	   << " DIM=" << space.size() 
	   << " DIMC=" << nstate << endl;
      int idx = 0;
      for(const auto& state : space){
	 cout << idx << " " << state.to_string2() << endl;
	 idx++;
      }
   }
   cout << "\nno. of Fe-states = " << nstate << endl;

   // generate restricted active space for S
   qspace space_S;
   // fe2s2
   ks = 10; 
   // fe4s4 
   //ks = 16;
   //vector<pair<int,int>> sym_S({{2*ks,ks}}); // 0h
   vector<pair<int,int>> sym_S({{2*ks,ks},		   // 0h
		   	        {2*ks-1,ks},{2*ks-1,ks-1}} // 1h
				);
   /*
   vector<pair<int,int>> sym_S({{2*ks,ks},		   // 0h
		   	        //{2*ks-1,ks},{2*ks-1,ks-1}, // 1h
				{2*ks-2,ks-1}} // ab
				);
   */ 
   /*
   vector<pair<int,int>> sym_S({{2*ks,ks},		   // 0h
		   	        {2*ks-1,ks},{2*ks-1,ks-1}, // 1h
				{2*ks-2,ks},{2*ks-2,ks-1},{2*ks-2,ks-2}, // 2h
			 	{2*ks-3,ks-1},{2*ks-3,ks-2}} // abb,aab
				);
   */
   nstate = 0;
   for(const auto& sym : sym_S){
      int ne = sym.first;
      int na = sym.second;
      auto space = get_fci_space(ks,na,ne-na);
      space_S.insert({sym,space});
      cout << "\nSYM=(" << ne << "," << na << ")" 
	   << " DIM=" << space.size() << endl;
      int idx = 0;
      for(const auto& state : space){
	 cout << idx << " " << state.to_string2() << endl;
	 idx++;
      }
      nstate += space.size();
   }
   cout << "\nno. of S-states = " << nstate << endl;

   // fe2s2
   int ne = 30, na = 15;
   //int ne = 10, na = 5;
   using qsym = pair<int,int>;
   vector<tuple<qsym,qsym,qsym>> sym_sectors;
   for(const auto& p1: space_Fe){
      auto sym1 = p1.first;
      int ne1 = sym1.first;
      int na1 = sym1.second;
      for(const auto& p2 : space_Fe){
         auto sym2 = p2.first;
         int ne2 = sym2.first;
         int na2 = sym2.second;
	 for(const auto& p3 : space_S){
            auto sym3 = p3.first;		
            int ne3 = sym3.first;
            int na3 = sym3.second;
	    if(ne1+ne2+ne3 == ne &&
	       na1+na2+na3 == na){
	       sym_sectors.push_back(make_tuple(sym1,sym2,sym3));
	    } 
	 }
      }	 
   }
   onspace space;
   int idx = 0, dc = 0;
   for(const auto& t : sym_sectors){
      auto s1 = get<0>(t); int d1 = space_Fe[s1].size();
      auto s2 = get<1>(t); int d2 = space_Fe[s2].size();
      auto s3 = get<2>(t); int d3 = space_S[s3].size();
      int d = d1*d2*d3;
      dc += d;
      cout << "(" << s1.first << "," << s1.second << "): " << d1 << " " 
	   << "(" << s2.first << "," << s2.second << "): " << d2 << " "
	   << "(" << s3.first << "," << s3.second << "): " << d3 << " "
	   << " dim=" << d << " dimc=" << dc
	   << endl;
      auto dp12 = direct_product(space_Fe[s1],space_Fe[s2]);
      auto dpFe2S2 = direct_product(dp12,space_S[s3]);
      assert(d == dpFe2S2.size());
      std::copy(dpFe2S2.begin(), dpFe2S2.end(), 
		std::back_inserter(space));

      idx++;
   }
   // map to the order of integrals 
   vector<int> loc({2,3,4,5,6,
        	    13,14,15,16,17,
        	    0,1,7,8,9,10,11,12,18,19});
   
   /* 
   // Better combination of sectors !!!
   int ne = 54, na = 27;
   using qsym = pair<int,int>;
   vector<tuple<qsym,qsym,qsym,qsym,qsym>> sym_sectors;
   for(const auto& p1: space_Fe){
      auto sym1 = p1.first;
      int ne1 = sym1.first;
      int na1 = sym1.second;
      for(const auto& p2 : space_Fe){
         auto sym2 = p2.first;
         int ne2 = sym2.first;
         int na2 = sym2.second;
	 for(const auto& p3 : space_Fe){
            auto sym3 = p3.first;		
            int ne3 = sym3.first;
            int na3 = sym3.second;
	    for(const auto& p4 : space_Fe){
               auto sym4 = p4.first;		
      	       int ne4 = sym4.first;
      	       int na4 = sym4.second;
	       for(const auto& p5 : space_S){
                  auto sym5 = p5.first;	
      	          int ne5 = sym5.first;
      	          int na5 = sym5.second;
		  if(ne1+ne2+ne3+ne4+ne5 == ne &&
		     na1+na2+na3+na4+na5 == na){
		     sym_sectors.push_back(make_tuple(sym1,sym2,sym3,sym4,sym5));
		  } 
	       }
	    }
	 }
      }	 
   }
   cout << sym_sectors.size() << endl;
   onspace space;
   int idx = 0, dc = 0;
   for(const auto& t : sym_sectors){
      auto s1 = get<0>(t); int d1 = space_Fe[s1].size();
      auto s2 = get<1>(t); int d2 = space_Fe[s2].size();
      auto s3 = get<2>(t); int d3 = space_Fe[s3].size();
      auto s4 = get<3>(t); int d4 = space_Fe[s4].size();
      auto s5 = get<4>(t); int d5 = space_S[s5].size();
      int d = d1*d2*d3*d4*d5;
      dc += d;
      cout << "(" << s1.first << "," << s1.second << "): " << d1 << " " 
	   << "(" << s2.first << "," << s2.second << "): " << d2 << " "
	   << "(" << s3.first << "," << s3.second << "): " << d3 << " "
	   << "(" << s4.first << "," << s4.second << "): " << d4 << " "
	   << "(" << s5.first << "," << s5.second << "): " << d5 << " "
	   << " dim=" << d << " dimc=" << dc
	   << endl;


      auto dp12 = direct_product(space_Fe[s1],space_Fe[s2]);
      auto dp123 = direct_product(dp12,space_Fe[s3]);
      auto dp1234 = direct_product(dp123,space_Fe[s4]);
      auto dpFe4S4 = direct_product(dp1234,space_S[s5]);
      assert(d == dpFe4S4.size());
      std::copy(dpFe4S4.begin(), dpFe4S4.end(), 
		std::back_inserter(space));

      idx++;
   }
   vector<int> loc({2,3,4,5,6,
		    7,8,9,10,11,
		    24,25,26,27,28,
		    29,30,31,32,33,
		    0,1,12,13,14,15,16,17,18,19,20,21,22,23,34,35});
   */

   map<int,int> perm;
   for(int i=0; i<loc.size(); i++){
      perm.insert(make_pair(loc[i],i));
   }
   vector<int> order;
   for(auto pr : perm){
      order.push_back(pr.second);
   }
   for(int i : order) cout << i << " ";
   cout << endl;

   int k = order.size();
   vector<int> image2(2*k);
   for(int i=0; i<k; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
   }

   int dim = space.size();
   cout << "dim of space = " << dim << endl;
   onspace sci_space;
   for(int i=0; i<dim; i++){
      auto state = space[i];
      auto state2 = state.permute(image2);	   
      sci_space.push_back(state2);
      //cout << "i=" << i << " " 
      //     << state.to_string2() << "->"
      //     << state2.to_string2() << endl; 
   }

   // solve
   int nroot = schd.nroots;
   vector<vector<double>> vs1(nroot);
   
   if(!schd.ciload){
      vector<double> es(nroot,0.0);
      linalg::matrix vs(dim,nroot);
      fci::sparse_hamiltonian sparseH;
      fci::ci_solver(sparseH, es, vs, sci_space, int2e, int1e, ecore);
      // analysis 
      for(int i=0; i<nroot; i++){
         vs1[i].resize(dim);
         copy(vs.col(i),vs.col(i)+dim,vs1[i].begin());
         //sci::pt2_solver(schd, es[i], vs1[i], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, vs1);
   }else{
      fci::ci_load(sci_space, vs1);
   }
   // check
   for(int i=0; i<nroot; i++){      
      coeff_population(sci_space, vs1[i], 1.e-3);
   }

   const double thresh = 1.e-3;

   // fe2s2
   /*
   vector<vector<int>> partitions({{2,3,4,5,6},
   				   {13,14,15,16,17},
   				   {0,1},
				   {7,8,9,10,11,12},
				   {18,19}});
   */
   vector<vector<int>> partitions({{2,3,4,5,6},
		   		   {2,3,4,5},
				   {2,3,4},
				   {2,3},
				   {2}});
   /*
   // fe4s4 
   vector<vector<int>> partitions({{2,3,4,5,6},
		   		  {7,8,9,10,11},
				  {24,25,26,27,28},
				  {29,30,31,32,33},
				  {0,1,12,13,14,15,16,17,18,19,20,21,22,23,34,35}});
   */
   int nimp = partitions.size();
   for(int i=0; i<nimp; i++){
      auto ilst = partitions[i];

      cout << "\n>>>>> I=" << i << " IMP=";
      for(int i : ilst) cout << i << " ";
      cout << endl;

      set<int> iset(ilst.begin(),ilst.end());
      vector<int> bath;
      for(int i : orderK){
         auto search = iset.find(i);
         if(search == iset.end()){
   	    bath.push_back(i);
         }
      }
   
      // bipartitions
      vector<int> order_new;
      copy(ilst.begin(),ilst.end(),std::back_inserter(order_new));
      copy(bath.begin(),bath.end(),std::back_inserter(order_new));
      cout << "\norder_new: " << order_new.size() << endl;
      for(int i : order_new) cout << i << " ";
      cout << endl;
   
      int pos = ilst.size();
      onspace space2;
      vector<vector<double>> vs2;
      tns::transform_coeff(sci_space, vs1, order_new, space2, vs2);
      tns::product_space pspace2;
      pspace2.get_pspace(space2, 2*pos);
      auto pr2 = pspace2.projection(vs2, thresh);
      cout << "pos=" << pos
           << " bdim=" << pr2.first
           << " SvN=" << pr2.second << endl;
   } // i

   return 0;
}
