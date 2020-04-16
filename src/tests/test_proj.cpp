#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../utils/fci.h"
#include "../utils/fci_rdm.h"
#include "../utils/sci.h"
#include "../utils/tns.h"
#include "../settings/global.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_proj(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_proj" << endl;
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
  
   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<double>> vs(nroot);
   
   if(!schd.ciload){
      fci::sparse_hamiltonian sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      sparseH.analysis();
      // pt2 for single root
      if(schd.ifpt2){
         sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, vs);
   }else{
      fci::ci_load(sci_space, vs);
   }
   // check 
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
   }

   vector<int> bdims;
   double SvN;
   
   vector<int> orderJ;
   tns::ordering_fiedler(int2e.J, orderJ);
/*
   tns::bipartite_entanglement(sci_space, vs, orderJ, bdims, SvN);
   cout << "bdims:";
   for(int i : bdims) cout << i << " ";
   cout << endl;
*/
   vector<int> orderK;
   tns::ordering_fiedler(int2e.K, orderK);
/*
   tns::bipartite_entanglement(sci_space, vs, orderK, bdims, SvN);
   cout << "bdims:";
   for(int i : bdims) cout << i << " ";
   cout << endl;
*/
   const double thresh = 1.e-3;

/*
   int pos = 7;
   tns::product_space pspace;
   pspace.get_pspace(sci_space, 2*pos);
   auto pr = pspace.projection(vs, thresh);
   cout << "pos=" << pos
        << " bdim=" << pr.first
        << " SvN=" << pr.second << endl;

   // partitions for fe2s2
   vector<int> fe1({2,3,4,5,6});
   vector<int> fe2({13,14,15,16,17});
   vector<int> felst;
   copy(fe1.begin(),fe1.end(),std::back_inserter(felst));
   copy(fe2.begin(),fe2.end(),std::back_inserter(felst));
   for(int i : felst) cout << i << " ";
   cout << endl;

   set<int> feset(felst.begin(),felst.end());
   vector<int> bath;
   for(int i : orderK){
      auto search = feset.find(i);
      if(search == feset.end()){
	 bath.push_back(i);
      }
   }
   for(int i : bath) cout << i << " ";
   cout << endl;

   // bipartitions
   vector<int> order_new;
   copy(bath.begin(),bath.end(),std::back_inserter(order_new));
   copy(fe1.begin(),fe1.end(),std::back_inserter(order_new));
   copy(fe2.begin(),fe2.end(),std::back_inserter(order_new));
   cout << "\norder_new:" << endl;
   for(int i : order_new) cout << i << " ";
   cout << endl;

   pos = bath.size();
   onspace space2;
   vector<vector<double>> vs2;
   tns::transform_coeff(sci_space, vs, order_new, space2, vs2);
   tns::product_space pspace2;
   pspace2.get_pspace(space2, 2*pos);

   auto pr2 = pspace2.projection(vs2, thresh);
   cout << "pos=" << pos
        << " bdim=" << pr2.first
        << " SvN=" << pr2.second << endl;
*/

   // partitions for fe4s4
   //vector<int> bath({0,1,12,13,14,15,16,17,18,19,20,21,22,23,34,35});
   
   // partitions for fe8s7
   /*
   vector<int> bath({0,1,2,
	             8,9,10,11,12,13,14,
		     20,21,22,
		     33,34,35,36,37,38,39,
		     50,51,52,53,54,55,56,57,58,
  		     69,70,71,72});
   */

   // partitions for fe2s2
   //vector<int> bath({0,1,7,8,9,10,11,12,18,19});

/*
   set<int> bset(bath.begin(),bath.end());
   vector<int> felst;
   for(int i : orderK){
      auto search = bset.find(i);
      if(search == bset.end()){
	 felst.push_back(i);
      }
   }

   // bipartitions
   vector<int> order_new;
   copy(bath.begin(),bath.end(),std::back_inserter(order_new));
   copy(felst.begin(),felst.end(),std::back_inserter(order_new));
   cout << "\norder_new: " << order_new.size() << endl;
   for(int i : order_new) cout << i << " ";
   cout << endl;

   //int pos = bath.size();
  
   order_new = orderK;

   //order_new = orderK;
   for(int pos=1; pos<order_new.size(); pos++){
   onspace space2;
   vector<vector<double>> vs2;
   tns::transform_coeff(sci_space, vs, order_new, space2, vs2);
   tns::product_space pspace2;
   pspace2.get_pspace(space2, 2*pos);
   auto pr2 = pspace2.projection(vs2, thresh);
   cout << "pos=" << pos
        << " bdim=" << pr2.first
        << " SvN=" << pr2.second << endl;
   }
*/

   // fe2s2
   vector<vector<int>> partitions({{2,3,4,5,6},
   				   {13,14,15,16,17},
   				   {0,1,7,8,9,10,11,12,18,19}});
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
      tns::transform_coeff(sci_space, vs, order_new, space2, vs2);
      tns::product_space pspace2;
      pspace2.get_pspace(space2, 2*pos);
      auto pr2 = pspace2.projection(vs2, thresh);
      cout << "pos=" << pos
           << " bdim=" << pr2.first
           << " SvN=" << pr2.second << endl;
   } // i

   return 0;
}
