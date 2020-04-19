#include "tns_ordering.h"
#include "tns_pspace.h"
#include "tns_comb.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace tns;
using namespace fock;

void comb::read_topology(string fname){
   cout << "\ncomb::read_topology fname=" << fname << endl;
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
   vector<string> v;
   string line;
   while(!istrm.eof()){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      boost::trim_left(line); // in case there is a space in FCIDUMP
      boost::split(v,line,boost::is_any_of(","),boost::token_compress_on);
      vector<int> branch;
      for(auto s : v){
	 branch.push_back(stoi(s));
      }
      topo.push_back(branch);
   }
   istrm.close();
}

void comb::init(){
   cout << "\ncomb::init" << endl;
   // initialize comb structure
   nbackbone = topo.size();
   nphysical = 0;
   ninternal = 0;
   if(topo[0].size() != 1 || topo[nbackbone-1].size() != 1){
      cout << "error: we assume the start and end nodes are leaves!" << endl;
      exit(1);
   }
   for(int i=0; i<nbackbone; i++){
      int size = topo[i].size();
      nphysical += size;
      if(size > 1) ninternal += 1;
   }
   ntotal = nphysical + ninternal;
   // add internal nodes
   int idx = nphysical;
   for(int i=0; i<nbackbone; i++){
      if(topo[i].size() > 1){
	 topo[i].insert(topo[i].begin(),idx);
         idx++;
      }
   }
   // coordinate of nodes in right canonical form
   for(int i=nbackbone-1; i>=0; i--){
      for(int j=topo[i].size()-1; j>=0; j--){
         rcoord.push_back(make_pair(i,j));
      }
   }
   // compute support of each node in right canonical form
   for(int i=nbackbone-1; i>=0; i--){
      int size = topo[i].size();
      if(size == 1){
	 // upper branch is just physical indices     
	 rsupport[make_pair(i,0)].push_back(topo[i][0]);
         if(i!=nbackbone-1){ 
	    // build recursively by copying right branch
	    copy(rsupport[make_pair(i+1,0)].begin(),
	         rsupport[make_pair(i+1,0)].end(),
		 back_inserter(rsupport[make_pair(i,0)]));
	 }
      }else{
	 // visit upper branch from the leaf
         for(int j=size-1; j>0; j--){
	    rsupport[make_pair(i,j)].push_back(topo[i][j]);
	    if(j!=size-1){
  	       copy(rsupport[make_pair(i,j+1)].begin(),
	            rsupport[make_pair(i,j+1)].end(),
		    back_inserter(rsupport[make_pair(i,j)]));
	    }
	 }
	 // branching node: upper
	 copy(rsupport[make_pair(i,1)].begin(),
	      rsupport[make_pair(i,1)].end(),
	      back_inserter(rsupport[make_pair(i,0)]));
	 // right - assuming the end node is leaf (which is true)
	 copy(rsupport[make_pair(i+1,0)].begin(),
	      rsupport[make_pair(i+1,0)].end(),
	      back_inserter(rsupport[make_pair(i,0)]));
      }
   }
   // sweep sequence: forward
   for(int i=1; i<nbackbone; i++){
      auto coord0 = make_pair(i-1,0);
      auto coord1 = make_pair(i,0);      
      sweep_seq.push_back(make_pair(coord0,coord1));
      // branch forward
      for(int j=1; j<topo[i].size(); j++){
         auto coord0 = make_pair(i,j-1);
         auto coord1 = make_pair(i,j);      
         sweep_seq.push_back(make_pair(coord0,coord1));
      }
      // branch backward
      for(int j=topo[i].size()-1; j>0; j--){
         auto coord0 = make_pair(i,j);
         auto coord1 = make_pair(i,j-1);      
         sweep_seq.push_back(make_pair(coord0,coord1));
      }
   }
   // backward
   for(int i=nbackbone-1; i>0; i--){
      auto coord0 = make_pair(i,0);
      auto coord1 = make_pair(i-1,0);      
      sweep_seq.push_back(make_pair(coord0,coord1));
   }
   assert(sweep_seq.size() == 2*(ntotal-1));
}

void comb::print(){
   cout << "\ncomb::print" << endl;
   cout << "nbackbone=" << nbackbone << " " 
	<< "nphysical=" << nphysical << " "
	<< "ninternal=" << ninternal << " " 
	<< "ntotal=" << ntotal << endl;
   cout << "--- topo ---" << endl;
   int idx = 0;
   for(auto& branch : topo){
      cout << "idx=" << idx << " : "; 
      for(int i : branch){
         cout << i << " ";  
      }
      cout << endl;
      idx++;
   }
   cout << "--- rcoord ---" << endl;
   for(int i=0; i<ntotal; i++){
      auto p = rcoord[i];
      cout << "i=" << i << " : (" << p.first << "," << p.second << ")" 
	   << "[" << topo[p.first][p.second] << "]" << endl;
   }
   cout << "--- rsupport ---" << endl;
   for(const auto& p : rsupport){
      auto coord = p.first;
      auto rsupp = p.second;
      cout << "coord=(" << coord.first << "," << coord.second << ") : ";
      for(const auto& k : rsupp){
         cout << k << " ";
      }
      cout << endl;
   }
   cout << "--- sweep_seq --" << endl;
   for(int i=0; i<sweep_seq.size(); i++){
      cout << "i=" << i << " : ";
      auto coord0 = sweep_seq[i].first;
      auto coord1 = sweep_seq[i].second;
      int x0 = coord0.first, y0 = coord0.second;
      int x1 = coord1.first, y1 = coord1.second; 
      cout << "(" << x0 << "," << y0 << ")[" << topo[x0][y0] << "]" 
	   << " - "
           << "(" << x1 << "," << y1 << ")[" << topo[x1][y1] << "]" 
	   << endl;
   }
}

void comb::get_rcanon(const onspace& space,
		      const vector<vector<double>>& vs,
		      const double thresh){
   cout << "\ncomb::get_rcanon" << endl;
   bool debug = true;
   vector<int> bas(nphysical);
   iota(bas.begin(), bas.end(), 0);
   // loop over nodes (except the last one)
   for(int idx=10; idx<ntotal-1; idx++){

      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      if(debug){
         cout << "\nidx=" << idx 
	      << " node=(" << i << "," << j << ")" 
              << "[" << topo[i][j] << "] ";
	 cout << "rsup=";
         for(int k : rsupport[make_pair(i,j)]) cout << k << " ";
         cout << endl;
      }
      // 1. generate 1D ordering
      auto rsupp = rsupport[make_pair(i,j)];
      stable_sort(rsupp.begin(), rsupp.end());
      vector<int> order;
      set_difference(bas.begin(), bas.end(), rsupp.begin(), rsupp.end(),
                     back_inserter(order));
      int pos = order.size();
      copy(rsupp.begin(), rsupp.end(), back_inserter(order));
      if(debug){
         cout << "pos=" << pos << endl;
	 cout << "order=";
         for(int k : order){
            cout << k << " ";
         }
         cout << endl;
      } 
      // 2. transform SCI coefficient
      onspace space2;
      vector<vector<double>> vs2;
      transform_coeff(space, vs, order, space2, vs2); 
      // 3. bipartition of space
      tns::product_space pspace2;
      pspace2.get_pspace(space2, 2*pos);
      // 4. projection of SCI wavefunction and save renormalized states
      //    (Schmidt decomposition for single state)
      auto rbasis = pspace2.right_projection(vs2,1.e-4);
      exit(1);
   } // idx
}
