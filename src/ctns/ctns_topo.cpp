#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <numeric> // iota
#include "ctns_topo.h"

using namespace std;
using namespace ctns;

// comb_coord
ostream& ctns::operator <<(ostream& os, const comb_coord& coord){
   os << "(" << coord.first << "," << coord.second << ")";
   return os;
}

// node
ostream& ctns::operator <<(ostream& os, const node& nd){
   os << "node: pindex=" << nd.pindex 
      << " type="   << nd.type
      << " center=" << nd.center 
      << " left="   << nd.left
      << " right="  << nd.right;
   return os;   
}

// topology
topology::topology(const string& fname){
   cout << "\ntopology::topology fname=" << fname << endl;
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
   // load topo from file
   vector<vector<int>> tmp;
   vector<string> v;
   string line;
   while(!istrm.eof()){
      line.clear();	    
      getline(istrm,line);
      if(line.empty() || line[0]=='#') continue;
      cout << line << endl;
      boost::trim_left(line); // in case there is a space 
      boost::split(v,line,boost::is_any_of(","),boost::token_compress_on);
      vector<int> branch;
      for(auto s : v){
	 branch.push_back(stoi(s));
      }
      tmp.push_back(branch);
   }
   istrm.close();
   // consistency check
   if(tmp[0].size() != 1 || tmp[tmp.size()-1].size() != 1){
      cout << "error: we assume the start and end nodes are leaves!" << endl;
      exit(1);
   }

   // special coord 
   const comb_coord coord_vac = std::make_pair(-2,-2); 
   const comb_coord coord_phys = std::make_pair(-1,-1);  

   // initialize topo structure: type & neighbor of each site
   nbackbone = tmp.size();
   nphysical = 0;
   nodes.resize(nbackbone);
   for(int i=nbackbone-1; i>=0; i--){
      int size = tmp[i].size();
      nphysical += size;
      if(i==nbackbone-1){
	 nodes[i].resize(1);
	 // type 0: end
	 auto& node = nodes[i][0];
	 node.pindex = tmp[i][0]; 
	 node.type   = 0;
	 node.center = coord_phys;
	 node.left   = make_pair(i-1,0);
	 node.right  = coord_vac;
      }else if(i==0){
	 nodes[i].resize(1);
	 // type 0: start
	 auto& node = nodes[i][0];
	 node.pindex = tmp[i][0]; 
	 node.type   = 0;
	 node.center = coord_phys;
	 node.left   = coord_vac;
	 node.right  = make_pair(i+1,0);
      }else{
	 if(size == 1){
	    nodes[i].resize(1);
	    // type 1: physical site on backbone
	    auto& node = nodes[i][0];
	    node.pindex = tmp[i][0];
	    node.type   = 1;
	    node.center = coord_phys;
	    node.left   = make_pair(i-1,0);
	    node.right  = make_pair(i+1,0);
	 }else if(size > 1){
	    nodes[i].resize(size+1);
	    // type 0: leaves on branch
            auto& node = nodes[i][size];
	    node.pindex = tmp[i][size-1];
	    node.type   = 0;
	    node.center = coord_phys;
	    node.left   = make_pair(i,size-1);
	    node.right  = coord_vac;
	    // type 2: physical site on branch
	    for(int j=size-1; j>=1; j--){
	       auto& nodej = nodes[i][j];
	       nodej.pindex = tmp[i][j-1];
	       nodej.type   = 2;
	       nodej.center = coord_phys;
	       nodej.left   = make_pair(i,j-1);
	       nodej.right  = make_pair(i,j+1);
	    } // j
	    // type 3: internal site on backbone
            auto& nodei = nodes[i][0];
	    nodei.pindex = -1; // no physical index
	    nodei.type   = 3;
	    nodei.center = make_pair(i,1);
	    nodei.left   = make_pair(i-1,0);
	    nodei.right  = make_pair(i+1,0);
	 }
      }
   }

   // coordinate of nodes in right canonical form
   for(int i=nbackbone-1; i>=0; i--){
      for(int j=nodes[i].size()-1; j>=0; j--){
         rcoord.push_back(make_pair(i,j));
      }
   }

   // compute support of each node in right canonical form
   for(int i=nbackbone-1; i>=0; i--){
      int size = tmp[i].size(); // same as input topo
      if(size == 1){
	 // upper branch is just physical indices     
	 nodes[i][0].rsupport.push_back(tmp[i][0]);
         if(i != nbackbone-1){ 
	    // build recursively by copying right branch
	    copy(nodes[i+1][0].rsupport.begin(),
	         nodes[i+1][0].rsupport.end(),
		 back_inserter(nodes[i][0].rsupport));
	 }
      }else{
	 // visit upper branch from the leaf
         for(int j=size; j>0; j--){
	    nodes[i][j].rsupport.push_back(tmp[i][j-1]);
	    if(j != size){
  	       copy(nodes[i][j+1].rsupport.begin(),
	            nodes[i][j+1].rsupport.end(),
		    back_inserter(nodes[i][j].rsupport));
	    }
	 }
	 // branching node: upper
	 copy(nodes[i][1].rsupport.begin(),
	      nodes[i][1].rsupport.end(),
	      back_inserter(nodes[i][0].rsupport));
	 // right - assuming the end node is leaf (which is true)
	 copy(nodes[i+1][0].rsupport.begin(),
	      nodes[i+1][0].rsupport.end(),
	      back_inserter(nodes[i][0].rsupport));
      }
   }
   // lsupport
   iswitch=-1;
   for(int idx=0; idx<rcoord.size(); idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      nodes[i][j].lsupport = support_rest(nodes[i][j].rsupport);
      // locate switch point for bipartition of H 
      if(iswitch == -1 && j == 0 && 
         nodes[i][j].lsupport.size()<=nodes[i][j].rsupport.size()){
         iswitch = i;
      }
   }
   // image2 simply from rsupport[0,0] (1D order)
   auto order = nodes[0][0].rsupport; 
   image2.resize(2*nphysical);
   for(int i=0; i<nphysical; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
   }
}

void topology::print() const{
   cout << "\ntopology::print"
	<< " nphysical=" << nphysical 
        << " nbackbone=" << nbackbone
        << " iswitch=" << iswitch	
	<< endl;
   cout << "topo:" << endl;
   for(int i=0; i<nbackbone; i++){
      cout << " i=" << i << " : ";
      for(int j=0; j<nodes[i].size(); j++){
	 cout << nodes[i][j].pindex << " ";
      }
      cout << endl; 
   } 
   cout << "rcoord:" << endl;
   for(int idx=0; idx<rcoord.size(); idx++){
      auto p = rcoord[idx];
      auto& node = nodes[p.first][p.second];
      cout << " idx=" << idx << " coord=" << p << " " << node << endl;
   }
   cout << "rsupport/lsupport:" << endl;
   for(int idx=0; idx<rcoord.size(); idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      cout << " coord=" << p << " rsupport: ";
      for(int k : nodes[i][j].rsupport) cout << k << " ";
      cout << endl;
      cout << " coord=" << p << " lsupport: ";
      for(int k : nodes[i][j].lsupport) cout << k << " ";
      cout << endl;
   }
   cout << "image2:" << endl;
   for(int i=0; i<2*nphysical; i++) cout << " " << image2[i];
   cout << endl;
   // check sweep sequence 
   get_sweeps();
}

vector<int> topology::support_rest(const vector<int>& rsupp) const{
   vector<int> bas(nphysical);
   iota(bas.begin(), bas.end(), 0);
   auto supp = rsupp;
   // order required in set_difference
   stable_sort(supp.begin(), supp.end()); 
   vector<int> rest;
   set_difference(bas.begin(), bas.end(), supp.begin(), supp.end(),
                  back_inserter(rest));
   return rest;
}

vector<directed_bond> topology::get_sweeps(const bool debug) const{
   cout << "\ntopology::get_sweeps" << endl;
   vector<directed_bond> sweeps;
   // sweep sequence: 
   for(int i=1; i<nbackbone-1; i++){
      // branch forward
      for(int j=1; j<nodes[i].size()-1; j++){
         auto coord0 = make_pair(i,j-1);
         auto coord1 = make_pair(i,j);      
         sweeps.push_back(make_tuple(coord0,coord1,1));
      }
      // branch backward
      for(int j=nodes[i].size()-2; j>0; j--){
         auto coord0 = make_pair(i,j-1);      
         auto coord1 = make_pair(i,j);
         sweeps.push_back(make_tuple(coord0,coord1,0));
      }
      // backbone forward
      if(i != nbackbone-2){
         auto coord0 = make_pair(i,0);
         auto coord1 = make_pair(i+1,0);      
         sweeps.push_back(make_tuple(coord0,coord1,1));
      }
   }
   // backward on backbone
   for(int i=nbackbone-2; i>=2; i--){
      auto coord0 = make_pair(i-1,0);      
      auto coord1 = make_pair(i,0);
      sweeps.push_back(make_tuple(coord0,coord1,0));
   }
   if(debug){
      // in this scheme, each internal bond is visited twice
      int ninternal = 0;
      for(const auto& p : rcoord){
	 auto& node = nodes[p.first][p.second];
	 if(node.type != 0) ninternal += 1; 
      }
      assert(sweeps.size() == 2*(ninternal-1));
      for(int idx=0; idx<sweeps.size(); idx++){
         auto coord0  = get<0>(sweeps[idx]);
         auto coord1  = get<1>(sweeps[idx]);
         auto forward = get<2>(sweeps[idx]);
         cout << " idx=" << idx 
              << " dbond=" << coord0 << "-" << coord1 
	      << " forward=" << forward << endl; 
      }
   }
   return sweeps;
}
