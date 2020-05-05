#include "tns_comb.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace tns;

vector<int> comb::support_rest(const vector<int>& rsupp){
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

void comb::topo_read(string fname){
   cout << "\ncomb::topo_read fname=" << fname << endl;
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

void comb::topo_init(){
   cout << "\ncomb::topo_init" << endl;
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
   // type of sites 
   for(int i=nbackbone-1; i>=0; i--){
      int size = topo[i].size();
      if(i==nbackbone-1 || i==0){ 
	 type[make_pair(i,0)] = 0; // type 0: end
      }else{
	 if(size == 1){
            type[make_pair(i,0)] = 1; // type 1: physical site on backbone
	 }else if(size > 1){
            type[make_pair(i,size-1)] = 0; // type 0: leaves on branch
	    for(int j=size-2; j>=1; j--){
	       type[make_pair(i,j)] = 2; // type 2: physical site on branch
	    } // j
            type[make_pair(i,0)] = 3; // type 3: internal site on backbone
	 }
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
   // lsupport
   for(int idx=0; idx<rcoord.size(); idx++){
      auto coord = rcoord[idx];
      lsupport[coord] = support_rest(rsupport[coord]);
      if(iswitch==-1 && coord.second == 0 && 
         lsupport[coord].size()<rsupport[coord].size()){
         iswitch = coord.first;
      }
   }
   // image2
   auto order = rsupport[make_pair(0,0)]; 
   image2.resize(2*nphysical);
   for(int i=0; i<nphysical; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
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

void comb::topo_print() const{
   cout << "\ncomb::topo_print" << endl;
   cout << "nbackbone=" << nbackbone << " " 
	<< "nphysical=" << nphysical << " "
	<< "ninternal=" << ninternal << " " 
	<< "ntotal=" << ntotal << endl;
   cout << "--- topo ---" << endl;
   cout << "iswitch=" << iswitch << endl; 
   int idx = 0;
   for(auto& branch : topo){
      cout << "idx=" << idx << " : "; 
      for(int i : branch){
         cout << i << " ";  
      }
      cout << endl;
      idx++;
   }
   cout << "--- rcoord & type & rsupport/lsupport ---" << endl;
   for(int i=0; i<ntotal; i++){
      auto p = rcoord[i];
      auto rsupp = rsupport.at(p);
      auto lsupp = lsupport.at(p);
      cout << "i=" << i << " : (" << p.first << "," << p.second << ")" 
	   << "[" << topo[p.first][p.second] << "]" 
	   << " type=" << type.at(p) << endl;
      cout << "rsupport: ";
      for(int k : rsupp) cout << k << " ";
      cout << endl;
      cout << "lsupport: ";
      for(int k : lsupp) cout << k << " ";
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
