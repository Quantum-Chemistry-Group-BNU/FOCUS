#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <numeric> // iota
#include "../core/tools.h"
#include "ctns_topo.h"
#include "oper_io.h"

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

// directed_bond
ostream& ctns::operator <<(ostream& os, const directed_bond& dbond){
   os << dbond.p0 << "-" << dbond.p1
      << " forward=" << dbond.forward
      << " cturn=" << dbond.is_cturn();
   return os;   
}

// topology
void topology::read(const string& fname){
   cout << "\nctns::topology::read fname=" << fname << endl;
   
   ifstream istrm(fname);
   if(!istrm){
      cout << "failed to open " << fname << '\n';
      exit(1);
   }
   // load topo from file
   vector<vector<int>> orbsgrid;
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
      orbsgrid.push_back(branch);
   }
   istrm.close();
   // consistency check
   if(orbsgrid[0].size() != 1 || orbsgrid[orbsgrid.size()-1].size() != 1){
      tools::exit("error: we assume the start and end nodes are leaves!");
   }

   // initialize topo structure: type & neighbor of each site
   nbackbone = orbsgrid.size();
   nphysical = 0;
   nodes.resize(nbackbone);
   for(int i=nbackbone-1; i>=0; i--){
      int size = orbsgrid[i].size();
      nphysical += size;
      if(i==nbackbone-1){
	 nodes[i].resize(1);
	 // type 0: end
	 auto& node = nodes[i][0];
	 node.pindex = orbsgrid[i][0]; 
	 node.type   = 0;
	 node.center = coord_phys;
	 node.left   = make_pair(i-1,0);
	 node.right  = coord_vac;
      }else if(i==0){
	 nodes[i].resize(1);
	 // type 0: start
	 auto& node = nodes[i][0];
	 node.pindex = orbsgrid[i][0]; 
	 node.type   = 0;
	 node.center = coord_phys;
	 node.left   = coord_vac;
	 node.right  = make_pair(i+1,0);
      }else{
	 if(size == 1){
	    nodes[i].resize(1);
	    // type 1: physical site on backbone
	    auto& node = nodes[i][0];
	    node.pindex = orbsgrid[i][0];
	    node.type   = 1;
	    node.center = coord_phys;
	    node.left   = make_pair(i-1,0);
	    node.right  = make_pair(i+1,0);
	 }else if(size > 1){
	    nodes[i].resize(size+1);
	    // type 0: leaves on branch
            auto& node = nodes[i][size];
	    node.pindex = orbsgrid[i][size-1];
	    node.type   = 0;
	    node.center = coord_phys;
	    node.left   = make_pair(i,size-1);
	    node.right  = coord_vac;
	    // type 2: physical site on branch
	    for(int j=size-1; j>=1; j--){
	       auto& nodej = nodes[i][j];
	       nodej.pindex = orbsgrid[i][j-1];
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
   ntotal = 0;
   for(int i=nbackbone-1; i>=0; i--){
      for(int j=nodes[i].size()-1; j>=0; j--){
         rcoord.push_back(make_pair(i,j));
	 rindex[make_pair(i,j)] = ntotal;
	 ntotal++;
      }
   }

   // compute support of each node in right canonical form
   for(int i=nbackbone-1; i>=0; i--){
      int size = orbsgrid[i].size(); // same as input topo
      if(size == 1){
	 // upper branch is just physical indices     
	 nodes[i][0].rsupport.push_back(orbsgrid[i][0]);
	 nodes[i][0].corbs.push_back(orbsgrid[i][0]);
         if(i != nbackbone-1){ 
	    // build recursively by copying right branch
	    copy(nodes[i+1][0].rsupport.begin(),
	         nodes[i+1][0].rsupport.end(),
		 back_inserter(nodes[i][0].rsupport));
	    nodes[i][0].rorbs = nodes[i+1][0].rsupport;
	 }
      }else{
	 // visit upper branch from the leaf
         for(int j=size; j>0; j--){
	    nodes[i][j].rsupport.push_back(orbsgrid[i][j-1]);
	    nodes[i][j].corbs.push_back(orbsgrid[i][j-1]);
	    if(j != size){
  	       copy(nodes[i][j+1].rsupport.begin(),
	            nodes[i][j+1].rsupport.end(),
		    back_inserter(nodes[i][j].rsupport));
	       nodes[i][j].rorbs = nodes[i][j+1].rsupport;
	    }
	 }
	 // branching node: upper
	 copy(nodes[i][1].rsupport.begin(),
	      nodes[i][1].rsupport.end(),
	      back_inserter(nodes[i][0].rsupport));
	 nodes[i][0].corbs = nodes[i][1].rsupport;
	 // right - assuming the end node is leaf (which is true)
	 copy(nodes[i+1][0].rsupport.begin(),
	      nodes[i+1][0].rsupport.end(),
	      back_inserter(nodes[i][0].rsupport));
	 nodes[i][0].rorbs = nodes[i+1][0].rsupport;
      }
   }
   // lsupport
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      nodes[i][j].lsupport = get_supp_rest(nodes[i][j].rsupport);
      nodes[i][j].lorbs = nodes[i][j].lsupport;
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
   cout << "\nctns::topology::print"
	<< " ntotal=" << ntotal
	<< " nphysical=" << nphysical 
        << " nbackbone=" << nbackbone
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
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      auto& node = nodes[p.first][p.second];
      cout << " idx=" << idx << " coord=" << p << " " << node << endl;
      assert(idx == rindex.at(p));
   }
   cout << "rsupport/lsupport:" << endl;
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      cout << " idx=" << idx << " coord=" << p;
      cout << " rsupport: ";
      for(int k : nodes[i][j].rsupport) cout << k << " ";
      cout << "; lsupport: ";
      for(int k : nodes[i][j].lsupport) cout << k << " ";
      cout << endl;
   }
   cout << "corbs/rorbs/lorbs:" << endl;
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      cout << " idx=" << idx << " coord=" << p;
      cout << " corbs: ";
      for(int k : nodes[i][j].corbs) cout << k << " ";
      cout << "; rorbs: ";
      for(int k : nodes[i][j].rorbs) cout << k << " ";
      cout << "; lorbs: ";
      for(int k : nodes[i][j].lorbs) cout << k << " ";
      cout << endl;
   }
   cout << "image2:" << endl;
   for(int i=0; i<2*nphysical; i++) cout << " " << image2[i];
   cout << endl;
   // just check sweep sequence 
   get_sweeps();
}

vector<directed_bond> topology::get_sweeps(const bool debug) const{
   if(debug) cout << "\nctns::topology::get_sweeps" << endl;
   vector<directed_bond> sweeps;
   // sweep sequence: 
   for(int i=1; i<nbackbone-1; i++){
      // branch forward
      for(int j=1; j<nodes[i].size()-1; j++){
         auto p0 = make_pair(i,j-1);
         auto p1 = make_pair(i,j);
         sweeps.push_back( directed_bond(p0,p1,1) );
      }
      // branch backward
      for(int j=nodes[i].size()-2; j>0; j--){
         auto p0 = make_pair(i,j-1);
         auto p1 = make_pair(i,j);
         sweeps.push_back( directed_bond(p0,p1,0) );
      }
      // backbone forward
      if(i != nbackbone-2){
         auto p0 = make_pair(i,0);
         auto p1 = make_pair(i+1,0);      
         sweeps.push_back( directed_bond(p0,p1,1) );
      }
   }
   // backward on backbone
   for(int i=nbackbone-2; i>=2; i--){
      auto p0 = make_pair(i-1,0);      
      auto p1 = make_pair(i,0); 
      sweeps.push_back( directed_bond(p0,p1,0) );
   }
   // check
   if(debug){
      // in this scheme, each internal bond is visited twice
      int ninternal = 0;
      for(const auto& p : rcoord){
	 auto& node = nodes[p.first][p.second];
	 if(node.type != 0) ninternal += 1; 
      }
      assert(sweeps.size() == 2*(ninternal-1));
      for(int idx=0; idx<sweeps.size(); idx++){
         const auto& p0 = sweeps[idx].p0;
         const auto& p1 = sweeps[idx].p1;
         const auto& forward = sweeps[idx].forward;
	 cout << " ibond=" << idx 
	      << " bond=" << p0 << "-" << p1 
	      << " forward=" << forward
	      << " cturn=" << sweeps[idx].is_cturn()
	      << endl;
      }
   }
   return sweeps;
}

vector<int> topology::get_supp_rest(const vector<int>& rsupp) const{
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

// sweep related
bool topology::check_partition(const int dots,
			       const directed_bond& dbond,
                               const bool debug,
			       const int verbose) const{
   if(debug) cout << "ctns::topology::check_partition: ";
   bool ifNC;
   auto p = dbond.get_current();
   if(dots == 1){
      // onedot
      auto suppl = get_suppl(p);
      auto suppr = get_suppr(p);
      auto suppc = get_suppc(p);
      int sl = suppl.size();
      int sr = suppr.size();
      int sc = suppc.size();
      assert(sc+sl+sr == nphysical);
      ifNC = (sl <= sr);
      if(debug){
         cout << "(sl,sr,sc)="
              << sl << "," << sr << "," << sc
	      << " ifNC=" << ifNC
              << endl;
	 if(verbose > 0){
            tools::print_vector(suppl, "suppl");
            tools::print_vector(suppr, "suppr");
            tools::print_vector(suppc, "suppc");
	 }
      }
   }else if(dots == 2){
      // twodot
      const auto& p0 = dbond.p0;
      const auto& p1 = dbond.p1;
      vector<int> suppl, suppr, suppc1, suppc2;
      if(!dbond.is_cturn()){
         //        c1   c2
         //        |    |
         //    l---p0---p1---r
         //
         suppl  = get_suppl(p0);
         suppr  = get_suppr(p1);
         suppc1 = get_suppc(p0);
         suppc2 = get_suppc(p1);
      }else{
         //       c2
         //       |
         //  c1---p1
         //       |
         //   l---p0---r
         //
         suppl  = get_suppl(p0);
         suppr  = get_suppr(p0);
         suppc1 = get_suppc(p1);
         suppc2 = get_suppr(p1);
      }
      int sl = suppl.size();
      int sr = suppr.size();
      int sc1 = suppc1.size();
      int sc2 = suppc2.size();
      assert(sc1+sc2+sl+sr == nphysical);
      ifNC = (sl+sc1 <= sc2+sr);
      if(debug){
         cout << "(sl,sr,sc1,sc2)=" 
              << sl << "," << sr << "," << sc1 << "," << sc2
	      << " ifNC=" << ifNC
              << endl;
	 if(verbose > 0){
            tools::print_vector(suppl , "suppl");
            tools::print_vector(suppr , "suppr");
            tools::print_vector(suppc1, "suppc1");
            tools::print_vector(suppc2, "suppc2");
	 }
      }
   }
   return ifNC;
}

// get fqop around node p for kind = {"l","c","r"}
string topology::get_fqop(const comb_coord& p,
		          const string kind,
		          const string scratch) const{
   string fqop;
   const auto& node = get_node(p);
   if(kind == "c"){
      if(node.type != 3){
         fqop = oper_fname(scratch, p, "c"); // physical dofs
      }else{
         auto pc = node.center;
         fqop = oper_fname(scratch, pc, "r"); // branching site
      }
   }else if(kind == "r"){
      auto pr = node.right;
      fqop = oper_fname(scratch, pr, "r");
   }else if(kind == "l"){
      auto pl = node.left;
      fqop = oper_fname(scratch, pl, "l");
   }
   return fqop;
}

// fqops for sweep optimization
vector<string> topology::get_fqops(const int dots,
		                   const directed_bond& dbond,
			           const string scratch,
			           const bool debug) const{
   vector<string> fqops;
   if(dots == 1){
      auto p = dbond.get_current();
      fqops.resize(3); // l,r,c
      fqops[0] = get_fqop(p, "l", scratch);
      fqops[1] = get_fqop(p, "r", scratch);
      fqops[2] = get_fqop(p, "c", scratch);
   }else if(dots == 2){
      const auto& p0 = dbond.p0;
      const auto& p1 = dbond.p1;
      fqops.resize(4); // l,r,c1,c2
      if(!dbond.is_cturn()){
         //        c1   c2
         //        |    |
         //    l---p0---p1---r
         //
	 fqops[0] = get_fqop(p0, "l", scratch);
	 fqops[1] = get_fqop(p1, "r", scratch);
	 fqops[2] = get_fqop(p0, "c", scratch);
	 fqops[3] = get_fqop(p1, "c", scratch);
      }else{
         //       c2
         //       |
         //  c1---p1
         //       |
         //   l---p0---r
         //
         fqops[0] = get_fqop(p0, "l", scratch);
         fqops[1] = get_fqop(p0, "r", scratch);
         fqops[2] = get_fqop(p1, "c", scratch);
         fqops[3] = get_fqop(p1, "r", scratch);
      }
   } 
   if(debug){
      cout << "ctns::topology::get_fqops dots=" << dots << endl;
      for(int i=0; i<fqops.size(); i++){
         cout << " fqop[" << i << "]=" << fqops[i] << endl;
      }
   }
   return fqops;
}

pair<string,string> topology::get_fbond(const directed_bond& dbond,
			       	        const string scratch,
			  		const bool debug) const{
   const auto& p0 = dbond.p0;
   const auto& p1 = dbond.p1;
   string frop, fdel;
   if(dbond.forward){
      frop = oper_fname(scratch, p0, "l");
      fdel = oper_fname(scratch, p1, "r");
   }else{
      frop = oper_fname(scratch, p1, "r");
      fdel = oper_fname(scratch, p0, "l");
   }
   if(debug){
      cout << "ctns::topology::get_fbond"
	   << " frop=" << frop
	   << " fdel=" << fdel
	   << endl; 
   }
   return std::make_pair(frop,fdel);
}
