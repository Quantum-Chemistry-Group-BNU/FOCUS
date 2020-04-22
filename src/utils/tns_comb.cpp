#include "../settings/global.h"
#include "../core/matrix.h"
#include "../core/onspace.h"
#include "../core/linalg.h"
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
using namespace linalg;

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
   cout << "--- rcoord & type ---" << endl;
   for(int i=0; i<ntotal; i++){
      auto p = rcoord[i];
      cout << "i=" << i << " : (" << p.first << "," << p.second << ")" 
	   << "[" << topo[p.first][p.second] << "]" 
	   << " type=" << type[p] << endl;
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

// compute renormalized bases {|r>} 
void comb::get_rbases(const onspace& space,
		      const vector<vector<double>>& vs,
		      const double thresh_proj){
   auto t0 = global::get_time();
   bool debug = true;
   cout << "\ncomb::get_rbases thresh_proj=" 
	<< scientific << thresh_proj << endl;
   vector<pair<int,int>> shapes;
   vector<int> bas(nphysical);
   iota(bas.begin(), bas.end(), 0);
   // loop over nodes (except the last one)
   for(int idx=0; idx<ntotal-1; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      if(debug){
         cout << "\nidx=" << idx 
	      << " node=(" << i << "," << j << ")[" << topo[i][j] << "] ";
	 cout << "rsup=";
         for(int k : rsupport[make_pair(i,j)]) cout << k << " ";
         cout << endl;
      }
      // 1. generate 1D ordering
      auto rsupp = rsupport[make_pair(i,j)];
      // order required in set_difference
      stable_sort(rsupp.begin(), rsupp.end()); 
      vector<int> order;
      set_difference(bas.begin(), bas.end(), rsupp.begin(), rsupp.end(),
                     back_inserter(order));
      int pos = order.size();
      // original order required [IMPORTANT]
      rsupp = rsupport[make_pair(i,j)]; 
      copy(rsupp.begin(), rsupp.end(), back_inserter(order));
      if(debug){
         cout << "pos=" << pos << endl;
	 cout << "order=";
         for(int k : order) cout << k << " ";
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
      auto rbasis = pspace2.right_projection(vs2,thresh_proj);
      rbases[p] = rbasis;
      if(debug){
	 int nbas = 0, ndim = 0;
         for(int k=0; k<rbasis.size(); k++){
	    rbasis[k].print("rsec_"+to_string(k));
	    nbas += rbasis[k].coeff.rows();
	    ndim += rbasis[k].coeff.cols();
	 }
	 cout << "rbasis: nbas,ndim=" << nbas << "," << ndim << endl;
	 shapes.push_back(make_pair(nbas,ndim));
      }
   } // idx
   if(debug){
      cout << "\nfinal results with thresh_proj = " << thresh_proj << endl;
      int Dmax = 0;
      for(int idx=0; idx<ntotal-1; idx++){
         auto p = rcoord[idx];
         int i = p.first, j = p.second;
	 cout << "idx=" << idx 
	      << " node=(" << i << "," << j << ")[" << topo[i][j] << "]"
	      << " nbas=" << shapes[idx].first << " ndim=" << shapes[idx].second
	      << endl;
	 Dmax = max(Dmax,shapes[idx].second);
      } // idx
      cout << "maximum bond dimension = " << Dmax << endl;
   }
   auto t1 = global::get_time();
   cout << "\ntiming for comb::get_rbases : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

// build site tensor from {|r>} basis
void comb::get_rcanon(const double thresh_ortho){
   auto t0 = global::get_time();
   bool debug = true;
   cout << "\ncomb::get_rcanon" << endl;
   // loop over sites
   for(int idx=0; idx<ntotal-1; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      if(debug){
         cout << "\nidx=" << idx 
	      << " node=(" << i << "," << j << ")[" << topo[i][j] << "] "
	      << endl;
      }
      auto rbasis = rbases[p]; 
      renorm_tensor rt;
      if(type[p] == 0){
	 
	 //       n             |vac>
	 //      \|/             \|/
	 //    -<-*-<-|vac>   n-<-*
	 //    			 \|/
	 if(debug) cout << "type 0: end or leaves" << endl; 
	 rt.qspace0 = qphys; 
	 rt.qspace1[qphys[0]] = 1; // in  
	 for(int k0=0; k0<qphys.size(); k0++){
	    rt.qspace[qphys[k0]] = 1; // out
	 }
	 rt.qblocks[make_tuple(0,qphys[0],qphys[0])] = identity_matrix(1);
	 rt.qblocks[make_tuple(1,qphys[0],qphys[1])] = identity_matrix(1);
	 rt.qblocks[make_tuple(2,qphys[0],qphys[2])] = identity_matrix(1);
	 rt.qblocks[make_tuple(3,qphys[0],qphys[3])] = identity_matrix(1);

      }else if(type[p] == 1 || type[p] == 2){
	 
	 rt.qspace0 = qphys;
	 pair<int,int> pre;
	 if(type[p] == 1){
	    //       n      
	    //      \|/      
	    //    -<-*-<-|r> 
	    if(debug) cout << "type 1: physical site on backbone" << endl;
	    pre = make_pair(i+1,0);
	 }else if(type[p] == 2){
	    //      |u>
	    //      \|/
	    //   n-<-*
	    //      \|/
	    if(debug) cout << "type 2: physical site on branch" << endl;
	    pre = make_pair(i,j+1);
	 }
         renorm_basis rbasis1; 
	 if(type[pre] == 0){
	    rbasis1 = get_rbasis_phys(); // exact repr. for this bond 
	 }else{
	    rbasis1 = rbases[pre];
	 }
	 // loop over symmetry blocks
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qspace[sym] = rbasis[k].coeff.cols();
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qspace1[sym1] = rbasis1[k1].coeff.cols();
	       // loop over physical indices
	       for(int k0=0; k0<qphys.size(); k0++){
	          if((sym.first != sym1.first+qphys[k0].first) ||
	             (sym.second != sym1.second+qphys[k0].second)){
		     rt.qblocks[make_tuple(k0,sym1,sym)] = matrix();
		  }else{
		     auto Bi = get_Bmatrix(space_phys[k0],rbasis1[k1].space,rbasis[k].space);
		     auto BL = dgemm("N","N",Bi,rbasis[k].coeff);
		     auto RBL = dgemm("T","N",rbasis1[k1].coeff,BL);
		     rt.qblocks[make_tuple(k0,sym1,sym)] = RBL;
		  }
	       } // k0
	    } // k1
	 } // k

      }else if(type[p] == 3){
	 
	 //      |u>      
	 //      \|/      
	 //    -<-*-<-|r> 
	 if(debug) cout << "type 3: internal site on backbone" << endl;
	 auto rbasis0 = rbases[make_pair(i,j+1)];
	 auto rbasis1 = rbases[make_pair(i+1,j)];
	 // qspace0
	 for(int k0=0; k0<rbasis0.size(); k0++){
	    auto sym0 = rbasis0[k0].sym;
	    int ndim = rbasis0[k0].coeff.cols();
	    for(int idim=0; idim<ndim; idim++){
	       rt.qspace0.push_back(sym0);
	    } // ibas
	 }
	 // loop over symmetry blocks
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qspace[sym] = rbasis[k].coeff.cols();
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qspace1[sym1] = rbasis1[k1].coeff.cols();
	       // loop over upper indices
	       int ioff = 0;
	       for(int k0=0; k0<rbasis0.size(); k0++){
	          auto sym0 = rbasis0[k0].sym;
		  int nbas = rbasis0[k0].coeff.rows();
		  int ndim = rbasis0[k0].coeff.cols();
		  if((sym.first != sym1.first+sym0.first) ||
	             (sym.second != sym1.second+sym0.second)){
		     for(int idim=0; idim<ndim; idim++){
		        rt.qblocks[make_tuple(ioff+idim,sym1,sym)] = matrix();
		     } // idim
		  }else{
		     vector<matrix> Wrl(ndim);
		     for(int ibas=0; ibas<nbas; ibas++){
			auto state0 = rbasis0[k0].space[ibas];
		        auto Bi = get_Bmatrix(state0,rbasis1[k1].space,rbasis[k].space);
		        auto BL = dgemm("N","N",Bi,rbasis[k].coeff);
		        auto RBL = dgemm("T","N",rbasis1[k1].coeff,BL);
			if(ibas == 0){
		           for(int idim=0; idim<ndim; idim++){
			      Wrl[idim] = rbasis0[k0].coeff(ibas,idim)*RBL;
			   } // idim
			}else{
		           for(int idim=0; idim<ndim; idim++){
			      Wrl[idim] += rbasis0[k0].coeff(ibas,idim)*RBL;
			   } // idim
			}
		     } // ibas
		     for(int idim=0; idim<ndim; idim++){
			rt.qblocks[make_tuple(ioff+idim,sym1,sym)] = Wrl[idim];
		     } // idim
		  }
	          ioff += ndim;
	       } // k0
	    } // k1
	 } // k

      } // type[p]
      rt.print("renorm_tensor_"+to_string(idx));
      rsites[p] = rt;

      cout << "check orthogonality for right canonical site:" << endl;
      int Dtot = 0;
      for(const auto& p : rt.qspace){
         auto& sym = p.first;
         int ndim = p.second;
	 Dtot += ndim;
         matrix Sr(ndim,ndim);
	 // S[r,r'] = \sum_{l,c} Ac[l,r]*Ac[l,r']
         for(const auto& p1 : rt.qspace1){
            auto& sym1 = p1.first;
            for(int i=0; i<rt.qspace0.size(); i++){
               auto& blk = rt.qblocks[make_tuple(i,sym1,sym)];
               if(blk.size() == 0) continue; 
               Sr += dgemm("N","N",blk.transpose(),blk);
            }
         }
         auto diff = normF(Sr - identity_matrix(ndim));
         cout << " qsym=(" << sym.first << "," << sym.second << ")"
              << " ndim=" << ndim << " |Sr-Id|_F=" << diff << endl;
         if(diff > thresh_ortho){
            Sr.print("Sr_sym("+to_string(sym.first)+","+to_string(sym.second)+")");
            cout << "error: deviate from identity matrix! diff=" << diff << endl;
            exit(1);
         }
      } // sym blocks
      cout << "total bond dimension=" << Dtot << endl;

   } // idx
   auto t1 = global::get_time();
   cout << "\ntiming for comb::get_rcanon : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
