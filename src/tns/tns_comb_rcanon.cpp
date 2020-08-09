#include "../core/tools.h"
#include "../core/linalg.h"
#include "tns_qsym.h"
#include "tns_comb.h"
#include "tns_ordering.h"
#include "tns_qtensor.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace tns;
using namespace fock;
using namespace linalg;

// compute renormalized bases {|r>} 
void comb::get_rbases(const onspace& space,
		      const vector<vector<double>>& vs,
		      const double thresh_proj){
   const bool debug = true;
   auto t0 = tools::get_time();
   cout << "\ncomb::get_rbases thresh_proj=" << scientific << thresh_proj << endl;
   vector<pair<int,int>> shapes; // for debug
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
      if(type[p] == 0){
         rbases[p] = get_rbasis_phys();
      }else{
         // 1. generate 1D ordering
         auto rsupp = rsupport[p]; // original order required [IMPORTANT]
         auto order = support_rest(rsupp);
         int pos = order.size(); // must be put here to account bipartition position
         copy(rsupp.begin(), rsupp.end(), back_inserter(order));
         if(debug){
            cout << "order=";
            for(int k : order) cout << k << " ";
            cout << endl;
            cout << "bipartition position=" << pos << endl;
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
      }
      // debug
      {
	 auto& rbasis = rbases[p];
	 int nbas = 0, ndim = 0;
         for(int k=0; k<rbasis.size(); k++){
	    if(debug) rbasis[k].print("rsec_"+to_string(k));
	    nbas += rbasis[k].coeff.rows();
	    ndim += rbasis[k].coeff.cols();
	 }
	 shapes.push_back(make_pair(nbas,ndim));
	 if(debug) cout << "rbasis: nbas,ndim=" << nbas << "," << ndim << endl;
      }
   } // idx
   // debug
   {
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
   auto t1 = tools::get_time();
   cout << "\ntiming for comb::get_rbases : " << setprecision(2) 
        << tools::get_duration(t1-t0) << " s" << endl;
}

// compute wave function at the start for right canonical form 
qtensor3 comb::get_rwavefuns(const onspace& space,
		             const vector<vector<double>>& vs,
		             const vector<int>& order,
		             const renorm_basis& rbasis){
   const bool debug = false;
   cout << "\ncomb::get_rwavefuns" << endl;
   // transform SCI coefficient
   onspace space2;
   vector<vector<double>> vs2;
   transform_coeff(space, vs, order, space2, vs2); 
   // bipartition of space
   tns::product_space pspace2;
   pspace2.get_pspace(space2, 2);
   // loop over symmetry of B;
   map<qsym,vector<int>> qsecB; // sym -> indices in spaceB
   map<qsym,map<int,int>> qmapA; // index in spaceA to idxA
   map<qsym,vector<tuple<int,int,int>>> qlst;
   for(int ib=0; ib<pspace2.dimB; ib++){
      auto& stateB = pspace2.spaceB[ib];
      qsym symB(stateB.nelec(), stateB.nelec_a());
      qsecB[symB].push_back(ib);
      for(const auto& pia : pspace2.colB[ib]){
	 int ia = pia.first;
	 int idet = pia.second;
	 // search unique
	 auto it = qmapA[symB].find(ia);
         if(it == qmapA[symB].end()){
            qmapA[symB].insert({ia,qmapA[symB].size()});
         };
	 int idxB = qsecB[symB].size()-1;
	 int idxA = qmapA[symB][ia];
	 qlst[symB].push_back(make_tuple(idxB,idxA,idet));
      }
   } // ib
   // construct rwavefuns 
   qtensor3 rwavefuns;
   rwavefuns.qmid = phys_qsym_space;
   // assuming the symmetry of wavefunctions are the same
   qsym sym_state(space[0].nelec(), space[0].nelec_a());
   int nroots = vs2.size();
   rwavefuns.qrow[sym_state] = nroots;
   // init empty blocks for all combinations 
   int idx = 0;
   for(auto it = qsecB.cbegin(); it != qsecB.cend(); ++it){
      auto& symB = it->first;
      rwavefuns.qcol[symB] = rbasis[idx].coeff.cols();
      for(int k0=0; k0<4; k0++){
	 auto key = make_tuple(phys_sym[k0],sym_state,symB);
         rwavefuns.qblocks[key] = empty_block; 
      }
      idx++;
   }
   // loop over symmetry sectors of |r>
   idx = 0;
   for(auto it = qsecB.cbegin(); it != qsecB.cend(); ++it){
      auto& symB = it->first;
      auto& idxB = it->second;
      int dimBs = idxB.size(); 
      int dimAs = qmapA[symB].size();
      if(debug){
         cout << "idx=" << idx << " symB(Ne,Na)=" << symB 
              << " dimBs=" << dimBs
              << " dimAs=" << qmapA[symB].size() 
              << endl;
      }
      // load renormalized basis
      auto& rsec = rbasis[idx];
      if(rsec.sym != symB){
         cout << "error: symmetry does not match!" << endl;
         exit(1);
      }
      if(dimAs != 1){
         cout << "error: dimAs=" << dimAs << " is not 1!" << endl;
         exit(1);
      }
      // construct <nm|psi>
      matrix<double> vrl(dimBs,nroots);
      for(int iroot = 0; iroot<nroots; iroot++){
         for(const auto& t : qlst[symB]){
            int ib = get<0>(t);
            int id = get<2>(t);
            vrl(ib,iroot) = vs2[iroot][id];
         }
      }
      // compute key
      auto it0 = qmapA[symB].begin();
      onstate state0 = pspace2.spaceA[it0->first];
      qsym sym0(state0.nelec(),state0.nelec_a());
      auto key = make_tuple(sym0,sym_state,symB);
      // c[n](i,r) = <nr|psi[i]> = <nb|psi[i]> [vlr(b,i)] * W(b,r)
      rwavefuns.qblocks[key].push_back(xgemm("T","N",vrl,rsec.coeff));
      idx++;
   } // symB sectors
   if(debug) rwavefuns.print("rwavefuns",1);
   return rwavefuns;
}

// exact boundary tensor:
//        n             |vac>
//        |               |
//     ---*---|vac>   n---*
//  |out> 	          |
qtensor3 comb::get_rbsite() const{
   qtensor3 qt3(qsym(0,0),phys_qsym_space,phys_qsym_space,vac_qsym_space);
   for(int k=0; k<4; k++){
      auto key = make_tuple(phys_sym[k],phys_sym[k],phys_sym[0]);
      qt3.qblocks[key][0] = identity_matrix<double>(1);
   }
   return qt3;
}

qtensor3 comb::get_lbsite() const{
   vector<bool> dir = {1,1,0};
   qtensor3 qt3(qsym(0,0),phys_qsym_space,vac_qsym_space,phys_qsym_space,dir);
   for(int k=0; k<4; k++){
      auto key = make_tuple(phys_sym[k],phys_sym[0],phys_sym[k]);
      qt3.qblocks[key][0] = identity_matrix<double>(1);
   }
   return qt3;
}

// build site tensor from {|r>} bases
void comb::rcanon_init(const onspace& space,
		       const vector<vector<double>>& vs,
		       const double thresh_proj){
   const bool debug = false;
   auto t0 = tools::get_time();
   cout << "\ncomb::rcanon_init" << endl;
   // compute renormalized bases {|r>}
   get_rbases(space, vs, thresh_proj);
   // loop over sites
   for(int idx=0; idx<ntotal-1; idx++){
      auto p = rcoord[idx];
      int i = p.first, j = p.second;
      if(debug){
         cout << "\nidx=" << idx 
	      << " node=(" << i << "," << j << ")[" << topo[i][j] << "] "
	      << endl;
      }
      auto& rbasis = rbases[p]; 
      qtensor3 rt;
      if(type[p] == 0){
	 
	 if(debug) cout << "type 0: end or leaves" << endl;
         rt = get_rbsite();

      }else if(type[p] == 1 || type[p] == 2){

	 rt.qmid = phys_qsym_space;
	 pair<int,int> p0;
	 if(type[p] == 1){
	    //     n      
	    //    \|/      
	    //  -<-*-<-|r> 
	    if(debug) cout << "type 1: physical site on backbone" << endl;
	    p0 = make_pair(i+1,0);
	 }else if(type[p] == 2){
	    //     |u>
	    //     \|/
	    //  n-<-*
	    //     \|/
	    if(debug) cout << "type 2: physical site on branch" << endl;
	    p0 = make_pair(i,j+1);
	 }
	 // load rbasis for previous site
	 auto& rbasis1 = rbases[p0];
	 // loop over symmetry blocks of out index 
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qrow[sym] = rbasis[k].coeff.cols();
	    // loop over symmetry blocks of in index 
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qcol[sym1] = rbasis1[k1].coeff.cols();
	       // loop over physical indices
	       for(int k0=0; k0<4; k0++){
		  auto key = make_tuple(phys_sym[k0],sym,sym1);
		  rt.qblocks[key] = empty_block;
		  if(sym == sym1 + phys_sym[k0]){
		     // B[i](b1,b)=<ci,b1|b>
		     auto Bi = get_Bmatrix<double>(phys_space[k0],rbasis1[k1].space,rbasis[k].space);
		     // BL[i](b1,r)=<ci,b1|b> W(b,r)
		     auto BL = xgemm("N","N",Bi,rbasis[k].coeff);
		     // BLR[i](r,l)= W(b,l)<ci,b1|r> = BL^T*W
		     auto BLR = xgemm("T","N",BL,rbasis1[k1].coeff); 
		     rt.qblocks[key].push_back(BLR);
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
	 // loop over symmetry blocks of out index
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qrow[sym] = rbasis[k].coeff.cols();
	    // loop over right blocks of in index
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qcol[sym1] = rbasis1[k1].coeff.cols();
	       // loop over upper indices
	       for(int k0=0; k0<rbasis0.size(); k0++){
	          auto sym0 = rbasis0[k0].sym;
		  int nbas = rbasis0[k0].coeff.rows();
		  int ndim = rbasis0[k0].coeff.cols();
		  auto key = make_tuple(sym0,sym,sym1);
	    	  rt.qmid[sym0] = ndim;
	          rt.qblocks[key] = empty_block;
		  // symmetry conversing combination
		  if(sym == sym1 + sym0){
		     vector<matrix<double>> Wlr(ndim);
		     for(int ibas=0; ibas<nbas; ibas++){
			auto state0 = rbasis0[k0].space[ibas];
		        auto Bi = get_Bmatrix<double>(state0,rbasis1[k1].space,rbasis[k].space);
		        auto BL = xgemm("N","N",Bi,rbasis[k].coeff);
		        auto BLR = xgemm("T","N",BL,rbasis1[k1].coeff);
			if(ibas == 0){
		           for(int idim=0; idim<ndim; idim++){
			      Wlr[idim] = rbasis0[k0].coeff(ibas,idim)*BLR;
			   } // idim
			}else{
		           for(int idim=0; idim<ndim; idim++){
			      Wlr[idim] += rbasis0[k0].coeff(ibas,idim)*BLR;
			   } // idim
			}
		     } // ibas
		     rt.qblocks[key] = Wlr;
		  }
	       } // k0
	    } // k1
	 } // k

      } // type[p]
      if(debug) rt.print("rsites_"+to_string(idx));
      rsites[p] = rt;
   } // idx
   // compute wave function at the start for right canonical form 
   auto p = make_pair(0,0), p0 = make_pair(1,0);
   rsites[p] = get_rwavefuns(space, vs, rsupport[p], rbases[p0]);
   auto t1 = tools::get_time();
   cout << "\ntiming for comb::rcanon_init : " << setprecision(2) 
        << tools::get_duration(t1-t0) << " s" << endl;
}

void comb::rcanon_check(const double thresh_ortho,
		        const bool ifortho){
   cout << "\ncomb::rcanon_check thresh_ortho=" << thresh_ortho << endl;
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      auto qt2 = contract_qt3_qt3_cr(rsites[p],rsites[p]);
      int Dtot = qt2.get_dim_row();
      int i = p.first, j = p.second;
      double mdiff = qt2.check_identity(thresh_ortho, false);
      cout << "idx=" << idx 
           << " node=(" << i << "," << j << ")[" << topo[i][j] << "]"
           << " Dtot=" << Dtot << " mdiff=" << mdiff << endl;
      if((ifortho || (!ifortho && idx != ntotal-1)) 
         && (mdiff>thresh_ortho)){
         cout << "error: deviate from identity matrix!" << endl;
         exit(1);
      }
   } // idx
}
