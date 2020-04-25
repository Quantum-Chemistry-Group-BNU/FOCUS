#include "../settings/global.h"
#include "../core/linalg.h"
#include "tns_qsym.h"
#include "tns_comb.h"
#include "tns_ordering.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace tns;
using namespace fock;
using namespace linalg;

// compute renormalized bases {|r>} 
comb_rbases comb::get_rbases(const onspace& space,
		    	     const vector<vector<double>>& vs,
		    	     const double thresh_proj){
   auto t0 = global::get_time();
   bool debug = true;
   cout << "\ncomb::get_rbases thresh_proj=" << scientific << thresh_proj << endl;
   comb_rbases rbases;
   vector<pair<int,int>> shapes; // for debug
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
      // debug
      {
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
   auto t1 = global::get_time();
   cout << "\ntiming for comb::get_rbases : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
   return rbases;
}

// compute wave function at the start for right canonical form 
qtensor3 comb::get_rwfuns(const onspace& space,
		          const vector<vector<double>>& vs,
		          const vector<int>& order,
		          const renorm_basis& rbasis){
   bool debug = true;
   cout << "\ncomb::get_rwfuns" << endl;
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
   // construct rwfuns 
   qtensor3 rwfuns;
   rwfuns.qspace0 = phys_qsym_space;
   // assuming the symmetry of wavefunctions are the same
   qsym sym_state(space[0].nelec(), space[0].nelec_a());
   int nroots = vs2.size();
   rwfuns.qspace[sym_state] = nroots;
   // init empty blocks for all combinations 
   int idx = 0;
   for(auto it = qsecB.cbegin(); it != qsecB.cend(); ++it){
      auto& symB = it->first;
      rwfuns.qspace1[symB] = rbasis[idx].coeff.cols();
      for(int k0=0; k0<4; k0++){
	 auto key = make_tuple(phys_sym[k0],symB,sym_state);
         rwfuns.qblocks[key] = empty_block; 
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
      matrix vrl(dimBs,nroots);
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
      auto key = make_tuple(sym0,symB,sym_state);
      // c[n][r,i] = <nr|psi[i]> = W(b,r)*<nb|psi[i]>(b,i)
      rwfuns.qblocks[key].push_back(dgemm("T","N",rsec.coeff,vrl));
      idx++;
   } // symB sectors
   if(debug) rwfuns.print("wavefuns",2);
   return rwfuns;
}

// build site tensor from {|r>} bases
void comb::rcanon_init(const onspace& space,
		       const vector<vector<double>>& vs,
		       const double thresh_proj,
		       const double thresh_ortho){
   auto t0 = global::get_time();
   bool debug = true;
   cout << "\ncomb::rcanon_init" << endl;
   // compute renormalized bases {|r>}
   auto rbases = get_rbases(space, vs, thresh_proj);
   // compute wave function at the start for right canonical form 
   rsites[make_pair(0,0)] = get_rwfuns(space, vs, rsupport[make_pair(0,0)], 
		   		       rbases[make_pair(1,0)]);
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
	 
	 //       n             |vac>
	 //      \|/             \|/
	 //    -<-*-<-|vac>   n-<-*
	 //    			 \|/
	 if(debug) cout << "type 0: end or leaves" << endl; 
	 rt.qspace0 = phys_qsym_space; 
	 rt.qspace1[phys_sym[0]] = 1; // |vac> in
	 for(int k=0; k<4; k++){
	    rt.qspace[phys_sym[k]] = 1; // out
	    for(int k0=0; k0<4; k0++){
	       auto key = make_tuple(phys_sym[k0],phys_sym[0],phys_sym[k]);
	       rt.qblocks[key] = empty_block;
	       if(k0 == k) rt.qblocks[key].push_back(identity_matrix(1));
	    } // k0
	 } // k

      }else if(type[p] == 1 || type[p] == 2){

	 rt.qspace0 = phys_qsym_space;
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
	    rbasis1 = get_rbasis_phys(); // we use exact repr. for this bond 
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
	       for(int k0=0; k0<4; k0++){
		  auto key = make_tuple(phys_sym[k0],sym1,sym);
		  rt.qblocks[key] = empty_block;
		  if(sym == sym1 + phys_sym[k0]){
		     auto Bi = get_Bmatrix(phys_space[k0],rbasis1[k1].space,rbasis[k].space);
		     auto BL = dgemm("N","N",Bi,rbasis[k].coeff);
		     auto RBL = dgemm("T","N",rbasis1[k1].coeff,BL);
		     rt.qblocks[key].push_back(RBL);
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
	 // loop over symmetry blocks
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qspace[sym] = rbasis[k].coeff.cols();
	    // loop over right blocks
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qspace1[sym1] = rbasis1[k1].coeff.cols();
	       // loop over upper indices
	       for(int k0=0; k0<rbasis0.size(); k0++){
	          auto sym0 = rbasis0[k0].sym;
		  int nbas = rbasis0[k0].coeff.rows();
		  int ndim = rbasis0[k0].coeff.cols();
		  auto key = make_tuple(sym0,sym1,sym); 
	    	  rt.qspace0[sym0] = ndim;
	          rt.qblocks[key] = empty_block;
		  // symmetry conversing combination
		  if(sym == sym1 + sym0){
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
		     rt.qblocks[key] = Wrl;
		  }
	       } // k0
	    } // k1
	 } // k

      } // type[p]
      if(debug) rt.print("rsites_"+to_string(idx));
      rsites[p] = rt;
   } // idx
   // debug
   {
      cout << "\ncheck orthogonality for right canonical sites:" << endl;
      for(int idx=0; idx<ntotal; idx++){
         auto p = rcoord[idx];
         int i = p.first, j = p.second;
         {
            cout << "\nidx=" << idx 
                 << " node=(" << i << "," << j << ")[" << topo[i][j] << "] "
                 << endl;
         }
         auto& rt = rsites[p]; 
         int Dtot = 0;
	 // loop over out blocks
         for(const auto& pr : rt.qspace){
            auto& sym = pr.first;
            int ndim = pr.second;
            Dtot += ndim;
            matrix Sr(ndim,ndim);
            // S[r,r'] = \sum_{l,c} Ac[l,r]*Ac[l,r']
	    // loop over upper blocks 
            for(const auto& p1 : rt.qspace1){
               auto& sym1 = p1.first;
	       // loop over in blocks
	       for(const auto& p0 : rt.qspace0){
		  auto& sym0 = p0.first;
		  auto& blk = rt.qblocks[make_tuple(sym0,sym1,sym)];
                  if(blk.size() == 0) continue; 
		  int ndim0 = p0.second;
                  for(int i=0; i<ndim0; i++){
                     Sr += dgemm("N","N",blk[i].transpose(),blk[i]);
                  }
	       } // p0
            } // p1
            auto diff = normF(Sr - identity_matrix(ndim));
            cout << " qsym=" << sym << " ndim=" << ndim 
		 << " |Sr-Id|_F=" << diff << endl;
            if(diff > thresh_ortho){
               Sr.print("Sr_sym"+sym.to_string());
               cout << "error: deviate from identity matrix! diff=" << diff << endl;
               exit(1);
            }
         } // sym blocks
         cout << "total bond dimension=" << Dtot << endl;
      } // idx
   } // debug
   auto t1 = global::get_time();
   cout << "\ntiming for comb::rcanon_init : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

// <n|Comb[i]>
vector<double> comb::rcanon_coeff(const onstate& state){
   int n = rsites[make_pair(0,0)].get_dim();
   vector<double> coeff(n);
   // compute fermionic sign changes
   auto sgn = state.permute_sgn(image2);
   // compute <n'|Comb> by contracting all sites
   qsym sym_vac(0,0);
   qsym sym_p, sym_l, sym_r, sym_u, sym_d;
   matrix mat_r, mat_u;
   sym_r = sym_vac;
   for(int i=nbackbone-1; i>=0; i--){
      int tp = type[make_pair(i,0)];
      // site on backbone with physical index
      if(tp == 0 || tp == 1){
         int orb = topo[i][0];
	 int na = state[2*orb], nb = state[2*orb+1];
	 qsym sym_p(na+nb,na);
         sym_l = sym_p + sym_r;
	 auto key = make_tuple(sym_p,sym_r,sym_l);
	 matrix mat = rsites[make_pair(i,0)].qblocks[key][0];
	 if(i==nbackbone-1){
	    mat_r = mat;
         }else{
            mat_r = dgemm("N","N",mat_r,mat); // (in,out),(out,out')->(in,out')
	 }
	 sym_r = sym_l; // update sym_r (in)
      }else if(tp == 3){
	 // propogate symmetry from leaves down to backbone
	 sym_u = sym_vac;
         for(int j=topo[i].size()-1; j>=1; j--){
	    int orb = topo[i][j];
	    int na = state[2*orb], nb = state[2*orb+1];
	    qsym sym_p(na+nb,na);
	    sym_d = sym_p + sym_u;
	    auto key = make_tuple(sym_p,sym_u,sym_d);
	    matrix mat = rsites[make_pair(i,j)].qblocks[key][0];
	    if(j==topo[i].size()-1){
	       mat_u = mat;
	    }else{
	       mat_u = dgemm("N","N",mat_u,mat);
	    }
	    sym_u = sym_d; // update sym_u (in)
         } // j
	 // internal site without physical index
	 //       u
	 //       |
	 // l--<--*--<--r
	 sym_l = sym_u + sym_r;
	 auto key = make_tuple(sym_u,sym_r,sym_l);
	 auto& blk = rsites[make_pair(i,0)].qblocks[key];
	 int dim_r = blk[0].rows(), dim_l = blk[0].cols(); 
	 // contract upper sites
	 matrix mat(dim_r,dim_l);
	 int dim_u = rsites[make_pair(i,0)].qspace0[sym_u];
	 for(int k=0; k<dim_u; k++){
	    mat += mat_u(0,k)*blk[k];
	 }
	 // contract right matrix
	 mat_r = dgemm("N","N",mat_r,mat);
	 sym_r = sym_l;
      } // tp
   } // j
   assert(mat_r.rows() == 1 && mat_r.cols() == n);
   for(int j=0; j<n; j++){
      coeff[j] = sgn*mat_r(0,j);
   }
   return coeff;
}

// ovlp[n,m] = <Comb[n]|SCI[m]>
matrix comb::rcanon_ovlp(const onspace& space,
	                 const vector<vector<double>>& vs){
   int n = rsites[make_pair(0,0)].get_dim();
   int dim = space.size();
   matrix cmat(n,dim);
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_coeff(space[i]);
      copy(coeff.begin(),coeff.end(),cmat.col(i));
   };
   // ovlp(m,n) = vs(dim,m)*mcoeff(n,dim)
   int m = vs.size();
   matrix vmat(dim,m);
   for(int im=0; im<m; im++){
      copy(vs[im].begin(),vs[im].end(),vmat.col(im));
   }
   matrix ovlp = dgemm("N","N",cmat,vmat);
   return ovlp;
}
