#include "../settings/global.h"
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
   bool debug = false;
   auto t0 = global::get_time();
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
      // 1. generate 1D ordering
      auto rsupp = rsupport[make_pair(i,j)]; // original order required [IMPORTANT]
      auto order = support_rest(rsupp);
      int pos = order.size(); // must be put here to account bipartition position
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
}

// compute wave function at the start for right canonical form 
qtensor3 comb::get_rwfuns(const onspace& space,
		          const vector<vector<double>>& vs,
		          const vector<int>& order,
		          const renorm_basis& rbasis){
   bool debug = false;
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
   rwfuns.qmid = phys_qsym_space;
   // assuming the symmetry of wavefunctions are the same
   qsym sym_state(space[0].nelec(), space[0].nelec_a());
   int nroots = vs2.size();
   rwfuns.qcol[sym_state] = nroots;
   // init empty blocks for all combinations 
   int idx = 0;
   for(auto it = qsecB.cbegin(); it != qsecB.cend(); ++it){
      auto& symB = it->first;
      rwfuns.qrow[symB] = rbasis[idx].coeff.cols();
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
      // c[n][r,i] = <nr|psi[i]> = W(b,r)*<nb|psi[i]> [vlr(b,i)] 
      rwfuns.qblocks[key].push_back(dgemm("T","N",rsec.coeff,vrl));
      idx++;
   } // symB sectors
   if(debug) rwfuns.print("rwfuns",1);
   return rwfuns;
}

// build site tensor from {|r>} bases
void comb::rcanon_init(const onspace& space,
		       const vector<vector<double>>& vs,
		       const double thresh_proj){
   bool debug = false;
   auto t0 = global::get_time();
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
	 
	 //       n             |vac>
	 //      \|/             \|/
	 //    -<-*-<-|vac>   n-<-*
	 //    			 \|/
	 if(debug) cout << "type 0: end or leaves" << endl; 
	 rt.qmid = phys_qsym_space; 
	 // in index: |vac> 
	 rt.qrow = vacuum;
	 // loop over symmetry blocks of out index 
	 for(int k=0; k<4; k++){
	    rt.qcol[phys_sym[k]] = 1; // out
	    // loop over physical indices
	    for(int k0=0; k0<4; k0++){
	       auto key = make_tuple(phys_sym[k0],phys_sym[0],phys_sym[k]);
	       rt.qblocks[key] = empty_block;
	       if(k0 == k) rt.qblocks[key].push_back(identity_matrix(1));
	    } // k0
	 } // k

      }else if(type[p] == 1 || type[p] == 2){

	 rt.qmid = phys_qsym_space;
	 pair<int,int> p0;
	 if(type[p] == 1){
	    //       n      
	    //      \|/      
	    //    -<-*-<-|r> 
	    if(debug) cout << "type 1: physical site on backbone" << endl;
	    p0 = make_pair(i+1,0);
	 }else if(type[p] == 2){
	    //      |u>
	    //      \|/
	    //   n-<-*
	    //      \|/
	    if(debug) cout << "type 2: physical site on branch" << endl;
	    p0 = make_pair(i,j+1);
	 }
         renorm_basis rbasis1; 
	 if(type[p] == 0){
	    rbasis1 = get_rbasis_phys(); // we use exact repr. for this bond 
	 }else{
	    rbasis1 = rbases[p0];
	 }
	 // loop over symmetry blocks of out index 
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qcol[sym] = rbasis[k].coeff.cols();
	    // loop over symmetry blocks of in index 
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qrow[sym1] = rbasis1[k1].coeff.cols();
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
	 // loop over symmetry blocks of out index
	 for(int k=0; k<rbasis.size(); k++){
	    auto sym = rbasis[k].sym;
	    rt.qcol[sym] = rbasis[k].coeff.cols();
	    // loop over right blocks of in index
            for(int k1=0; k1<rbasis1.size(); k1++){
	       auto sym1 = rbasis1[k1].sym;
	       rt.qrow[sym1] = rbasis1[k1].coeff.cols();
	       // loop over upper indices
	       for(int k0=0; k0<rbasis0.size(); k0++){
	          auto sym0 = rbasis0[k0].sym;
		  int nbas = rbasis0[k0].coeff.rows();
		  int ndim = rbasis0[k0].coeff.cols();
		  auto key = make_tuple(sym0,sym1,sym);
	    	  rt.qmid[sym0] = ndim;
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
   // compute wave function at the start for right canonical form 
   auto p = make_pair(0,0), p0 = make_pair(1,0);
   rsites[p] = get_rwfuns(space, vs, rsupport[p], rbases[p0]);
   auto t1 = global::get_time();
   cout << "\ntiming for comb::rcanon_init : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void comb::rcanon_check(const double thresh_ortho,
		        const bool ifortho){
   cout << "\ncomb::rcanon_check thresh_ortho=" << thresh_ortho << endl;
   for(int idx=0; idx<ntotal; idx++){
      auto p = rcoord[idx];
      auto qt2 = contract_qt3_qt3_lc(rsites[p],rsites[p]);
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

// <det|Comb[n]> by contracting the Comb
vector<double> comb::rcanon_CIcoeff(const onstate& state){
   // compute fermionic sign changes
   auto sgn = state.permute_sgn(image2);
   // compute <n'|Comb> by contracting all sites
   qsym sym_vac(0,0);
   qsym sym_p, sym_l, sym_r, sym_u, sym_d;
   matrix mat_r, mat_u;
   sym_r = sym_vac;
   // loop over sites on backbone
   for(int i=nbackbone-1; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = type[p];
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
         int k = topo[i][0];
	 int na = state[2*k], nb = state[2*k+1];
	 qsym sym_p(na+nb,na);
         sym_l = sym_p + sym_r;
	 auto key = make_tuple(sym_p,sym_r,sym_l);
	 matrix mat = rsites[p].qblocks[key][0];
	 if(i==nbackbone-1){
	    mat_r = mat;
         }else{
	    // (in,out'),(out',out)->(in,out)
	    mat_r = dgemm("N","N",mat_r,mat); 
	 }
	 sym_r = sym_l; // update sym_r (in)
      }else if(tp == 3){
	 // propogate symmetry from leaves down to backbone
	 sym_u = sym_vac;
         for(int j=topo[i].size()-1; j>=1; j--){
	    int k = topo[i][j];
	    int na = state[2*k], nb = state[2*k+1];
	    qsym sym_p(na+nb,na);
	    sym_d = sym_p + sym_u;
	    auto key = make_tuple(sym_p,sym_u,sym_d);
	    matrix mat = rsites[make_pair(i,j)].qblocks[key][0];
	    if(j==topo[i].size()-1){
	       mat_u = mat;
	    }else{
	       // (in,out'),(out',out)->(in,out)
	       mat_u = dgemm("N","N",mat_u,mat);
	    }
	    sym_u = sym_d; // update sym_u (in)
         } // j
	 // deal with internal site without physical index
	 sym_l = sym_u + sym_r;
	 auto key = make_tuple(sym_u,sym_r,sym_l);
	 auto& blk = rsites[p].qblocks[key];
	 int dim_r = blk[0].rows(), dim_l = blk[0].cols(); 
	 matrix mat(dim_r,dim_l);
	 // contract upper sites
	 int dim_u = rsites[p].qmid[sym_u];
	 for(int k=0; k<dim_u; k++){
	    // (in,c)*(c,l,r)->(in=1,l,r)
	    mat += mat_u(0,k)*blk[k]; 
	 }
	 // contract right matrix
	 mat_r = dgemm("N","N",mat_r,mat);
	 sym_r = sym_l;
      } // tp
   } // j
   int n = rsites[make_pair(0,0)].get_dim_col();
   assert(mat_r.rows() == 1 && mat_r.cols() == n);
   vector<double> coeff(n);
   for(int j=0; j<n; j++){
      coeff[j] = sgn*mat_r(0,j);
   }
   return coeff;
}

// ovlp[m,n] = <SCI[m]|Comb[n]>
matrix comb::rcanon_CIovlp(const onspace& space,
	                   const vector<vector<double>>& vs){
   cout << "\ncomb::rcanon_CIovlp" << endl;
   int n = rsites[make_pair(0,0)].get_dim_col();
   int dim = space.size();
   // cmat(n,d) = <d|Comb[n]>
   matrix cmat(n,dim);
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_CIcoeff(space[i]);
      copy(coeff.begin(),coeff.end(),cmat.col(i));
   };
   // ovlp(m,n) = vs(d,m)*cmat(n,d)
   int m = vs.size();
   matrix vmat(dim,m);
   for(int im=0; im<m; im++){
      copy(vs[im].begin(),vs[im].end(),vmat.col(im));
   }
   auto ovlp = dgemm("T","T",vmat,cmat);
   return ovlp;
}
