#ifndef CTNS_COMB_INIT_H
#define CTNS_COMB_INIT_H

#include "../core/tools.h"
#include "../core/onspace.h"
#include "ctns_comb.h"
#include "ctns_bipart.h"
#include "ctns_phys.h"

namespace ctns{

/*
// compute wave function at the start for right canonical form 
qtensor3 comb::get_rwavefuns(const onspace& space,
		             const vector<vector<double>>& vs,
		             const vector<int>& order,
		             const renorm_basis& rbasis){
   bool debug = false;
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
      auto key = make_tuple(sym0,sym_state,symB);
      // c[n](i,r) = <nr|psi[i]> = <nb|psi[i]> [vlr(b,i)] * W(b,r)
      rwavefuns.qblocks[key].push_back(dgemm("T","N",vrl,rsec.coeff));
      idx++;
   } // symB sectors
   if(debug) rwavefuns.print("rwavefuns",1);
   return rwavefuns;
}

*/

// compute renormalized bases {|r>} from SCI wavefunctions 
template <typename Tm>
void get_rbases(comb<Tm>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<Tm>>& vs,
		const double thresh_proj){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rbases thresh_proj=" << std::scientific << thresh_proj << std::endl;
   // loop over nodes (except the last one)
   for(int idx=0; idx<icomb.topo.rcoord.size()-1; idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = icomb.topo.nodes[i][j];
      if(debug){
	 std::cout << "\nidx=" << idx << " node=" << p; 
	 std::cout << " rsupport=";
         for(int k : node.rsupport) std::cout << k << " ";
	 std::cout << std::endl;
      }
      if(node.type == 0){
         icomb.rbases[p] = get_rbasis_phys<Tm>();
      }else{
	 // Generate {|r>} at the internal nodes
         // 1. generate 1D ordering
         const auto& rsupp = node.rsupport; 
         auto order = node.lsupport;
         int bpos = order.size(); // must be put here to account bipartition position
         copy(rsupp.begin(), rsupp.end(), back_inserter(order));
         if(debug){
            std::cout << "bpos=" << bpos;
	    std::cout << " order=";
            for(int k : order) std::cout << k << " ";
	    std::cout << std::endl;
         }
         // 2. transform SCI coefficient
	 fock::onspace space2;
	 std::vector<std::vector<Tm>> vs2;
         transform_coeff(space, vs, order, space2, vs2); 
         // 3. bipartition of space and compute renormalized states
         icomb.rbases[p] = right_projection(space2, vs2, 2*bpos, thresh_proj, debug);
      }
   } // idx
   // print information for all renormalized basis {|r>} at each bond
   std::cout << "\nfinal rbases with thresh_proj = " << thresh_proj << std::endl;
   int Dmax = 0;
   for(int idx=0; idx<icomb.topo.rcoord.size()-1; idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto shape = get_shape(icomb.rbases[p]);
      std::cout << "idx=" << idx << " node=" << p
           << " shape=" << shape.first << "," << shape.second 
           << std::endl;
      Dmax = std::max(Dmax,shape.second);
   } // idx
   std::cout << "maximum bond dimension = " << Dmax << std::endl;
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::get_rbases : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

// build site tensor from {|r>} bases
template <typename Tm>
void get_rsites(comb<Tm>& icomb){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rsites" << std::endl;
   // loop over sites
   for(int idx=0; idx<icomb.topo.rcoord.size()-1; idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = icomb.topo.nodes[i][j];
      if(debug) std::cout << "\nidx=" << idx << " node=" << p << " ";      
      if(node.type == 0){
	 
	 if(debug) std::cout << "type 0: end or leaves" << std::endl;
         icomb.rsites[p] = get_right_bsite<Tm>();

      }else{
	    
	 if(debug){
	    if(node.type == 3){
	       //    |u>(0)      
	       //     |
	       //  ---*---|r>(1) 
	       std::cout << "type 3: internal site on backbone" << std::endl;
	    }else{
	       //     n            |u> 
	       //     |             |
	       //  ---*---|r>   n---*
	       //                   |
	       std::cout << "type 1/2: physical site on backbone/branch" << std::endl;
	    }
	 }
         const auto& rbasis_l = icomb.rbases[p]; 
	 const auto& rbasis_c = node.type==3? icomb.rbases[node.center] : get_rbasis_phys<Tm>(); 
	 const auto& rbasis_r = icomb.rbases[node.right];
	 auto qmid = get_qsym_space(rbasis_c);
	 auto qrow = get_qsym_space(rbasis_l); 
	 auto qcol = get_qsym_space(rbasis_r);
	 qtensor3<Tm> qt3(qsym(0,0), qmid, qrow, qcol);
	 for(int kl=0; kl<rbasis_l.size(); kl++){ // left
            for(int kr=0; kr<rbasis_r.size(); kr++){ // right 
	       for(int kc=0; kc<rbasis_c.size(); kc++){ // upper 
		  auto& blk = qt3(kc,kl,kr);
	          if(blk.size() == 0) continue;
		  // construct site R[c][lr] = <qc,qr|ql> = W*[c'c] W*[r'r] <D[c'],D[r']|D[l']> W[l',l]
		  auto Wc = rbasis_c[kc].coeff.H();
		  auto Wr = rbasis_r[kr].coeff.conj();
		  auto Wl = rbasis_l[kl].coeff; 
		  for(int dc=0; dc<Wc.cols(); dc++){
		     auto state_c = rbasis_c[kc].space[dc];
		     // tmp1[c'][r'l'] = <D[c'],D[r']|[l']>
		     auto tmp1 = fock::get_Bmatrix<Tm>(state_c,rbasis_r[kr].space,rbasis_l[kl].space);
		     // tmp2[c'][r'l] = tmp1[c'][r'l']Wl[l'l]
		     auto tmp2 = linalg::xgemm("N","N",tmp1,Wl);
		     // tmp3[c'](l,r)= Wr*[r'r]tmp2[c'][r'l] = tmp2^T*Wr.conj() 
		     auto tmp3 = linalg::xgemm("T","N",tmp2,Wr);
		     // R[c][lr] = sum_c' Wc*[c'c]tmp3[c'][lr]
		     for(int ic=0; ic<Wc.rows(); ic++){
		        blk[ic] += Wc(ic,dc)*tmp3;
		     } // ic
		  } // ibas
	       } // kc
	    } // kr
	 } // kl
         icomb.rsites[p] = std::move(qt3);
      } // type[p]
      if(debug) icomb.rsites[p].print("rsites_"+std::to_string(idx));
   } // idx
/*
   // compute wave function at the start for right canonical form 
   auto p = make_pair(0,0), p0 = make_pair(1,0);
   rsites[p] = get_rwavefuns(space, vs, rsupport[p], rbases[p0]);
*/
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::get_rsites: " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

template <typename Tm>
void rcanon_check(comb<Tm>& icomb,
		  const double thresh_ortho,
		  const bool ifortho){
   std::cout << "\nctns::rcanon_check thresh_ortho=" << thresh_ortho << std::endl;
   int ntotal = icomb.topo.rcoord.size();
   for(int idx=0; idx<ntotal; idx++){
      auto p = icomb.topo.rcoord[idx];
      // check right canonical form
      auto qt2 = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
      int Dtot = qt2.qrow.get_dimAll();
      double maxdiff = qt2.check_identity(thresh_ortho, false);
      std::cout << "idx=" << idx << " node=" << p 
                << " Dtot=" << Dtot << " maxdiff=" << maxdiff << std::endl;
      if((ifortho || (!ifortho && idx != ntotal-1)) && (maxdiff>thresh_ortho)){
	 std::cout << "error: deviate from identity matrix!" << std::endl;
         exit(1);
      }
   } // idx
}

// initialize RCF from SCI wavefunctions
template <typename Tm>
void rcanon_init(comb<Tm>& icomb,
		 const fock::onspace& space,
		 const std::vector<std::vector<Tm>>& vs,
		 const double thresh_proj){
   auto t0 = tools::get_time();
   std::cout << "\nctns::rcanon_init" << std::endl;
   // compute renormalized bases {|r>} from SCI wavefunctions
   get_rbases(icomb, space, vs, thresh_proj); 
   // form sites from rbases
   get_rsites(icomb); 
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::rcanon_init : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // ctns

#endif
