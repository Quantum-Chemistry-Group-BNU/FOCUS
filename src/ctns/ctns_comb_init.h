#ifndef CTNS_COMB_INIT_H
#define CTNS_COMB_INIT_H

#include "../core/tools.h"
#include "../core/onspace.h"
#include "ctns_comb.h"
#include "ctns_bipart.h"
#include "ctns_phys.h"

namespace ctns{

// compute renormalized bases {|r>} from SCI wavefunctions 
template <typename Tm>
void get_rbases(comb<Tm>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<Tm>>& vs,
		const double thresh_proj){
   const bool debug = false;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rbases thresh_proj=" << std::scientific << thresh_proj << std::endl;
   // loop over nodes (except the last one)
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = icomb.topo.nodes[i][j];
      if(debug){
	 std::cout << "\nidx=" << idx << " node=" << p; 
	 std::cout << " rsupport=";
         for(int k : node.rsupport) std::cout << k << " ";
	 std::cout << std::endl;
      }
      if(node.type == 0 && p != std::make_pair(0,0)){
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
         icomb.rbases[p] = right_projection(2*bpos, space2, vs2, thresh_proj, debug);
      }
   } // idx
   // print information for all renormalized basis {|r>} at each bond
   std::cout << "\nfinal rbases with thresh_proj=" << thresh_proj << std::endl;
   int Dmax = 0;
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      // shape can be different from dim(rspace) if associated weight is zero!
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
   const bool debug = false;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rsites" << std::endl;
   // loop over sites
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = icomb.topo.nodes[i][j];
      if(debug) std::cout << "\nidx=" << idx << " node=" << p << " ";     
      if(node.type == 0 && p != std::make_pair(0,0)){
	 
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
	 const auto& rbasis_c = (node.type==3)? icomb.rbases[node.center] : get_rbasis_phys<Tm>(); 
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
		  // construct site R[c][lr] = <qc,qr|ql> 
		  // 			     = W*[c'c] W*[r'r] <D[c'],D[r']|D[l']> W[l',l]
		  auto Wc = rbasis_c[kc].coeff.H();
		  auto Wr = rbasis_r[kr].coeff.conj();
		  auto Wl = rbasis_l[kl].coeff; 
		  for(int dc=0; dc<Wc.cols(); dc++){
		     auto state_c = rbasis_c[kc].space[dc];
		     // tmp1[c'][r'l'] = <D[c'],D[r']|[l']>
		     auto tmp1 = fock::get_Bcouple<Tm>(state_c,rbasis_r[kr].space,rbasis_l[kl].space);
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
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::get_rsites : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

// compute wave function at the start for right canonical form
template <typename Tm>
void get_rwfuns(comb<Tm>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<Tm>>& vs,
		const double thresh_proj){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rwfuns thresh_proj=" << std::scientific << thresh_proj << std::endl;
   // qrow: we assume all the states are of the same symmetry
   fock::onstate det = space[0];
   const bool Htype = tools::is_complex<Tm>();
   auto sym_states = get_qsym<Tm>(space[0]);
   for(int i=0; i<space.size(); i++){
      auto sym = get_qsym<Tm>(space[i]);
      if(sym != sym_states){
         std::cout << "error: symmetry is different in space!" << std::endl;
	 std::cout << "sym_states=" << sym_states 
		   << " det=" << space[i] << " sym=" << sym
		   << std::endl;
	 exit(1);
      }
   }
   int nroot = vs.size(); 
   qsym_space qrow({{sym_states, nroot}});
   // qcol
   auto& rbasis = icomb.rbases[std::make_pair(0,0)];
   auto qcol = get_qsym_space(rbasis);
   // rwfuns[l,r]
   qtensor2<Tm> rwfuns(qsym(0,0), qrow, qcol);
   // find the match position for qcol in qrow
   int cpos = -1; 
   for(int bc=0; bc<qcol.size(); bc++){
      if(sym_states == qcol.get_sym(bc)){
	 cpos = bc;
	 break;
      }
   }
   assert(cpos != -1);
   std::map<fock::onstate,int> index; // index of a state
   int idx = 0;
   for(const auto& state : rbasis[cpos].space){
      index[state] = idx;
      idx++;
   }
   // setup wavefunction: map vs2 to correct position
   fock::onspace space2;
   std::vector<std::vector<Tm>> vs2;
   const auto& order = icomb.topo.nodes[0][0].rsupport;
   // transform SCI coefficient to order
   transform_coeff(space, vs, order, space2, vs2); 
   const auto& rbas = rbasis[cpos].coeff;
   linalg::matrix<Tm> wfs(rbas.rows(), nroot);
   for(int i=0; i<space2.size(); i++){
      int ir = index.at(space2[i]);
      for(int iroot=0; iroot<nroot; iroot++){
         wfs(ir,iroot) = vs2[iroot][i];
      } // iroot
   } // i
   // construct the boundary matrix: |psi[i]> = \sum_a |rbas[a]><rbas[a]|psi[i]>
   // In RCF the site is defined as 
   //    W[i,a] =  <rbas[a]|psi[i]> = (rbas^+*wfs)^T = wfs^T*rbas.conj()
   // such that W*[i,a]W[j,a] = delta[i,j]
   rwfuns(0,cpos) = linalg::xgemm("T","N",wfs,rbas.conj());
   if(debug){
      rwfuns.print("rwfuns",2);
      std::cout << "\ncheck state overlaps" << std::endl;
      // ova
      auto ova = xgemm("N","N",rwfuns(0,cpos).conj(),rwfuns(0,cpos).T());
      ova.print("ova_rwfuns");
      // ova0
      linalg::matrix<Tm> ova0(nroot,nroot);
      for(int i=0; i<nroot; i++){
         for(int j=0; j<nroot; j++){
	    ova0(i,j) = linalg::xdot(vs[i].size(),vs[i].data(),vs[j].data());
	 }
      }
      ova0.print("ova0_vs");
      auto diff = linalg::normF(ova-ova0);
      std::cout << "diff=" << diff << std::endl;
      if(diff > 1.e-10){
         std::cout << "error: too large diff=" << diff << std::endl;
	 exit(1); 
      }
   }
   icomb.rwfuns = rwfuns; 
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
   // compute wave functions at the start for right canonical form 
   get_rwfuns(icomb, space, vs, thresh_proj);   
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::rcanon_init : " << std::setprecision(2) 
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

} // ctns

#endif
