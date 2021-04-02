#ifndef CTNS_INIT_H
#define CTNS_INIT_H

#include "../core/tools.h"
#include "../core/onspace.h"
#include "ctns_bipart.h"
#include "ctns_comb.h"
#include "ctns_phys.h"

namespace ctns{

// initialize RCF from SCI wavefunctions
template <typename Km>
void rcanon_init(comb<Km>& icomb,
		 const fock::onspace& space,
		 const std::vector<std::vector<typename Km::dtype>>& vs,
		 const double thresh_proj){
   auto t0 = tools::get_time();
   std::cout << "\nctns::rcanon_init" << std::endl;
   
   // 1. compute renormalized bases {|r>} from SCI wavefunctions
   get_rbases(icomb, space, vs, thresh_proj);

   // 2. build sites from rbases
   get_rsites(icomb); 
 
   // 3. compute wave functions at the start for right canonical form 
   get_rwfuns(icomb, space, vs, thresh_proj);  
  
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::rcanon_init : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

// compute renormalized bases {|r>} from SCI wavefunctions 
template <typename Km>
void get_rbases(comb<Km>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<typename Km::dtype>>& vs,
		const double thresh_proj){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rbases thresh_proj=" << std::scientific << thresh_proj << std::endl;

   // loop over nodes/bond (except the last one) - parallelizable
   const auto& topo = icomb.topo;
   for(int idx=0; idx<topo.rcoord.size(); idx++){
      auto p = topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = topo.nodes[i][j];
      if(debug){
	 std::cout << "\nidx=" << idx << " node=" << p << " type=" << node.type; 
	 std::cout << " rsupport=";
         for(int k : node.rsupport) std::cout << k << " ";
	 std::cout << std::endl;
      }
      if(node.type == 0 && p != std::make_pair(0,0)){
         // for boundary site, we choose to use identity
	 icomb.rbases[p] = get_rbasis_phys<typename Km::dtype>(Km::isym);
      }else{
	 // Generate {|r>} at the internal nodes
         // 1. generate 1D ordering
         const auto& rsupp = node.rsupport; 
         auto order = node.lsupport;
         int bpos = order.size(); // must be put here to account bipartition position
         copy(rsupp.begin(), rsupp.end(), back_inserter(order));
         if(debug){
            std::cout << " bpos=" << bpos;
	    std::cout << " order=";
            for(int k : order) std::cout << k << " ";
	    std::cout << std::endl;
         }
         // 2. transform SCI coefficient
	 fock::onspace space2;
	 std::vector<std::vector<typename Km::dtype>> vs2;
         transform_coeff(space, vs, order, space2, vs2); 
         // 3. bipartition of space and compute renormalized states
         right_projection<Km>(icomb.rbases[p], 2*bpos, space2, vs2, thresh_proj, debug);
      }
   } // idx
   
   // print information for all renormalized basis {|r>} at each bond
   std::cout << "\nfinal rbases with thresh_proj=" << thresh_proj << std::endl;
   int Dmax = 0;
   for(int idx=0; idx<topo.rcoord.size(); idx++){
      auto p = topo.rcoord[idx];
      int i = p.first, j = p.second;
      // shape can be different from dim(rspace) if associated weight is zero!
      auto shape = get_shape(icomb.rbases[p]);
      std::cout << " idx=" << idx << " node=" << p
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
template <typename Km>
void get_rsites(comb<Km>& icomb){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rsites" << std::endl;

   // loop over sites
   const auto& topo = icomb.topo;
   for(int idx=0; idx<topo.rcoord.size(); idx++){
      auto p = topo.rcoord[idx];
      int i = p.first, j = p.second;
      auto& node = topo.nodes[i][j];
      if(debug) std::cout << "\nidx=" << idx << " node=" << p << " ";     
      if(node.type == 0 && p != std::make_pair(0,0)){
	 
	 if(debug) std::cout << "type 0: end or leaves" << std::endl;
         get_right_bsite(Km::isym, icomb.rsites[p]);

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
         const auto& rbasis_l = icomb.rbases.at(p); 
	 const auto& rbasis_c = (node.type==3)? icomb.rbases.at(node.center) : get_rbasis_phys<typename Km::dtype>(Km::isym); 
	 const auto& rbasis_r = icomb.rbases.at(node.right);
	 auto qmid = get_qbond(rbasis_c);
	 auto qrow = get_qbond(rbasis_l); 
	 auto qcol = get_qbond(rbasis_r);
	 qtensor3<typename Km::dtype> qt3(qsym(), qmid, qrow, qcol);
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
		     auto tmp1 = fock::get_Bcouple<typename Km::dtype>(state_c,rbasis_r[kr].space,rbasis_l[kl].space);
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
template <typename Km>
void get_rwfuns(comb<Km>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<typename Km::dtype>>& vs,
		const double thresh_proj){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::get_rwfuns thresh_proj=" << std::scientific << thresh_proj << std::endl;

   // determine symmetry of rwfuns
   const auto& det = space[0];
   auto sym_states = get_qsym_onstate(Km::isym, space[0]);
   // determine qrow: we assume all the dets are of the same symmetry!
   for(int i=0; i<space.size(); i++){
      auto sym = get_qsym_onstate(Km::isym, space[i]);
      if(sym != sym_states){
         std::cout << "error: symmetry is different in space!" << std::endl;
	 std::cout << "sym_states=" << sym_states 
		   << " det=" << space[i] << " sym=" << sym
		   << std::endl;
	 exit(1);
      }
   }
   int nroot = vs.size(); 
   qbond qrow({{sym_states, nroot}});
   // qcol
   const auto& rbasis = icomb.rbases.at(std::make_pair(0,0));
   auto qcol = get_qbond(rbasis);
   // rwfuns[l,r] for RCF
   qtensor2<typename Km::dtype> rwfuns(qsym(), qrow, qcol, {0, 1});
   //
   // construct the boundary matrix: |psi[i]> = \sum_a |rbas[a]>(<rbas[a]|psi[i]>)
   // In RCF the site is defined as 
   //    W[i,a] =  <rbas[a]|psi[i]> = (rbas^+*wfs)^T = wfs^T*rbas.conj()
   // such that W*[i,a]W[j,a] = delta[i,j]
   //
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
   std::vector<std::vector<typename Km::dtype>> vs2;
   const auto& order = icomb.topo.nodes[0][0].rsupport;
   // transform SCI coefficient to order
   transform_coeff(space, vs, order, space2, vs2); 
   const auto& rbas = rbasis[cpos].coeff;
   linalg::matrix<typename Km::dtype> wfs(rbas.rows(), nroot);
   for(int i=0; i<space2.size(); i++){
      int ir = index.at(space2[i]);
      for(int iroot=0; iroot<nroot; iroot++){
         wfs(ir,iroot) = vs2[iroot][i];
      } // iroot
   } // i
   rwfuns(0,cpos) = linalg::xgemm("T","N",wfs,rbas.conj());
   icomb.rwfuns = std::move(rwfuns);

   if(debug){
      icomb.rwfuns.print("rwfuns",2);
      std::cout << "\ncheck state overlaps" << std::endl;
      // ova = <CTNS[i]|CTNS[j]>
      auto ova = xgemm("N","N",icomb.rwfuns(0,cpos).conj(),icomb.rwfuns(0,cpos).T());
      ova.print("ova_rwfuns");
      // ova0 = <CI[i]|CI[j]>
      linalg::matrix<typename Km::dtype> ova0(nroot,nroot);
      for(int i=0; i<nroot; i++){
         for(int j=0; j<nroot; j++){
	    ova0(i,j) = linalg::xdot(vs[i].size(),vs[i].data(),vs[j].data());
	 }
      }
      ova0.print("ova0_vs");
      auto diff = linalg::normF(ova-ova0);
      std::cout << "diff of ova matrix = " << diff << std::endl;
      const double thresh = 1.e-8;
      if(diff > thresh){
         std::cout << "error: too large diff=" << diff << " thresh=" << thresh << std::endl;
	 exit(1); 
      }
   }
}

} // ctns

#endif
