#ifndef CTNS_INIT_H
#define CTNS_INIT_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../core/tools.h"
#include "../core/onspace.h"
#include "ctns_comb.h"
#include "init_phys.h"
#include "init_bipart.h"

const bool debug_init = true;
extern const bool debug_init;

namespace ctns{

// initialize RCF from SCI wavefunctions
template <typename Km>
void rcanon_init(comb<Km>& icomb,
		 const fock::onspace& space,
		 const std::vector<std::vector<typename Km::dtype>>& vs,
		 const double thresh_proj,
		 const double rdm_vs_svd){
   std::cout << "\nctns::rcanon_init Km=" << qkind::get_name<Km>() << std::endl;
   auto t0 = tools::get_time();

   // 1. compute renormalized bases {|r>} from SCI wavefunctions
   init_rbases(icomb, space, vs, thresh_proj, rdm_vs_svd);

   // 2. build sites from rbases
   init_rsites(icomb); 

   // 3. compute wave functions at the start for right canonical form 
   init_rwfuns(icomb, space, vs);  

   auto t1 = tools::get_time();
   tools::timing("ctns::rcanon_init", t0, t1);
}


// compute renormalized bases {|r>} from SCI wavefunctions 
template <typename Km>
void init_rbases(comb<Km>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<typename Km::dtype>>& vs,
		const double thresh_proj,
		const double rdm_vs_svd){
   using Tm = typename Km::dtype;
   std::cout << "\nctns::init_rbases" << std::scientific << std::setprecision(2) 
	     << " thresh_proj=" << thresh_proj << " rdm_vs_svd=" << rdm_vs_svd
	     << std::endl;
   auto t0 = tools::get_time();
  
   // loop over bond - parallelizable
   const auto& topo = icomb.topo;
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int idx=0; idx<topo.rcoord.size(); idx++){
      auto p = topo.rcoord[idx];
      int i = p.first, j = p.second;
      const auto& node = topo.nodes[i][j];
      if(debug_init){
	 std::cout << "\nidx=" << idx << " node=" << p << " type=" << node.type; 
	 std::cout << " rsupport=";
         for(int k : node.rsupport) std::cout << k << " ";
	 std::cout << std::endl;
      }
      // for boundary site, we simply choose to use identity
      if(node.type == 0 && p != std::make_pair(0,0)){
         
	 icomb.rbases[p] = get_rbasis_phys<Tm>(Km::isym);

      // Generate {|r>} at the internal nodes
      }else{
	 
	 // 1. generate 1D ordering
         const auto& rsupp = node.rsupport; 
         auto order = node.lsupport;
         int bpos = order.size(); // must be put here to account bipartition position
         copy(rsupp.begin(), rsupp.end(), back_inserter(order));
         
	 // 2. transform SCI coefficient to this ordering
	 fock::onspace space2;
	 std::vector<std::vector<Tm>> vs2;
         fock::transform_coeff(space, vs, order, space2, vs2); 

	 // 3. bipartition of space and compute renormalized states [time-consuming part!]
	 right_projection<Km>(icomb.rbases[p], 2*bpos, space2, vs2, 
			      thresh_proj, rdm_vs_svd, debug_init);

      } // node type
   } // idx

   // print information for all renormalized basis {|r>} at each bond
   if(debug_init){
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
   }

   auto t1 = tools::get_time();
   tools::timing("ctns::init_rbases", t0, t1);
}


// build site tensor from {|r>} bases
template <typename Km>
void init_rsites(comb<Km>& icomb){
   using Tm = typename Km::dtype;
   std::cout << "\nctns::init_rsites Km=" << qkind::get_name<Km>() << std::endl;
   auto t0 = tools::get_time();
   
   // loop over sites - parallelizable
   const auto& topo = icomb.topo;
   for(int idx=0; idx<topo.rcoord.size(); idx++){
      auto p = topo.rcoord[idx];
      int i = p.first, j = p.second;
      const auto& node = topo.nodes[i][j];
      if(debug_init){ 
	 std::cout << " idx=" << idx << " node=" << p << "[" << node.type << "]";   
      } 
      auto ti = tools::get_time();
 
      // type=0: end or leaves
      if(node.type == 0 && p != std::make_pair(0,0)){
	
         get_right_bsite(Km::isym, icomb.rsites[p]);

      // physical/internal on backbone/branch
      }else{

	 //  node.type == 3: internal site on backbone
	 //    |u>(0)      
	 //     |
	 //  ---*---|r>(1)
	 //
	 //  node.type = 1/2: physical site on backbone/branch
	 //     n            |u> 
	 //     |             |
	 //  ---*---|r>   n---*
	 //                   |
         const auto& rbasis_l = icomb.rbases.at(p); 
	 const auto& rbasis_c = (node.type==3)? icomb.rbases.at(node.center) : get_rbasis_phys<Tm>(Km::isym); 
	 const auto& rbasis_r = icomb.rbases.at(node.right);
	 auto qmid = get_qbond(rbasis_c);
	 auto qrow = get_qbond(rbasis_l); 
	 auto qcol = get_qbond(rbasis_r);
	 qtensor3<Tm> qt3(qsym(Km::isym), qmid, qrow, qcol);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic) collapse(3)
#endif
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
		  // 
	          // 20210902: new implementation using hash table for constructing <D[c'],D[r']|D[l']>
	          //
		  const auto& rspace = rbasis_r[kr].space;
		  const auto& lspace = rbasis_l[kl].space;
		  int dimr = rspace.size(), diml = lspace.size(), dimlc = Wl.cols();
   		  std::unordered_map<fock::onstate,size_t> helper;
		  for(int j=0; j<diml; j++){
		     helper[lspace[j]] = j;
		  }
		  for(int dc=0; dc<Wc.cols(); dc++){
		     auto state_c = rbasis_c[kc].space[dc];
		     //
		     // old implementation:
		     //
		     // // tmp1[c'][r'l'] = <D[c'],D[r']|[l']>
		     // auto tmp1 = fock::get_Bcouple<Tm>(state_c,rbasis_r[kr].space,rbasis_l[kl].space);
		     // // tmp2[c'][r'l] = tmp1[c'][r'l']Wl[l'l]
		     // auto tmp2 = linalg::xgemm("N","N",tmp1,Wl);
		     // 
		     // 20210902: new implementation using hash table & sparse matrix multiplication
		     // 	  given D[c'], tmp1(r',l')=<D[c'],D[r']|[l']> is extremely sparse!
		     //
	             linalg::matrix<Tm> tmp12(dimr,dimlc);
	             for(int i=0; i<dimr; i++){
		        auto state = state_c.join(rspace[i]);
		        auto search = helper.find(state);
		        if(search != helper.end()){
	       	           int j = search->second;
			   for(int k=0; k<dimlc; k++){
			      tmp12(i,k) += Wl(j,k);
			   } // k
	                } // j
	             } // i
		     // tmp3[c'](l,r)= Wr*[r'r]tmp2[c'][r'l] = tmp2^T*Wr.conj() 
		     auto tmp3 = linalg::xgemm("T","N",tmp12,Wr);
		     // R[c][lr] = sum_c' Wc*[c'c]tmp3[c'][lr]
		     for(int ic=0; ic<Wc.rows(); ic++){
		        blk[ic] += Wc(ic,dc)*tmp3;
		     } // ic
		  } // ibas
	       } // kc
	    } // kr
	 } // kl
         icomb.rsites[p] = std::move(qt3);

      } // node type

      auto tf = tools::get_time(); 
      if(debug_init){ 
         const double thresh_ortho = 1.e-10;
         auto ova = contract_qt3_qt3_cr(icomb.rsites.at(p),icomb.rsites.at(p));
         double maxdiff = ova.check_identityMatrix(thresh_ortho, false);
         auto dt = tools::get_duration(tf-ti);
         std::cout << " shape(l,c,r)=("
                   << icomb.rsites[p].qrow.get_dimAll() << ","
                   << icomb.rsites[p].qmid.get_dimAll() << ","
                   << icomb.rsites[p].qcol.get_dimAll() << ")"
                   << " maxdiff=" << std::scientific << maxdiff 
	           << " TIMING=" << dt << " S"
		   << std::endl;
         if(maxdiff>thresh_ortho) tools::exit("error: deviate from identity matrix!");
      }
   } // idx

   auto t1 = tools::get_time();
   tools::timing("ctns::init_rsites", t0, t1);
}


// compute wave function at the start for right canonical form
template <typename Km>
void init_rwfuns(comb<Km>& icomb,
		const fock::onspace& space,
		const std::vector<std::vector<typename Km::dtype>>& vs){
   using Tm = typename Km::dtype;
   std::cout << "\nctns::init_rwfuns Km=" << qkind::get_name<Km>() << std::endl;
   auto t0 = tools::get_time();
   
   // determine symmetry of rwfuns
   const auto& det = space[0];
   auto sym_states = get_qsym_onstate(Km::isym, space[0]);
   // check symmetry: we assume all the dets are of the same symmetry!
   for(int i=0; i<space.size(); i++){
      auto sym = get_qsym_onstate(Km::isym, space[i]);
      if(sym != sym_states){
	 std::cout << "sym_states=" << sym_states 
		   << " det=" << space[i] << " sym=" << sym
		   << std::endl;
	 tools::exit("error: symmetry is different in space!");
      }
   }
   int nroots = vs.size(); 
   qbond qrow({{sym_states, nroots}});
   const auto& rbasis = icomb.rbases.at(std::make_pair(0,0));
   auto qcol = get_qbond(rbasis);
   if(qcol.size() != 1) tools::exit("error: multiple symmetries in qcol!"); 
   //
   // construct the boundary matrix: |psi[i]> = \sum_a |rbas[a]>(<rbas[a]|psi[i]>)
   // In RCF the site is defined as 
   //    W[i,a] =  <rbas[a]|psi[i]> = (rbas^+*wfs)^T = wfs^T*rbas.conj()
   // such that W*[i,a]W[j,a] = delta[i,j]
   //
   qtensor2<Tm> rwfuns(qsym(Km::isym), qrow, qcol, {0, 1}); // rwfuns[l,r] for RCF
   // setup wavefunction: map vs2 to correct position
   fock::onspace space2;
   std::vector<std::vector<Tm>> vs2;
   const auto& order = icomb.topo.nodes[0][0].rsupport;
   fock::transform_coeff(space, vs, order, space2, vs2);
   //
   // Needs to locate position of space2[i] in rbasis[0].space
   // NOTE: for isym=1, ndet can be larger than space.size()
   //       due to the possible reorder of basis in init_bipart.h
   //
   std::map<fock::onstate,int> index; // index of a state
   int ndet = rbasis[0].space.size();
   for(int i=0; i<ndet; i++){
      const auto& state = rbasis[0].space[i];
      index[state] = i;
   }
   linalg::matrix<Tm> wfs(ndet, nroots);
   for(int i=0; i<space2.size(); i++){
      int ir = index.at(space2[i]);
      for(int iroot=0; iroot<nroots; iroot++){
         wfs(ir,iroot) = vs2[iroot][i];
      } // iroot
   } // i
   rwfuns(0,0) = linalg::xgemm("T","N",wfs,rbasis[0].coeff.conj());
   icomb.rwfuns = std::move(rwfuns);

   // check overlaps
   if(debug_init){
      icomb.rwfuns.print("rwfuns",2);
      std::cout << "\ncheck state overlaps ..." << std::endl;
      // ova = <CTNS[i]|CTNS[j]>
      auto ova = xgemm("N","N",icomb.rwfuns(0,0).conj(),icomb.rwfuns(0,0).T());
      ova.print("ova_rwfuns");
      // ova0 = <CI[i]|CI[j]>
      linalg::matrix<Tm> ova0(nroots,nroots);
      for(int i=0; i<nroots; i++){
         for(int j=0; j<nroots; j++){
	    ova0(i,j) = linalg::xdot(vs[i].size(),vs[i].data(),vs[j].data());
	 }
      }
      ova0.print("ova0_vs");
      auto diff = linalg::normF(ova-ova0);
      std::cout << "diff of ova matrices = " << diff << std::endl;
      const double thresh = 1.e-8;
      if(diff > thresh){ 
	 std::string msg = "error: too large diff=";
	 tools::exit(msg+std::to_string(diff)+" with thresh="+std::to_string(thresh));
      }
   } // debug_init
   
   auto t1 = tools::get_time();
   tools::timing("ctns::init_rwfuns", t0, t1);
}


} // ctns

#endif
