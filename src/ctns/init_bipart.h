#ifndef INIT_BIPART_H
#define INIT_BIPART_H

#include <vector>
#include <algorithm> // transform
#include "../core/onspace.h"
#include "qtensor/qtensor.h"
#include "init_rbasis.h"

namespace ctns{

//
// bipartition representation of the determinant space: lspace/rspace
//
struct bipart_space{
   public:
      void init(const int isym, // choose qsym according to isym 
		const int bpos, 
		const fock::onspace& space, 
		const int dir){
         // becore bpos
         if(dir == 0){
            for(int i=0; i<space.size(); i++){
               auto lstate = (bpos==0)? fock::onstate() : space[i].get_before(bpos);
	       // isym=1 will keep both |D> and |Df>: key must be standard!
               if(isym == 1) lstate = lstate.make_standard();
               auto itl = uset.find(lstate);
               if(itl == uset.end()){ // not found - new basis
                  uset.insert(lstate);
                  auto ql = get_qsym_onstate(isym,lstate);
                  basis[ql].push_back(lstate);
	          // We also put flipped determinant into basis [only for N!]
                  if(isym == 1 && lstate.norb_single() != 0){ 
		     basis[ql.flip()].push_back(lstate.flip());
		  }
               }
	    } // i
	 // after bpos
         }else{
            for(int i=0; i<space.size(); i++){
               auto rstate = (bpos==0)? space[i] : space[i].get_after(bpos);
	       // isym=1 will keep both |D> and |Df>: key must be standard!
               if(isym == 1) rstate = rstate.make_standard();
               auto itr = uset.find(rstate);
               if(itr == uset.end()){
                  uset.insert(rstate);
                  auto qr = get_qsym_onstate(isym,rstate);
                  basis[qr].push_back(rstate);
                  if(isym == 1 && rstate.norb_single() != 0){ 
		     basis[qr.flip()].push_back(rstate.flip());
		  }
               }
	    } // i
	 }
      }
      // get syms,dims,index
      void update_maps(){
         for(const auto& p : basis){
            const auto& sym = p.first;
            const auto& space = p.second;
            syms.push_back(sym);
	    dims[sym] = space.size();
            // setup map from state to index
	    int idx = 0;
            for(const auto& state : space){
               index[sym][state] = idx;
               idx++;
            }
         }
      }
      int get_dimAll() const{
	 int dim = 0;
	 for(const auto& p : dims){
	    dim += p.second;	
	 }
	 return dim;
      }
      // print basis
      void print_basis(const std::string name){
	 std::cout << "\nbipart_space: " << name << std::endl; 
         for(const auto& p : basis){
            const auto& sym = p.first;
            const auto& space = p.second;
	    std::cout << "qsym=" << sym << " ndim=" << space.size() << std::endl;
	    int istate = 0;
	    for(const auto& state : space){
	       std::cout << " istate=" << istate << " " << state << std::endl;
	       istate++;
	    }
	 }
      } 
   public:
      std::set<fock::onstate> uset; // only store TR-representative
      std::map<qsym,fock::onspace> basis; // determinant basis
      // updated information
      std::vector<qsym> syms; // symmetry
      std::map<qsym,int> dims; // dimension
      std::map<qsym,int> dim0; // dimension for |D> without singly occupied orbitals
      std::map<qsym,std::map<fock::onstate,int>> index; // index of a state
};

//
// wavefunction in the bipartite representation: (ql,qr)->block (support multistate)
//
template <typename Tm>
struct bipart_ciwfs{
   public:
      bipart_ciwfs(const int isym,
		   const int bpos, 
		   const fock::onspace& space, 
		   const std::vector<std::vector<Tm>>& vs,
		   const bipart_space& lspace, 
		   const bipart_space& rspace){
	 lsyms = lspace.syms;
         rsyms = rspace.syms;	
	 for(const auto& ql : lsyms){
            for(const auto& qr : rsyms){
	       auto key = std::make_pair(ql,qr);	    
	       qblocks[key].resize(0);
	    }
	 } 
	 nroots = vs.size(); 
         for(int i=0; i<space.size(); i++){
            auto lstate = (bpos==0)? fock::onstate() : space[i].get_before(bpos);
            auto rstate = (bpos==0)? space[i] : space[i].get_after(bpos);
            auto ql = get_qsym_onstate(isym, lstate);
            auto qr = get_qsym_onstate(isym, rstate);
            int nl = lspace.dims.at(ql);
            int nr = rspace.dims.at(qr);
            int il = lspace.index.at(ql).at(lstate);
            int ir = rspace.index.at(qr).at(rstate);
            auto key = std::make_pair(ql,qr);
	    if(qblocks[key].size() == 0){
               qblocks[key].resize(nroots); // init
               for(int iroot=0; iroot<nroots; iroot++){
		  linalg::matrix<Tm> mat(nl,nr);		
                  mat(il,ir) = vs[iroot][i]; 
                  qblocks[key][iroot] = mat;
               }
            }else{
               for(int iroot=0; iroot<nroots; iroot++){
                  qblocks[key][iroot](il,ir) = vs[iroot][i]; 
               }
            }
         } // idet
      }
      // rhor[r,r'] = psi[l,r]^T*psi[l,r']^*    
      linalg::matrix<Tm> get_rhor(const bipart_space& lspace, 
		      		  const bipart_space& rspace, 
				  const qsym qr){ 
         int dim = rspace.dims.at(qr);
	 linalg::matrix<Tm> rhor(dim,dim);
         for(const auto& ql : lspace.syms){
            auto key = std::make_pair(ql,qr);
            const auto& blk = qblocks[key];
            if(blk.size() == 0) continue;
            for(int iroot=0; iroot<nroots; iroot++){
               rhor += linalg::xgemm("T","N",blk[iroot],blk[iroot].conj());
            }
         }
         rhor *= 1.0/nroots;   
	 return rhor; 
      }
      // debug
      void print() const{
	 std::cout << "bipart_ciwfs:" << std::endl;
         for(int i=0; i<lsyms.size(); i++){
            const auto& ql = lsyms[i];
            for(int j=0; j<rsyms.size(); j++){
	       const auto& qr = rsyms[j];
	       auto key = std::make_pair(ql,qr);
	       const auto& blk = qblocks.at(key);
	       std::cout << " i,j=" << i << "," << j 
		         << " ql=" << ql << " qr=" << qr
			 << " blk=" << blk.size() 
			 << std::endl; 
	    } // j
	 } // i
      }
   public:
      int nroots;
      std::vector<qsym> lsyms, rsyms;
      std::map<std::pair<qsym,qsym>, std::vector<linalg::matrix<Tm>>> qblocks;
};

//
// update rbasis for qr
//
template <typename Tm>
void update_rbasis(renorm_basis<Tm>& rbasis, 
		   const qsym& qr, // qr 
		   const bipart_space& rspace, 
		   const std::vector<double>& eigs, 
		   linalg::matrix<Tm>& U, 
		   int& dimBc, double& popBc, double& SvN,
		   const double thresh_proj,
		   const bool debug){
   const bool debug_basis = false;
   int dim = rspace.dims.at(qr);
   // selection of important states if sig2 > thresh_proj
   double popBi = 0.0;   
   std::vector<int> kept;
   for(int i=0; i<eigs.size(); i++){ // enable reduced SVD
      if(eigs[i] > thresh_proj){
         kept.push_back(i);
         popBi += eigs[i];
         SvN += -eigs[i]*log2(eigs[i]); // compute entanglement entropy
      }
   }
   int dimBi = kept.size();
   dimBc += dimBi;
   popBc += popBi;
   if(debug){
     std::cout << " qr=" << qr << " dimB,dimBi=" << dim << "," << dimBi 
               << " popBi=" << popBi << " popBc=" << popBc << " 1-popBc=" << 1.0-popBc 
	       << std::endl;
   }
   // save renormalized sector (sym,space,coeff)
   if(dimBi > 0){
      renorm_sector<Tm> rsec;
      rsec.sym = qr;
      rsec.space = rspace.basis.at(qr);
      rsec.coeff.resize(dim, dimBi);
      for(int i=0; i<dimBi; i++){
	 int idx = kept[i];
         if(debug_basis) std::cout << " i=" << i << " idx=" << idx 
			           << " eig=" << std::scientific << eigs[idx] 
				   << std::endl; 
	 std::copy(U.col(idx), U.col(idx)+dim, rsec.coeff.col(i)); // copy U[i] to coeff
      }
      rbasis.push_back(rsec);
      // check orthogonality
      if(debug_basis){
         auto ova = linalg::xgemm("N","N",rsec.coeff.H(),rsec.coeff);
         double diff = linalg::normF(ova - linalg::identity_matrix<Tm>(dimBi));
         std::cout << " orthonormality=" << diff << std::endl;
         if(diff > 1.e-10) tools::exit("error: basis is not orthonormal!");
      }
   } // dimBi>0
}

//
// compute {|r>} basis for a given bipartition specified by the position bpos
//
template <typename Km>
void right_projection(renorm_basis<typename Km::dtype>& rbasis,
		      const int bpos,
		      const fock::onspace& space,
		      const std::vector<std::vector<typename Km::dtype>>& vs,
		      const double thresh_proj,
		      const double rdm_vs_svd,
		      const bool debug){
   const bool debug_basis = false;
   using Tm = typename Km::dtype;
   auto t0 = tools::get_time();
   if(debug){
      std::cout << "ctns::right_projection<Km> thresh_proj=" 
                << std::scientific << std::setprecision(2) << thresh_proj << std::endl;
   }
   
   // 1. prepare bipartition form of psi
   bipart_space lspace, rspace;
   lspace.init(Km::isym, bpos, space, 0);
   rspace.init(Km::isym, bpos, space, 1);
   // update space info
   lspace.update_maps();
   rspace.update_maps();
   // construct wfs
   bipart_ciwfs<Tm> wfs(Km::isym, bpos, space, vs, lspace, rspace);

   // 2. decimation for each qr
   int nroots = vs.size();
   int dimBc = 0; double popBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      int dimr = rspace.dims[qr];
      if(debug_basis){
	 std::cout << "\nqr=" << qr << " dimr=" << dimr << std::endl;
         for(const auto& state : rspace.basis[qr]){
	    std::cout << " state=" << state << " index=" << rspace.index[qr][state] << std::endl;
         }
      }
      // 2.1 produce rbasis properly
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(const auto& ql : lspace.syms){
         auto key = std::make_pair(ql,qr);
	 const auto& blk = wfs.qblocks[key];
	 if(blk.size() == 0) continue;
         if(debug_basis) std::cout << "find matched ql =" << ql << std::endl;
	 matched += 1;
         if(matched > 1) tools::exit("multiple matched ql is not supported!");
	 kramers::get_renorm_states_nkr(blk, sigs2, U, rdm_vs_svd, debug_basis);
      } // ql
      // 2.2 select important renormalized states from (sigs2,U) 
      // NOTE: it is possible that matched=0, when we add flipped det in bipart_space with sym=NSz!
      //       then this part needs to be skipped as no sig2 and U are generated.
      if(matched == 1){
         update_rbasis(rbasis, qr, rspace, sigs2, U, dimBc, popBc, SvN, thresh_proj, debug);
      }
   } // qr
   assert(std::abs(popBc-1.0) < 10*thresh_proj);

   if(debug){
      std::cout << "dim(space,lspace,rspace)=" << space.size() << "," 
                << lspace.get_dimAll() << "," << rspace.get_dimAll() 
                << " dimBc=" << dimBc << " popBc=" << popBc << " SvN=" << SvN << std::endl; 
      auto t1 = tools::get_time();
      tools::timing("ctns::right_projection<Km>", t0, t1);
   }
}

//
// time-reversal symmetry adapted right_projection
//
template <> 
inline void right_projection<qkind::cNK>(renorm_basis<std::complex<double>>& rbasis,
		      	                 const int bpos,
		      	                 const fock::onspace& space,
		      	                 const std::vector<std::vector<std::complex<double>>>& vs,
		      	                 const double thresh_proj,
					 const double rdm_vs_svd,
		      	                 const bool debug){
   const bool debug_basis = false;
   using Tm = std::complex<double>;
   auto t0 = tools::get_time();
   if(debug){
      std::cout << "ctns::right_projection<cNK> thresh_proj=" 
                << std::scientific << std::setprecision(2) << thresh_proj << std::endl;
   }
   
   // 1. bipartition form of psi
   bipart_space lspace, rspace;
   lspace.init(qkind::cNK::isym, bpos, space, 0);
   rspace.init(qkind::cNK::isym, bpos, space, 1);
   // reorder rspace.basis to form Kramers-paired structure
   for(const auto& pr : rspace.basis){
      const auto& qr = pr.first;
      const auto& qr_space = pr.second;
      int dim = qr_space.size();
      // odd electron case {|D>,|Df>}
      if(qr.ne()%2 == 1){
         assert(dim%2 == 0);
	 fock::onspace rdets(dim);
         for(int i=0; i<dim/2; i++){
            rdets[i] = qr_space[2*i];
            rdets[i+dim/2] = qr_space[2*i+1];
         }
         rspace.basis[qr] = rdets;
      // even electron case {|D>,|Df>,|D0>} 
      }else{
	 fock::onspace rdets0, rdets1, rdets;
	 for(int i=0; i<dim; i++){
	    auto& state = qr_space[i];
	    if(state.norb_single() == 0){
	       rdets0.push_back(state);
	    }else{
	       if(state.is_standard()) rdets1.push_back(state);
	    }
	 }
	 rspace.dim0[qr] = rdets0.size();
         assert(2*rdets1.size()+rdets0.size() == dim);
	 rdets = rdets1;
	 std::transform(rdets1.begin(),rdets1.end(),rdets1.begin(),
	                [](const fock::onstate& state){ return state.flip(); });		 
	 std::copy(rdets1.begin(),rdets1.end(),back_inserter(rdets));
	 std::copy(rdets0.begin(),rdets0.end(),back_inserter(rdets));
         rspace.basis[qr] = rdets;
      }
   }
   // update space info
   lspace.update_maps();
   rspace.update_maps();
   // construct wfs
   bipart_ciwfs<Tm> wfs(qkind::cNK::isym, bpos, space, vs, lspace, rspace);
   
   // 2. form dm for each qr
   int nroots = vs.size();
   int dimBc = 0; double popBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      int dimr = rspace.dims[qr];
      int dim0 = (qr.ne()%2 == 1)? 0 : rspace.dim0[qr];
      int dim1 = (dimr-dim0)/2;
      assert(dim0 + dim1*2 == dimr);
      if(debug_basis){
	 std::cout << "\nqr=" << qr << " (dim,dim1,dim0)=" << dimr 
		   << "," << dim1 << "," << dim0 << std::endl;
         for(const auto& state : rspace.basis[qr]){
	    std::cout << " state=" << state << " index=" << rspace.index[qr][state] << std::endl;
         }
      }
      // 2.0 phase for determinant with open-shell electrons
      std::vector<double> phases(dim1);
      for(int i=0; i<dim1; i++){
         phases[i] = rspace.basis[qr][i].parity_flip();
      }
      // 2.1 diagonalized properly to yield rbasis (depending on qr)
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(const auto& ql : lspace.syms){
         auto key = std::make_pair(ql,qr);
	 const auto& blk = wfs.qblocks[key];
	 if(blk.size() == 0) continue;
         if(debug_basis) std::cout << "find matched ql =" << ql << std::endl;
	 matched += 1;
         if(matched > 1) tools::exit("multiple matched ql is not supported!");
	 kramers::get_renorm_states_kr(qr, phases, blk, sigs2, U, rdm_vs_svd, debug_basis);
      } // ql
      // 2.2 select important renormalized states from (sigs2,U) 
      if(matched == 1){
         update_rbasis(rbasis, qr, rspace, sigs2, U, dimBc, popBc, SvN, thresh_proj, debug);
      }
   } // qr
   assert(std::abs(popBc-1.0) < 10*thresh_proj);

   if(debug){
      std::cout << "dim(space,lspace,rspace)=" << space.size() << "," 
           << lspace.get_dimAll() << "," << rspace.get_dimAll() 
           << " dimBc=" << dimBc << " popBc=" << popBc << " SvN=" << SvN << std::endl; 
      auto t1 = tools::get_time();
      tools::timing("ctns::right_projection<cNK>", t0, t1);
   }
}

} // ctns

#endif
