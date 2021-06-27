#ifndef CTNS_BIPART_H
#define CTNS_BIPART_H

#include <vector>
#include <algorithm> // transform
#include "../core/onspace.h"
#include "qtensor/qtensor.h"
#include "ctns_kind.h"
#include "ctns_rbasis.h"

namespace ctns{

// transform space and coefficient upon permutation
template <typename Tm>
void transform_coeff(const fock::onspace& space,
 	             const std::vector<std::vector<Tm>>& vs,
		     const std::vector<int>& order,
		     fock::onspace& space2,
		     std::vector<std::vector<Tm>>& vs2){
   // image2
   int k = order.size();
   std::vector<int> image2(2*k);
   for(int i=0; i<k; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
   }
   // update basis vector and signs 
   space2.clear();
   std::vector<double> sgns;
   for(const auto& state : space){
      space2.push_back(state.permute(image2));
      auto sgn = state.permute_sgn(image2); 
      // for later multiplication with complex<double>
      sgns.push_back(static_cast<double>(sgn));
   }
   int dim = space.size();
   int nroots = vs.size();
   vs2.resize(nroots);
   for(int i=0; i<nroots; i++){
      vs2[i].resize(dim);
      std::transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
	             [](const Tm& x, const double& y){ return x*y; });
   }
}

// bipartition representation of the determinant space: lspace/rspace
struct bipart_qspace{
   public:
      void init(const int isym, // choose qsym according to isym 
		const int bpos, 
		const fock::onspace& space, 
		const int dir){
         // becore bpos
         if(dir == 0){
            for(int i=0; i<space.size(); i++){
               auto lstate = (bpos==0)? fock::onstate() : space[i].get_before(bpos);
               if(isym == 1) lstate = lstate.make_standard(); // key must be standard!
               auto itl = uset.find(lstate);
               if(itl == uset.end()){ // not found - new basis
                  uset.insert(lstate);
                  auto ql = get_qsym_onstate(isym,lstate);
                  basis[ql].push_back(lstate);
	          // We also put flipped determinant into basis [only for N, but not for NSz!]
                  if(isym == 1 && lstate.norb_single() != 0) basis[ql.flip()].push_back(lstate.flip());
               }
	    } // i
	 // after bpos
         }else{
            for(int i=0; i<space.size(); i++){
               auto rstate = (bpos==0)? space[i] : space[i].get_after(bpos);
               if(isym == 1) rstate = rstate.make_standard();
               auto itr = uset.find(rstate);
               if(itr == uset.end()){
                  uset.insert(rstate);
                  auto qr = get_qsym_onstate(isym,rstate);
                  basis[qr].push_back(rstate);
                  if(isym == 1 && rstate.norb_single() != 0) basis[qr.flip()].push_back(rstate.flip());
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
	 std::cout << "\nbipart_qspace: " << name << std::endl; 
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

// wavefunction in the bipartite representation: (ql,qr)->block (support multistate)
template <typename Tm>
struct bipart_ciwfs{
   public:
      bipart_ciwfs(const int isym,
		   const int bpos, 
		   const fock::onspace& space, 
		   const std::vector<std::vector<Tm>>& vs,
		   const bipart_qspace& lspace, 
		   const bipart_qspace& rspace){
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
      linalg::matrix<Tm> get_rhor(const bipart_qspace& lspace, 
		      		  const bipart_qspace& rspace, 
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

// update rbasis for qr
template <typename Tm>
void update_rbasis(renorm_basis<Tm>& rbasis, 
		   const qsym& qr, // qr 
		   const bipart_qspace& rspace, 
		   const std::vector<double>& eigs, 
		   linalg::matrix<Tm>& U, 
		   int& dimBc, double& sumBc, double& SvN,
		   const double thresh,
		   const bool debug){
   const bool debug_basis = false;
   int dim = rspace.dims.at(qr);
   // selection of important states if sig2 > thresh
   double sumBi = 0.0;   
   std::vector<int> kept;
   for(int i=0; i<eigs.size(); i++){ // enable reduced SVD
      if(eigs[i] > thresh){
         kept.push_back(i);
         sumBi += eigs[i];
         SvN += -eigs[i]*log2(eigs[i]); // compute entanglement entropy
      }
   }
   int dimBi = kept.size();
   dimBc += dimBi;
   sumBc += sumBi;
   if(debug){
     std::cout << " qr=" << qr << " dimB,dimBi=" << dim << "," << dimBi 
               << " sumBi=" << sumBi << " sumBc=" << sumBc << std::endl;
   }
   // save renormalized sector (sym,space,coeff)
   if(dimBi > 0){
      renorm_sector<Tm> rsec;
      rsec.sym = qr;
      rsec.space = rspace.basis.at(qr);
      rsec.coeff.resize(dim, dimBi);
      int i = 0;
      for(const int idx : kept){
         if(debug_basis) std::cout << " i=" << i << " eig=" << std::scientific 
     	    		           << eigs[idx] << std::endl;
         std::copy(U.col(idx), U.col(idx)+dim, rsec.coeff.col(i)); // copy U[i] to coeff
         i += 1;
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

// compute {|r>} basis for a given bipartition specified by the position n 
template <typename Km>
void right_projection(renorm_basis<typename Km::dtype>& rbasis,
		      const int bpos,
		      const fock::onspace& space,
		      const std::vector<std::vector<typename Km::dtype>>& vs,
		      const double thresh,
		      const bool debug){
   using Tm = typename Km::dtype;
   auto t0 = tools::get_time();
   const bool debug_basis = false;
   if(debug){
      std::cout << "ctns::right_projection<Km> thresh=" 
                << std::scientific << std::setprecision(2) << thresh << std::endl;
   }
   //
   // 1. prepare bipartition form of psi
   //
   bipart_qspace lspace, rspace;
   lspace.init(Km::isym, bpos, space, 0);
   rspace.init(Km::isym, bpos, space, 1);
   // update space info
   lspace.update_maps();
   rspace.update_maps();
   // construct wfs
   bipart_ciwfs<Tm> wfs(Km::isym, bpos, space, vs, lspace, rspace);
   //
   // 2. form dm for each qr
   //
   int nroots = vs.size();
   int dimBc = 0; 
   double sumBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      int dimr = rspace.dims[qr];
      if(debug_basis){
	 std::cout << "qr=" << qr << " dimr=" << dimr << std::endl;
         for(const auto& state : rspace.basis[qr]){
	    std::cout << " state=" << state << " index=" << rspace.index[qr][state] << std::endl;
         }
      }
      //
      // 3. produce rbasis properly
      //
      if(debug_basis) std::cout << "-qr=" << qr << " find matched ql ..." << std::endl;
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(const auto& ql : lspace.syms){
         auto key = std::make_pair(ql,qr);
	 const auto& blk = wfs.qblocks[key];
	 if(blk.size() == 0) continue;
	 matched += 1;
         if(matched > 1) tools::exit("multiple matched ql is not supported!");
	 int diml = lspace.dims[ql];
	 if(dimr <= static_cast<int>(1.5*diml)){ // 1.5 is an empirical factor based on performance
            if(debug_basis) std::cout << " RDM-based decimation: ql=" << ql << " dim(l,r)=" << diml << "," << dimr << std::endl;
            linalg::matrix<Tm> rhor(dimr,dimr);
	    for(int iroot=0; iroot<nroots; iroot++){
               rhor += linalg::xgemm("T","N",blk[iroot],blk[iroot].conj());
            } // iroot
            rhor *= 1.0/nroots;   
	    sigs2.resize(dimr);
            linalg::eig_solver(rhor, sigs2, U, 1);
	 }else{
            if(debug_basis) std::cout << " SVD-based decimation: ql=" << ql << " dim(l,r)=" << diml << "," << dimr << std::endl;
	    linalg::matrix<Tm> vrl(dimr,diml*nroots);
	    for(int iroot=0; iroot<nroots; iroot++){
	       auto blkt = blk[iroot].T();
	       std::copy(blkt.data(), blkt.data()+dimr*diml, vrl.col(iroot*diml));
	    } // iroot
	    vrl *= 1.0/std::sqrt(nroots);
	    linalg::matrix<Tm> vt; // size of sig2,U,vt will be determined inside svd_solver!
	    linalg::svd_solver(vrl, sigs2, U, vt, 1);
	    std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
			   [](const double& x){ return x*x; });
	 }
      }
      //
      // 4. select important renormalized states from (sigs2,U) 
      //    NOTE: it is possible that matched=0, when we add flipped det in bipart_qspace with sym=NSz!
      //          then, this part needs to be skipped as no sig2 and U are generated in part 3.
      //
      if(matched == 1){
         update_rbasis(rbasis, qr, rspace, sigs2, U, dimBc, sumBc, SvN, thresh, debug);
      }
   } // qr
   if(debug){
      std::cout << "dim(space,lspace,rspace)=" << space.size() << "," 
                << lspace.get_dimAll() << "," << rspace.get_dimAll() 
                << " dimBc=" << dimBc << " sumBc=" << sumBc << " SvN=" << SvN << std::endl; 
      auto t1 = tools::get_time();
      tools::timing("ctns::right_projection<Km>", t0, t1);
   }
}

// time-reversal symmetry adapted right_projection
template <> 
inline void right_projection<kind::cNK>(renorm_basis<std::complex<double>>& rbasis,
		      	         const int bpos,
		      	         const fock::onspace& space,
		      	         const std::vector<std::vector<std::complex<double>>>& vs,
		      	         const double thresh,
		      	         const bool debug){
   auto t0 = tools::get_time();
   const bool debug_basis = false;
   if(debug){
      std::cout << "ctns::right_projection<cNK> thresh=" 
                << std::scientific << std::setprecision(4) << thresh << std::endl;
   }
   //
   // 1. bipartition form of psi
   //
   bipart_qspace lspace, rspace;
   lspace.init(kind::cNK::isym, bpos, space, 0);
   rspace.init(kind::cNK::isym, bpos, space, 1);
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
	       if(state.is_standard()){
	          rdets1.push_back(state);
	       }
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
   bipart_ciwfs<std::complex<double>> wfs(kind::cNK::isym, bpos, space, vs, lspace, rspace);
   //
   // 2. form dm for each qr
   //
   int dimBc = 0; double sumBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      int dim = rspace.dims[qr];
      int dim0 = (qr.ne()%2 == 1)? 0 : rspace.dim0[qr];
      int dim1 = (dim-dim0)/2;
      assert(dim0 + dim1*2 == dim);
      // phase for determinant with open-shell electrons
      std::vector<double> phases(dim1);
      for(int i=0; i<dim1; i++){
         phases[i] = rspace.basis[qr][i].parity_flip();
      }
      if(debug_basis){
	 std::cout << "qr=" << qr << " (dim,dim1,dim0)=" 
	           << dim << "," << dim1 << "," << dim0 << std::endl;
         for(const auto& state : rspace.basis[qr]){
	    std::cout << " state=" << state << " index=" << rspace.index[qr][state] << std::endl;
         }
      }
      //
      // 3. diagonalized properly to yield rbasis (depending on qr)
      //
      auto rhor = wfs.get_rhor(lspace, rspace, qr);
      std::vector<double> eigs(dim); 
      linalg::matrix<std::complex<double>> U(dim,dim);
      eig_solver_kr<std::complex<double>>(qr, rhor, eigs, U, phases);
      //
      // 4. select important renormalized states from (eigs,U) 
      //
      update_rbasis(rbasis, qr, rspace, eigs, U, dimBc, sumBc, SvN, thresh, debug);
   } // qr
   if(debug){
      std::cout << "dim(space,lspace,rspace)=" << space.size() << "," 
           << lspace.get_dimAll() << "," << rspace.get_dimAll() 
           << " dimBc=" << dimBc << " sumBc=" << sumBc << " SvN=" << SvN << std::endl; 
      auto t1 = tools::get_time();
      tools::timing("ctns::right_projection<cNK>", t0, t1);
   }
}

} // ctns

#endif
