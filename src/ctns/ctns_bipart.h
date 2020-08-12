#ifndef CTNS_BIPART_H
#define CTNS_BIPART_H

#include <vector>
#include <algorithm> // transform
#include "../core/onspace.h"
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
      double sgn = state.permute_sgn(image2); // for multiplication with complex<double>
      sgns.push_back(sgn);
   }
   int dim = space.size();
   int nroot = vs.size();
   vs2.resize(nroot);
   for(int i=0; i<nroot; i++){
      vs2[i].resize(dim);
      std::transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
	             [](const Tm& x, const double& y){ return x*y; });
   }
}

// bipartition representation of the determinant space
struct bipart_qspace{
   public:
      template <typename Tm>
      void init(const int bpos, 
		const fock::onspace& space, 
		const int dir){
         if(dir == 0){
            for(int i=0; i<space.size(); i++){
               auto lstate = (bpos==0)? fock::onstate() : space[i].get_before(bpos);
               lstate = lstate.make_standard(); // key must be standard!
               auto itl = uset.find(lstate);
               if(itl == uset.end()){ // not found - new basis
                  uset.insert(lstate);
                  auto ql = get_qsym<Tm>(lstate);
                  basis[ql].push_back(lstate);
                  if(lstate.norb_single() != 0) basis[ql.flip()].push_back(lstate.flip());
               }
	    } // i
         }else{
            for(int i=0; i<space.size(); i++){
               auto rstate = (bpos==0)? space[i] : space[i].get_after(bpos);
               rstate = rstate.make_standard();
               auto itr = uset.find(rstate);
               if(itr == uset.end()){
                  uset.insert(rstate);
                  auto qr = get_qsym<Tm>(rstate);
                  basis[qr].push_back(rstate);
                  if(rstate.norb_single() != 0) basis[qr.flip()].push_back(rstate.flip());
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
      std::set<fock::onstate> uset; // only TR-representative
      std::map<qsym,fock::onspace> basis; // determinant basis
      // updated information
      std::vector<qsym> syms; // symmetry
      std::map<qsym,int> dims; // dimension
      std::map<qsym,int> dim0; // dimension for |D> without singly occupied orbitals
      std::map<qsym,std::map<fock::onstate,int>> index; // index of a state
};

template <typename Tm>
struct bipart_ciwfs{
   public:
      bipart_ciwfs(const int bpos, 
		   const fock::onspace& space, 
		   const std::vector<std::vector<Tm>>& vs,
		   const bipart_qspace& lspace, 
		   const bipart_qspace& rspace){
         nroot = vs.size(); 
         for(int i=0; i<space.size(); i++){
            auto lstate = (bpos==0)? fock::onstate() : space[i].get_before(bpos);
            auto rstate = (bpos==0)? space[i] : space[i].get_after(bpos);
            auto ql = get_qsym<Tm>(lstate);
            auto qr = get_qsym<Tm>(rstate);
            int nl = lspace.dims.at(ql);
            int nr = rspace.dims.at(qr);
            int il = lspace.index.at(ql).at(lstate);
            int ir = rspace.index.at(qr).at(rstate);
            auto key = std::make_pair(ql,qr);
            if(qblocks[key].size() == 0){
               qblocks[key].resize(nroot); // init
               for(int iroot=0; iroot<nroot; iroot++){
		  linalg::matrix<Tm> mat(nl,nr);		
                  mat(il,ir) = vs[iroot][i]; 
                  qblocks[key][iroot] = mat;
               }
            }else{
               for(int iroot=0; iroot<nroot; iroot++){
                  qblocks[key][iroot](il,ir) = vs[iroot][i]; 
               }
            }
         }
      }
      // rhor[r,r'] = psi[l,r]^T*psi[l,r']^*    
      linalg::matrix<Tm> get_rhor(const bipart_qspace& lspace, 
		      		  const bipart_qspace& rspace, 
				  const qsym qr){ 
         int dim = rspace.dims.at(qr);
	 linalg::matrix<Tm> rhor(dim,dim);
         for(const auto& ql : lspace.syms){
            auto key = std::make_pair(ql,qr);
            auto& blk = qblocks[key];
            if(blk.size() == 0) continue;
            for(int iroot=0; iroot<nroot; iroot++){
               rhor += xgemm("T","N",blk[iroot],blk[iroot].conj());
            }
         }
         rhor *= 1.0/nroot;   
	 return rhor; 
      }
   public:
      int nroot;
      std::map<std::pair<qsym,qsym>, std::vector<linalg::matrix<Tm>>> qblocks;
};

// compute {|r>} basis for a given bipartition specified by the position n 
renorm_basis<double> right_projection(const int bpos,
				      const fock::onspace& space,
		                      const std::vector<std::vector<double>>& vs,
		                      const double thresh_proj,
				      const bool debug=false);
 
renorm_basis<std::complex<double>> right_projection(const int bpos,
				      const fock::onspace& space,
		                      const std::vector<std::vector<std::complex<double>>>& vs,
		                      const double thresh_proj,
				      const bool debug=false);

} // ctns

#endif
