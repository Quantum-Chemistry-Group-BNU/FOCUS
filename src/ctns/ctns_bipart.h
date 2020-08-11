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
      transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
	        [](const Tm& x, const double& y){ return x*y; });
   }
}

// bipartition representation of the determinant space
struct bipart_qspace{
   public:
      // get syms,dims,index
      void update(){
         for(const auto& pl : basis){
            const auto& ql = pl.first;
            const auto& ql_space = pl.second;
            syms.push_back(ql);
	    dims[ql] = ql_space.size();
            // setup map from state to index
	    int idx = 0;
            for(const auto& state : ql_space){
               index[ql][state] = idx;
               idx += 1;
            }
         }
      }
      int get_dim() const{
	 int dimt = 0;
	 for(const auto& p : dims){
	    dimt += p.second;	
	 }
	 return dimt;
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
      //bipart_qspace lspace, rspace; // not used right now
      std::map<std::pair<qsym,qsym>, std::vector<linalg::matrix<Tm>>> qblocks;
};

// compute {|r>} basis for a given bipartition specified by the position n 
renorm_basis<double> right_projection(const fock::onspace& space,
		                      const std::vector<std::vector<double>>& vs,
		                      const int bpos, 
		                      const double thresh_proj,
				      const bool debug=false);
 
renorm_basis<std::complex<double>> right_projection(const fock::onspace& space,
		                      const std::vector<std::vector<std::complex<double>>>& vs,
		                      const int bpos, 
		                      const double thresh_proj,
				      const bool debug=false);

/*
renorm_basis<double> right_projection0(const fock::onspace& space,
		                      const std::vector<std::vector<double>>& vs,
		                      const double thresh_proj,
				      const bool debug=false);
*/
 
renorm_basis<std::complex<double>> right_projection0(const fock::onspace& space,
		                      const std::vector<std::vector<std::complex<double>>>& vs,
		                      const double thresh_proj,
				      const bool debug=false);

} // ctns

#endif
