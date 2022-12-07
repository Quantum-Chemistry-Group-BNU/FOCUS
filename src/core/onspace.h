#ifndef ONSPACE_H
#define ONSPACE_H

#include <vector>
#include "onstate.h"
#include "matrix.h"

namespace fock{

using onspace = std::vector<onstate>;

struct onspace_compact{
   public:
      onspace_compact(const onspace& space){
         dim = space.size();
	 if(dim > 0){
	    size = space[0]._size;
	    len = space[0]._len;
	    reprs.resize(dim*len);
            for(size_t i=0; i<dim; i++){
               std::copy_n(space[i]._repr, len, &reprs[i*len]);
	    }
	 }
      }
      // save
      void save(std::ofstream& ofs) const{
	 ofs.write((char*)(&dim), sizeof(dim));
	 ofs.write((char*)(&size), sizeof(size));
         ofs.write((char*)(&len), sizeof(len));
         ofs.write(reinterpret_cast<const char*>(reprs.data()),
	           sizeof(reprs[0])*dim*len); 
      }
   public:
      size_t dim = 0;
      int size = 0;
      int len = 0;
      std::vector<unsigned long> reprs;
};
      
// print
void check_space(onspace& space);

// --- FCI space ---
// Fock space
onspace get_fci_space(const int k);
// spinless case
onspace get_fci_space(const int k, const int n);
// k - number of spatial orbitals 
onspace get_fci_space(const int ks, const int na, const int nb);

// --- Permute underlying basis (in CTNS) ---
// transform space and coefficient upon permutation
template <typename Tm>
void transform_coeff(const onspace& space,
 	             const std::vector<std::vector<Tm>>& vs,
		     const std::vector<int>& order,
		     onspace& space2,
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
      for(int j=0; j<dim; j++){
 	 vs2[i][j] = vs[i][j]*sgns[j];
      }
   }
}

// --- Direct product space ---
// coupling matrix for basis: B0[j,i] = <D[0],D[j]|D[i]>
template <typename Tm>
linalg::matrix<Tm> get_Bcouple(const fock::onstate& state0,
		               const onspace& space1,
			       const onspace& space){
   int m = space1.size(), n = space.size();
   linalg::matrix<Tm> B(m,n);
   for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
	 B(j,i) = static_cast<double>(state0.join(space1[j]) == space[i]);
      } // j
   } // i
   return B;
}

} // fock

#endif
