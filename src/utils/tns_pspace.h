#ifndef TNS_PSPACE_H
#define TNS_PSPACE_H

#include "../core/onspace.h"
#include "../core/matrix.h"
#include "tns_qsym.h"
#include <string>
#include <vector>
#include <tuple>

namespace tns{

// renormalized states from determinants
struct renorm_sector{
   public:
      void print(const std::string msg, const int level=0) const;
   public:
      qsym sym;
      fock::onspace space;
      linalg::matrix coeff;
};
// this is just like atomic basis
using renorm_basis = std::vector<renorm_sector>;
// rbasis for type-0 physical site
renorm_basis get_rbasis_phys();

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space, const int n);
      // compute left basis
      std::pair<int,double> projection(const std::vector<std::vector<double>>& vs,
		      		       const double thresh_proj=1.e-15);
      // compute {|r>} basis for a given bipartition 
      renorm_basis right_projection(const std::vector<std::vector<double>>& vs,
		      	 	    const double thresh_proj=1.e-15,
				    const bool debug=false);
   public:
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance [NOT sorted!]
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dim, dimA, dimB;
};

} // tns

#endif
