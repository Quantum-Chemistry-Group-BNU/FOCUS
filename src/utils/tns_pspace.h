#ifndef TNS_PSPACE_H
#define TNS_PSPACE_H

#include "../core/onspace.h"
#include "../core/tools.h"
#include <vector>
#include <tuple>

namespace tns{

// renormalized states from determinants
struct renorm_basis{
   public:
      using qsym = std::pair<int,int>;

};

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space, const int n);
      std::pair<int,double> projection(const std::vector<std::vector<double>>& vs,
		      		       const double thresh=1.e-6);
      void right_projection(const std::vector<std::vector<double>>& vs,
		      	    const double thresh=1.e-6);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dim, dimA, dimB;
};

} // tns

#endif
