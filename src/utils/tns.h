#ifndef TNS_H
#define TNS_H

#include "../core/onspace.h"
#include <vector>
#include <tuple>

namespace tns{

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space, const int n);
      std::pair<int,double> projection(const std::vector<std::vector<double>>& vs,
		      		       const double thresh=1.e-6);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dim, dimA, dimB;
};

// brute-force
void ordering_bf(const fock::onspace& space,
	         const std::vector<std::vector<double>>& vs,
	         std::vector<int>& order);
			
// compute ordering
void ordering_ga(const fock::onspace& space,
	         const std::vector<std::vector<double>>& vs,
	         std::vector<int>& order);
			
} // tns

#endif
