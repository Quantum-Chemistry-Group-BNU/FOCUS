#ifndef TNS_PSPACE_H
#define TNS_PSPACE_H

#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../core/tools.h"
#include <string>
#include <vector>
#include <tuple>

namespace tns{

// renormalized states from determinants
struct renorm_sector{
   public:
      void print(std::string msg, const int level=0){
	 cout << "renorm_sector: " << msg 
	      << " qsym=(" << sym.first << "," << sym.second << ")"
	      << " shape=" << coeff.rows() << "," << coeff.cols() << endl; 
	 if(level >= 1){
	    for(int i=0; i<space.size(); i++){
	       cout << " idx=" << i << " state=" << space[i].to_string2() << endl;
	    }
	 }
	 if(level >= 2) coeff.print("coeff");
      }
   public:
      using qsym = std::pair<int,int>;
      qsym sym;
      fock::onspace space;
      linalg::matrix coeff;
};
// this is just like atomic basis
using renorm_basis = std::vector<renorm_sector>;

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space, const int n);
      std::pair<int,double> projection(const std::vector<std::vector<double>>& vs,
		      		       const double thresh_proj=1.e-15);
      renorm_basis right_projection(const std::vector<std::vector<double>>& vs,
		      	 	    const double thresh_proj=1.e-15);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance [NOT sorted!]
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dim, dimA, dimB;
};

} // tns

#endif
