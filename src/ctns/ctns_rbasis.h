#ifndef CTNS_RBASIS_H
#define CTNS_RBASIS_H

#include <string>
#include <vector>
#include <tuple>
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "ctns_qsym.h"

namespace ctns{

// renorm_sector: renormalized states from determinants
template <typename Tm>
struct renorm_sector{
   public:
      void print(const std::string msg, const int level=0) const{
	 std::cout << "renorm_sector: " << msg << " qsym=" << sym 
                   << " shape=" << coeff.rows() << "," << coeff.cols() << std::endl; 
         if(level >= 1){
            for(int i=0; i<space.size(); i++){
	       std::cout << " idx=" << i << " state=" << space[i] << std::endl;
            }
            if(level >= 2) coeff.print("coeff");
         }
      }
   public:
      qsym sym;
      fock::onspace space;
      linalg::matrix<Tm> coeff;
};
// renorm_basis: just like atomic basis (vector of symmetry sectors)
template <typename Tm>
using renorm_basis = std::vector<renorm_sector<Tm>>;

// rbasis for type-0 physical site 
template <typename Tm>
renorm_basis<Tm> get_rbasis_phys(){
   const bool Htype = tools::is_complex<Tm>();
   renorm_basis<Tm> rbasis(2);
   rbasis[0].sym = qsym(0,0);
   rbasis[0].space.push_back(fock::onstate("00"));
   rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
   rbasis[1].sym = qsym(2,0);
   rbasis[1].space.push_back(fock::onstate("11"));
   rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
   if(Htype){
      rbasis.resize(3);
      rbasis[2].sym = qsym(1,0);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].space.push_back(fock::onstate("10")); // b
      rbasis[2].coeff = linalg::identity_matrix<Tm>(2);
   }else{
      rbasis.resize(4);
      rbasis[2].sym = qsym(1,1);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].coeff = linalg::identity_matrix<Tm>(1);
      rbasis[3].sym = qsym(1,-1);
      rbasis[3].space.push_back(fock::onstate("10")); // b
      rbasis[3].coeff = linalg::identity_matrix<Tm>(1);
   }
   return rbasis;
}

} // ctns

#endif
