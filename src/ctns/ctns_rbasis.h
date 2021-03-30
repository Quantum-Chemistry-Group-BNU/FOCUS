#ifndef CTNS_RBASIS_H
#define CTNS_RBASIS_H

#include <string>
#include <vector>
#include <tuple>
#include "../core/onspace.h"
#include "../core/matrix.h"
/*
#include "ctns_qsym.h"
*/

namespace ctns{

// renorm_sector: renormalized states from determinants
template <typename Tm>
struct renorm_sector{
   public:
      void print(const std::string name, const int level=0) const;      
   public:
      qsym sym;
      fock::onspace space;
      linalg::matrix<typename Tm::dtype> coeff;
};

template <typename Tm>
void renorm_sector<Tm>::print(const std::string name, const int level) const
{
   std::cout << "renorm_sector: " << name << " qsym=" << sym 
             << " shape=" << coeff.rows() << "," << coeff.cols() << std::endl; 
   if(level >= 1){
      for(int i=0; i<space.size(); i++){
         std::cout << " idx=" << i << " state=" << space[i] << std::endl;
      }
      if(level >= 2) coeff.print("coeff");
   }
}

// renorm_basis: just like atomic basis (vector of symmetry sectors)
template <typename Tm>
using renorm_basis = std::vector<renorm_sector<Tm>>;

// rows,cols of rbasis
template <typename Tm>
std::pair<int,int> get_shape(const renorm_basis<Tm>& rbasis){
   int rows = 0, cols = 0;
   for(int i=0; i<rbasis.size(); i++){
      rows += rbasis[i].coeff.rows();
      cols += rbasis[i].coeff.cols();
   }
   return std::make_pair(rows,cols);
}

/*
template <typename Tm>
qsym_space get_qsym_space(const renorm_basis<Tm>& rbasis){
   qsym_space qs;
   for(int i=0; i<rbasis.size(); i++){
      qs.dims.push_back({rbasis[i].sym, rbasis[i].coeff.cols()});
   }
   return qs;
}
*/

// rbasis for type-0 physical site 
template <typename Tm>
void get_rbasis_phys(renorm_basis<Tm>& rbasis){
   // (N)
   if(Tm::isym == 1){
      rbasis.resize(3);
      // |00>
      rbasis[0].sym = qsym(0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
      // |11>
      rbasis[1].sym = qsym(2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
      // a=|01> & b=|10>
      rbasis[2].sym = qsym(1,0);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].space.push_back(fock::onstate("10")); // b
      rbasis[2].coeff = linalg::identity_matrix<typename Tm::dtype>(2);
   }else if(Tm::isym == 2){
      rbasis.resize(4);
      // |00>
      rbasis[0].sym = qsym(0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
      // |11>
      rbasis[1].sym = qsym(2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
      // |01>
      rbasis[2].sym = qsym(1,1);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
      // |10>
      rbasis[3].sym = qsym(1,-1);
      rbasis[3].space.push_back(fock::onstate("10")); // b
      rbasis[3].coeff = linalg::identity_matrix<typename Tm::dtype>(1);
   }
}

} // ctns

#endif
