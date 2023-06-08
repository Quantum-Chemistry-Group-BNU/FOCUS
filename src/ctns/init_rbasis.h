#ifndef INIT_RBASIS_H
#define INIT_RBASIS_H

#include <string>
#include <vector>
#include <tuple>
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../qtensor/qtensor.h"

namespace ctns{

   // renorm_sector: renormalized states from determinants
   template <typename Tm>
      struct renorm_sector{
         private:
            // serialize [for MPI] 
            friend class boost::serialization::access;
            template <class Archive>
               void serialize(Archive & ar, const unsigned int version){
                  ar & sym & space & coeff;
               }
         public:
            void print(const std::string name, const int level=0) const{
               std::cout << "renorm_sector: " << name << " qsym=" << sym 
                  << " shape=" << coeff.rows() << "," << coeff.cols() 
                  << std::endl; 
               if(level >= 1){
                  for(int i=0; i<space.size(); i++){
                     std::cout << " idx=" << i << " state=" << space[i] << std::endl;
                  }
                  if(level >= 2) coeff.print("coeff");
               }
            }
            size_t size() const{ return coeff.size(); } 
         public:
            qsym sym;
            fock::onspace space;
            linalg::matrix<Tm> coeff;
      };
   // renorm_basis: just like atomic basis (vector of symmetry sectors)
   template <typename Tm>
      using renorm_basis = std::vector<renorm_sector<Tm>>;

   // count rows,cols of rbasis
   template <typename Tm>
      std::pair<int,int> get_shape(const renorm_basis<Tm>& rbasis){
         int rows = 0, cols = 0;
         for(int i=0; i<rbasis.size(); i++){
            rows += rbasis[i].coeff.rows();
            cols += rbasis[i].coeff.cols();
         }
         return std::make_pair(rows,cols);
      }

   // qbond of rbasis
   template <typename Tm>
      qbond get_qbond(const renorm_basis<Tm>& rbasis){
         qbond qs;
         for(int i=0; i<rbasis.size(); i++){
            qs.dims.push_back({rbasis[i].sym, rbasis[i].coeff.cols()});
         }
         return qs;
      }

} // ctns

#endif
