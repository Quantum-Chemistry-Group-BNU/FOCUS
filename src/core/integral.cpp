#include "integral.h"
#include "blas.h"

using namespace integral;

template <typename Tm>
linalg::matrix<Tm> one_body<Tm>::mat_full() const{
   linalg::matrix<Tm> mat(sorb,sorb,data.data());
   return mat;
}

template <typename Tm>
linalg::matrix<Tm> one_body<Tm>::mat_alpha() const{
   const int norb = sorb/2;
   linalg::matrix<Tm> mat(norb,norb);
   for(int j=0; j<norb; j++){
      linalg::xcopy(norb, data[(2*j)*sorb], 2, mat.col(j), 1);
   }
   return mat; 
}

template <typename Tm>
linalg::matrix<Tm> one_body<Tm>::mat_beta() const{
   const int norb = sorb/2;
   linalg::matrix<Tm> mat(norb,norb);
   for(int j=0; j<norb; j++){
      linalg::xcopy(norb, data[(2*j+1)*sorb+1], 2, mat.col(j), 1);
   }
   return mat;
}

template <typename Tm>
void one_body<Tm>::from_spatial(const linalg::matrix<Tm>& mat_alpha,
      const linalg::matrix<Tm>& mat_beta){
   assert(mat_alpha.rows() == mat_alpha.cols() and 2*mat_alpha.rows() == sorb);
   assert(mat_beta.rows() == mat_beta.cols() and 2*mat_beta.rows() == sorb);
   const int norb = sorb/2;
   // copy alpha
   for(int j=0; j<mat_alpha.cols(); j++){
      linalg::xcopy(norb, mat_alpha.col(j), 1, data[(2*j)*sorb], 2);
   }
   // copy beta
   for(int j=0; j<mat_beta.cols(); j++){
      linalg::xcopy(norb, mat_beta.col(j), 1, data[(2*j+1)*sorb+1], 2);
   }
}

template <typename Tm>
one_body<Tm> one_body<Tm>::rotate_spatial(const linalg::matrix<Tm>& urot) const{
   one_body<Tm> int1e_new;
   /*
   const int norb = urot.rows();
   assert(sorb == 2*norb);
   one_body<Tm> int1e_new(sorb);
   // U^+*M*U;
   auto int1e_alpha = this->mat_alpha();
   auto tmpa = linalg::xgemm("N","N",int1e_alpha,urot);
   int1e_alpha = linalg::xgemm("C","N",urot,tmpa);
   // U^+*M*U;
   auto int1e_beta = this->mat_beta();
   auto tmpb = linalg::xgemm("N","N",int1e_beta,urot);
   int1e_beta = linalg::xgemm("C","N",urot,tmpb);
   // construct
   int1e_new.from_spatial(int1e_alpha,int1e_beta);
   */
   return int1e_new;
}
