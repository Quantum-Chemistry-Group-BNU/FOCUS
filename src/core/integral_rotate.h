#ifndef INTEGRAL_ROTATE_H
#define INTEGRAL_ROTATE_H

#include "integral.h"

namespace integral{

   // rotate integrals
   template <typename Tm>
      void rotate_spatial(const one_body<Tm>& int1e,
            one_body<Tm>& int1e_new,
            const linalg::matrix<Tm>& urot){
         std::cout << "rotate_spatial for int1e" << std::endl;
         const int norb = urot.rows();
         assert(int1e.sorb == 2*norb);
         int1e_new.sorb = int1e.sorb;
         int1e_new.init_mem();
         // U^+*M*U;
         auto int1e_alpha = int1e.mat_alpha();
         auto tmpa = linalg::xgemm("N","N",int1e_alpha,urot);
         int1e_alpha = linalg::xgemm("C","N",urot,tmpa);
         // U^+*M*U;
         auto int1e_beta = int1e.mat_beta();
         auto tmpb = linalg::xgemm("N","N",int1e_beta,urot);
         int1e_beta = linalg::xgemm("C","N",urot,tmpb);
         // construct
         int1e_new.from_spatial(int1e_alpha,int1e_beta);
      }

   template <typename Tm>
      void rotate_spatial_index(const two_body<Tm>& int2e,
            two_body<Tm>& int2e_new,
            const linalg::matrix<Tm>& urot,
            const int index){
         int sorb = int2e.sorb;
         int2e_new.sorb = sorb;
         int2e_new.init_mem();
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               size_t ij = i*(i-1)/2+j;
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<k; l++){
                     size_t kl = k*(k-1)/2+l;
                     if(kl < ij) continue;
                     // <i1j2||k1l2> = <i1j2|k1l2> - <i1j2|l1k2> 
                     Tm sum = 0.0;
                     if(index == 0){
                        // einsum('pjkl,pi->ijkl',int2e,urot.conj())
                        for(int p=i%2; p<sorb; p+=2){
                           sum += int2e.get(p,j,k,l)*tools::conjugate(urot(p/2,i/2));
                        }
                     }else if(index == 1){
                        // einsum('ijkl,pj->ijkl',int2e,urot.conj())
                        for(int p=j%2; p<sorb; p+=2){
                           sum += int2e.get(i,p,k,l)*tools::conjugate(urot(p/2,j/2));
                        }
                     }else if(index == 2){
                        // einsum('ijpl,pk->ijkl',int2e,urot)
                        for(int p=k%2; p<sorb; p+=2){
                           sum += int2e.get(i,j,p,l)*urot(p/2,k/2);
                        }
                     }else if(index == 3){
                        // einsum('ijkp,pl->ijkl',int2e,urot)
                        for(int p=l%2; p<sorb; p+=2){
                           sum += int2e.get(i,j,k,p)*urot(p/2,l/2);
                        }
                     }
                     int2e_new.set(i,j,k,l,sum);
                  } // l
               } // k
            } // j
         } // i
      }

   template <typename Tm>
      void rotate_spatial(const two_body<Tm>& int2e,
            two_body<Tm>& int2e_new,
            const linalg::matrix<Tm>& urot){
         std::cout << "rotate_spatial for int2e" << std::endl;
         const int norb = urot.rows();
         assert(int2e.sorb == 2*norb);
         two_body<Tm> int2e_tmp1, int2e_tmp2;
         int2e_tmp1 = int2e;
         rotate_spatial_index(int2e_tmp1, int2e_tmp2, urot, 3);
         rotate_spatial_index(int2e_tmp2, int2e_tmp1, urot, 2);
         rotate_spatial_index(int2e_tmp1, int2e_tmp2, urot, 1);
         rotate_spatial_index(int2e_tmp2, int2e_tmp1, urot, 0);
         int2e_new = std::move(int2e_tmp1);
         int2e_new.initQ();
      }

} // integral

#endif
