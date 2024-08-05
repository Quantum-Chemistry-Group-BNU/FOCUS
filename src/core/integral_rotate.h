#ifndef INTEGRAL_ROTATE_H
#define INTEGRAL_ROTATE_H

#include "integral.h"

namespace integral{

   template <typename Tm>
      std::vector<Tm> get_fulltensor(const two_body<Tm>& int2e){
         int sorb = int2e.sorb;
         // t[i,j,k,l] = <ij||kl>
         std::vector<Tm> tensor(sorb*sorb*sorb*sorb,0);
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<sorb; l++){
                     size_t addr = ((i*sorb+j)*sorb+k)*sorb+l;
                     tensor[addr] = int2e.get(i,j,k,l);
                  }
               }
            }
         }
         return tensor;
      }

   template <typename Tm>
      void check_fulltensor(const std::vector<Tm>& tensor,
            const int sorb){
         // i<->j
         double diffij = 0;
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<sorb; l++){
                     size_t addr1 = ((i*sorb+j)*sorb+k)*sorb+l;
                     size_t addr2 = ((j*sorb+i)*sorb+k)*sorb+l;
                     diffij += std::norm(tensor[addr1] + tensor[addr2]);
                  }
               }
            }
         }
         // k<->l
         double diffkl = 0;
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<sorb; l++){
                     size_t addr1 = ((i*sorb+j)*sorb+k)*sorb+l;
                     size_t addr2 = ((i*sorb+j)*sorb+l)*sorb+k;
                     diffkl += std::norm(tensor[addr1] + tensor[addr2]);
                  }
               }
            }
         }
         // ij<->kl
         double diffijkl = 0;
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<sorb; l++){
                     size_t addr1 = ((i*sorb+j)*sorb+k)*sorb+l;
                     size_t addr2 = ((k*sorb+l)*sorb+i)*sorb+j;
                     diffijkl += std::norm(tensor[addr1] - tools::conjugate(tensor[addr2]));
                  }
               }
            }
         }
         std::cout << "check_fulltensor:" << std::endl;
         std::cout << " norm=" << linalg::xnrm2(tensor.size(), tensor.data()) << std::endl;
         std::cout << " diffij=" << diffij << std::endl;
         std::cout << " diffkl=" << diffkl << std::endl;
         std::cout << " diffijkl=" << diffijkl << std::endl;
      }

   // rotate integrals
   template <typename Tm>
      void rotate_spatial_plain(const one_body<Tm>& int1e,
            one_body<Tm>& int1e_new,
            const linalg::matrix<Tm>& urot){
         std::cout << "rotate_spatial_plain for int1e" << std::endl;
         const int norb = urot.rows();
         assert(int1e.sorb == 2*norb);
         int sorb = int1e.sorb;
         int1e_new.sorb = sorb;
         int1e_new.init_mem();
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               Tm sum = 0.0;
               for(int p=i%2; p<sorb; p+=2){
                  for(int q=j%2; q<sorb; q+=2){
                     sum += int1e.get(p,q)*tools::conjugate(urot(p/2,i/2))*urot(q/2,j/2);
                  } // q 
               } // p
               int1e_new.set(i,j,sum);
            } // j
         } // i
      }

   template <typename Tm>
      void rotate_spatial_plain(const two_body<Tm>& int2e,
            two_body<Tm>& int2e_new,
            const linalg::matrix<Tm>& urot){
         std::cout << "rotate_spatial_plain for int2e" << std::endl;
         const int norb = urot.rows();
         assert(int2e.sorb == 2*norb);
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
                     // einsum('pqrs,pi,qj,rk,sl->ijkl',int2e,uc,uc,u,u)
                     Tm sum = 0.0;
                     for(int p=i%2; p<sorb; p+=2){
                        for(int q=j%2; q<sorb; q+=2){
                           for(int r=k%2; r<sorb; r+=2){
                              for(int s=l%2; s<sorb; s+=2){
                                 sum += int2e.get(p,q,r,s)
                                    *tools::conjugate(urot(p/2,i/2))
                                    *tools::conjugate(urot(q/2,j/2))
                                    *urot(r/2,k/2)
                                    *urot(s/2,l/2);
                              } // s
                           } // r
                        } // q 
                     } // p
                     int2e_new.set(i,j,k,l,sum);
                  } // l
               } // k
            } // j
         } // i
         int2e_new.initQ();
      }

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
      void rotate_spatial(const two_body<Tm>& int2e,
            two_body<Tm>& int2e_new,
            const linalg::matrix<Tm>& urot){
         std::cout << "rotate_spatial for int2e" << std::endl;
         const int norb = urot.rows();
         assert(int2e.sorb == 2*norb);
         int sorb = int2e.sorb;
         int pair = sorb*(sorb-1)/2;
         // we need to use this structure because <<pq||kl> does not have 8-fold symmetry
         linalg::matrix<Tm> int2e_half(pair,pair); 
         // <ij||rs> -> <ij||kl> = sum_rs <ij||rs>*C[r,k]*C[s,l]
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               size_t ij = i*(i-1)/2+j;
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<k; l++){
                     size_t kl = k*(k-1)/2+l;
                     Tm sum = 0.0;
                     for(int r=k%2; r<sorb; r+=2){
                        for(int s=l%2; s<sorb; s+=2){
                           sum += int2e.get(i,j,r,s)
                              *urot(r/2,k/2)
                              *urot(s/2,l/2);
                        } // s
                     } // r
                     int2e_half(ij,kl) = sum;
                  } // l
               } // k
            } // j
         } // i
         // <ij||kl> = sum_pq <pq||kl>*C[p,i]*C[q,j]
         int2e_new.sorb = sorb;
         int2e_new.init_mem();
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               size_t ij = i*(i-1)/2+j;
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<k; l++){
                     size_t kl = k*(k-1)/2+l;
                     if(kl < ij) continue;
                     Tm sum = 0.0;
                     for(int p=i%2; p<sorb; p+=2){
                        for(int q=j%2; q<sorb; q+=2){
                           if(p == q) continue;
                           size_t pq = p>q? p*(p-1)/2+q : q*(q-1)/2+p;
                           Tm sgn = p>q? 1.0 : -1.0;
                           sum += int2e_half(pq,kl)
                              *sgn
                              *tools::conjugate(urot(p/2,i/2))
                              *tools::conjugate(urot(q/2,j/2));
                        } // q 
                     } // p
                     int2e_new.set(i,j,k,l,sum);
                  } // l
               } // k
            } // j
         } // i
         const bool debug = false;
         if(debug){
            std::cout << std::setprecision(8) << std::endl;
            std::cout << linalg::xnrm2(int2e.data.size(), int2e.data.data()) << std::endl;
            std::cout << linalg::xnrm2(int2e_new.data.size(), int2e_new.data.data()) << std::endl;
            auto tensor0 = get_fulltensor(int2e);
            auto tensor1 = get_fulltensor(int2e_new);
            check_fulltensor(tensor0, sorb);
            check_fulltensor(tensor1, sorb);
            // plain
            rotate_spatial_plain(int2e,int2e_new,urot);
            auto tensor2 = get_fulltensor(int2e_new);
            check_fulltensor(tensor2, sorb);
            linalg::xaxpy(tensor2.size(), -1.0, tensor1.data(), tensor2.data());
            std::cout << "diff=" << linalg::xnrm2(tensor2.size(), tensor2.data()) << std::endl;
            exit(1);
         }
         int2e_new.initQ();
      }

} // integral

#endif
