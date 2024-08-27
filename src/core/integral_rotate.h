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
         const int norb = urot.rows();
         std::cout << "\nintegral::rotate_spatial for int1e: norb=" << norb << std::endl;
         auto t0 = tools::get_time();
         
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
         
         auto t1 = tools::get_time();
         tools::timing("integral::rotate_spatial", t0, t1);
      }

   template <typename Tm>
      void rotate_spatial(const two_body<Tm>& int2e,
            two_body<Tm>& int2e_new,
            const linalg::matrix<Tm>& urot){
         const int norb = urot.rows();
         std::cout << "\nintegral::rotate_spatial for int2e: norb=" << norb << std::endl;
         auto t0 = tools::get_time();

         // initialize memory
         assert(int2e.sorb == 2*norb);
         int sorb = int2e.sorb;
         int2e_new.sorb = sorb;
         int2e_new.init_mem();
         
         // large mocoefficient matrix
         linalg::matrix<Tm> mocoeff(sorb,sorb);
         linalg::matrix<Tm> mocoeff_conj(sorb,sorb);
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               if(i%2 != j%2) continue;
               mocoeff(i,j) = urot(i/2,j/2); 
               mocoeff_conj(i,j) = tools::conjugate(urot(i/2,j/2)); 
            }
         }
         
         // we need to use this structure, because 
         // <<pq||kl> does not have 8-fold symmetry!!!
         int pair = sorb*(sorb-1)/2;
         linalg::matrix<Tm> int2e_half(pair,pair); 
         
         // half transformation-1: 
         // <ij||kl> = sum_rs <ij||rs>*C[r,k]*C[s,l] (i>j)
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               size_t ij = i*(i-1)/2+j;
               // O[r,s]
               linalg::matrix<Tm> tmp1(sorb,sorb);
               for(int r=0; r<sorb; r++){
                  for(int s=0; s<r; s++){
                     tmp1(r,s) = int2e.get(i,j,r,s);
                     tmp1(s,r) = -tmp1(r,s);
                  }
               }
               // X[k,l] = C[r,k]*(O[r,s]*C[s,l])
               auto tmp2 = linalg::xgemm("N","N",tmp1,mocoeff);
               auto tmp3 = linalg::xgemm("T","N",mocoeff,tmp2);
               // save
               for(int k=0; k<sorb; k++){
                  for(int l=0; l<k; l++){
                     size_t kl = k*(k-1)/2+l;
                     int2e_half(ij,kl) = tmp3(k,l);
                  } // l
               } // k
            } // j
         } // i
         
         // half transformation-2: 
         // <ij||kl> = sum_pq <pq||kl>*C[p,i]*C[q,j]
         for(int k=0; k<sorb; k++){
            for(int l=0; l<k; l++){
               size_t kl = k*(k-1)/2+l;
               // O[p,q] 
               linalg::matrix<Tm> tmp1(sorb,sorb);
               for(int p=0; p<sorb; p++){
                  for(int q=0; q<p; q++){
                     size_t pq = p*(p-1)/2+q;
                     tmp1(p,q) = int2e_half(pq,kl);
                     tmp1(q,p) = -tmp1(p,q);
                  }
               }
               // X[i,j] = C_conj[p,i]*(O[p,q]*C_conj[q,j])
               auto tmp2 = linalg::xgemm("N","N",tmp1,mocoeff_conj);
               auto tmp3 = linalg::xgemm("T","N",mocoeff_conj,tmp2);
               // save
               for(int i=0; i<sorb; i++){
                  for(int j=0; j<i; j++){
                     size_t ij = i*(i-1)/2+j;
                     int2e_new.set(i,j,k,l,tmp3(i,j));
                  } // l
               } // k
            } // j
         } // i
         int2e_new.initQ();
         
         const bool debug = false;
         if(debug){
            std::cout << std::setprecision(8) << std::endl;
            // old
            std::cout << "|int2e|=" << linalg::xnrm2(int2e.data.size(), int2e.data.data()) << std::endl;
            auto tensor0 = get_fulltensor(int2e);
            check_fulltensor(tensor0, sorb);
            // new
            std::cout << "|int2e_new|=" << linalg::xnrm2(int2e_new.data.size(), int2e_new.data.data()) << std::endl;
            auto tensor1 = get_fulltensor(int2e_new);
            check_fulltensor(tensor1, sorb);
            // plain
            rotate_spatial_plain(int2e,int2e_new,urot);
            auto tensor2 = get_fulltensor(int2e_new);
            check_fulltensor(tensor2, sorb);
            linalg::xaxpy(tensor2.size(), -1.0, tensor1.data(), tensor2.data());
            std::cout << "diff=" << linalg::xnrm2(tensor2.size(), tensor2.data()) << std::endl;
            exit(1);
         }

         auto t1 = tools::get_time();
         tools::timing("integral::rotate_spatial", t0, t1);
      }

} // integral

#endif
