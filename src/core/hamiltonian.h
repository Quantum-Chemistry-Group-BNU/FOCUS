#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <tuple>
#include "onstate.h"
#include "onspace.h"
#include "integral.h"
#include "matrix.h"

namespace fock{

   // <Di|H|Di> =  hpp + \sum_{p>q}<pq||pq> = hpp + \sum_{p>q}Q[p,q] 
   template <typename Tm>
      double get_Hii(const onstate& state1,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         double Hii = 0.0;
         std::vector<int> olst;
         state1.get_olst(olst);
         for(int i=0; i<olst.size(); i++){
            int p = olst[i];
            Hii += std::real(int1e.get(p,p));
            for(int j=0; j<i; j++){
               int q = olst[j];
               Hii += int2e.getQ(p,q);
            }
         }
         return Hii;
      }

   // differ by single: hpq + <pk||qk> 
   template <typename Tm>
      std::pair<Tm,long> get_HijS(const onstate& state1, 
            const onstate& state2,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         int p[1], q[1];
         state1.diff_orb(state2,p,q);
         Tm Hij = int1e.get(p[0],q[0]); // hpq
                                        // loop over occupied state in state2
#ifdef DEBUG
         for(int k=0; k<state2.size(); k++){
            if(state2[k]){
               Hij += int2e.get(p[0],k,q[0],k);
            }
         }
#else
         for(int i=0; i<state2.len(); i++){
            unsigned long repr = state2.repr(i);
            while(repr != 0){
               int j = 63-__builtin_clzl(repr);
               int k = i*64+j;
               Hij += int2e.get(p[0],k,q[0],k); // <pk||qk>
               repr &= ~(1ULL<<j);
            }
         }
#endif
         int sgn = state1.parity(p[0])*state2.parity(q[0]);
         Hij *= static_cast<double>(sgn);
         long ph1 = sgn*(p[0]+q[0]*int1e.sorb);
         return std::make_pair(Hij, ph1);
      }

   // differ by double: <p0p1||q0q1>
   template <typename Tm>
      std::pair<Tm,long> get_HijD(const onstate& state1, 
            const onstate& state2,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         int p[2], q[2];
         state1.diff_orb(state2,p,q);
         int sgn = state1.parity(p[0])*state1.parity(p[1])
            *state2.parity(q[0])*state2.parity(q[1]);
         Tm Hij = static_cast<double>(sgn)*int2e.get(p[0],p[1],q[0],q[1]);
         long ph2 = sgn*(p[0]+(q[0]+(p[1]+q[1]*int1e.sorb)*int1e.sorb)*int1e.sorb);
         return std::make_pair(Hij, ph2);
      }

   // <Di|H|Dj>
   template <typename Tm>
      Tm get_Hij(const onstate& state1, 
            const onstate& state2,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         Tm Hij = 0.0;
         auto pr = state1.diff_type(state2);
         if(pr == std::make_pair(0,0)){
            Hij = get_Hii(state1,int2e,int1e);
         }else if(pr == std::make_pair(1,1)){
            Hij = get_HijS(state1,state2,int2e,int1e).first;
         }else if(pr == std::make_pair(2,2)){
            Hij = get_HijD(state1,state2,int2e,int1e).first;
         }
         return Hij;
      }

   // generate represenation of H in this space
   template <typename Tm>
      linalg::matrix<Tm> get_Hmat(const onspace& space,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore){
         auto dim = space.size();
         std::cout << "fock::get_Hmat dim=" << dim << std::endl; 
         linalg::matrix<Tm> H(dim,dim);
         // column major
         for(size_t j=0; j<dim; j++){
            for(size_t i=0; i<dim; i++){
               H(i,j) = get_Hij(space[i],space[j],int2e,int1e);
            }
            H(j,j) += ecore;
         }
         return H;
      }

} // fock

#endif
