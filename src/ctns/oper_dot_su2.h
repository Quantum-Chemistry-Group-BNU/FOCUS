#ifndef OPER_DOT_SU2_H
#define OPER_DOT_SU2_H

/*
   Dot operators: CABPQSH

   The local basis is {|0>,|2>,|a>,|b>} in consistent with ctns_phys.h

   We use the convention that p1+*p2+*q2*q1 where p1<p2 and q2>q1, i.e., 
   The index in the middle is larger than that close to the boundary.
   This is different from the ordering used in onstate.h
*/

#include "init_phys.h"
#include "oper_dict.h"

namespace ctns{

   const bool debug_oper_dot_su2 = true;
   extern const bool debug_oper_dot_su2;

   // init local operators on dot
   template <typename Tm>
      void oper_init_dot(opersu2_dict<Tm>& qops,
            const int isym,
            const bool ifkr,
            const int kp,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const int size,
            const int rank,
            const bool ifdist1){
         // setup basic information
         qops.sorb = int2e.sorb;
         qops.isym = isym;
         qops.ifkr = ifkr;
         qops.cindex.push_back(2*kp);
         if(!ifkr) qops.cindex.push_back(2*kp+1);
         // rest of spatial orbital indices
         for(int k=0; k<int1e.sorb/2; k++){
            if(k == kp) continue;
            qops.krest.push_back(k);
         }
         auto qphys = get_qbond_phys(isym);
         qops.qbra = qphys;
         qops.qket = qphys;
         qops.oplist = "CABPQSH";
         // initialize memory
         qops.init(true);
         // compute local operators on dot
         oper_dot_opC(qops, kp);
         oper_dot_opA(qops, kp);
         oper_dot_opB(qops, kp);
         oper_dot_opP(qops, kp, int2e);
         oper_dot_opQ(qops, kp, int2e);
         oper_dot_opS(qops, kp, int2e, int1e);
         oper_dot_opH(qops, kp, int2e, int1e);
         // scale full {Sp,H} on dot to avoid repetition in parallelization
#ifndef SERIAL
         if(size > 1){
            for(auto& pr : qops('S')){
               int iproc = distribute1(ifkr,size,pr.first);
               if(iproc != rank) pr.second.set_zero();
            }
            if(rank != 0) qops('H')[0].set_zero();
         }
#endif
      }

   // kA^+
   template <typename Tm>
      void oper_dot_opC(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opC" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0] = kA^+
         // [[0. 0. 0.]
         //  [0. 0. -sqrt(2)]
         //  [1. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,2) = -std::sqrt(2.0);
         mat(2,0) = 1;
         qops('C')[ka].from_matrix(mat);
         if(debug_oper_dot_su2) qops('C')[ka].to_matrix().print("c0+");
      }

   // A[kA,kB] = kA^+kB^+
   template <typename Tm>
      void oper_dot_opA(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opA" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(c[1])
         // [[0. 0. 0.]
         //  [sqrt(2) 0. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,0) = std::sqrt(2.0);
         qops('A')[oper_pack(ka,kb)].from_matrix(mat);
         if(debug_oper_dot_su2) qops('A')[oper_pack(ka,kb)].to_matrix().print("App0");
      }

   // B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
   // B[kB,kA] = kB^+kA = B[kA,kB].K(1)
   // B[kB,kB] = kB^+kB = B[kA,kA].K(2)
   template <typename Tm>
      void oper_dot_opB(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opB" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [Bpp]^0
         // [[0. 0. 0. ]
         //  [0. sqrt(2) 0. ]
         //  [0. 0. 1/sqrt(2)]]
         linalg::matrix<Tm> matBpp0(3,3);
         matBpp0(1,1) = std::sqrt(2);
         matBpp0(2,2) = 1.0/std::sqrt(2);
         qops('B')[oper_pack(ka,ka)].from_matrix(matBpp0);
         if(debug_oper_dot_su2) qops('B')[oper_pack(ka,ka)].to_matrix().print("Bpp0");
         // [Bpp]^1
         // [[0. 0. 0. ]
         //  [0. 0. 0. ]
         //  [0. 0. sqrt(3/2)]]
         linalg::matrix<Tm> matBpp1(3,3);
         matBpp1(2,2) = std::sqrt(3.0/2.0);
         qops('B')[oper_pack(ka,kb)].from_matrix(matBpp1);
         if(debug_oper_dot_su2) qops('B')[oper_pack(ka,kb)].to_matrix().print("Bpp1");
      }

   // build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   template <typename Tm>
      void oper_dot_opH(opersu2_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opH" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0.]
         //  [0. 1. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,1) = 1;
         stensor2su2<Tm> qt2nanb(qsym(3,0,0), qops.qbra, qops.qket);
         qt2nanb.from_matrix(mat);
         const auto& qBpp0 = qops('B')[oper_pack(ka,ka)];
         int N = qops('H')[0].size();
         Tm* ptr = qops('H')[0].data();
         linalg::xaxpy(N, int2e.get(ka,kb,ka,kb), qt2nanb.data(), ptr);
         linalg::xaxpy(N, std::sqrt(2.0)*int1e.get(ka,ka), qBpp0.data(), ptr);
         if(debug_oper_dot_su2) qops('H')[0].to_matrix().print("H");
      }

   // build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   template <typename Tm>
      void oper_dot_opS(opersu2_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opS" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [Ts]^1/2
         // [[0. 0. 0. ]
         //  [0. 0. 0. ]
         //  [0. 1. 0. ]]
         linalg::matrix<Tm> mat(3,3);
         mat(2,1) = 1;
         stensor2su2<Tm> qt2Ts(qsym(3,-1,1), qops.qbra, qops.qket); // ka^+ kb ka
         qt2Ts.from_matrix(mat);
         // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
         const auto akA = qops('C')[ka].H();
         for(auto& pr : qops('S')){
            int p = pr.first;
            auto& opS = pr.second;
            int N = opS.size();
            assert(p%2 == 0);
            linalg::xaxpy(N, 0.5*int1e.get(p,ka), akA.data(), opS.data());
            linalg::xaxpy(N, int2e.get(p,kb,ka,kb), qt2Ts.data(), opS.data());
         }
      }

   // Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
   // 				   <pq||ab> ba = (a^+b^+)^+
   template <typename Tm>
      void oper_dot_opP(opersu2_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opP" << std::endl;   
         int ka = 2*k0, kb = ka+1;
         // [[0. -sqrt(2) 0.]
         //  [0. 0. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(0,1) = -std::sqrt(2.0);
         stensor2su2<Tm> qt2Css0(qsym(3,-2,0), qops.qbra, qops.qket);
         qt2Css0.from_matrix(mat);
         for(auto& pr : qops('P')){
            int pq = pr.first;
            auto& opP = pr.second;
            auto upq = oper_unpack(pq);
            int p = upq.first, q = upq.second; 
            int N = opP.size();
            if(p%2 == q%2) continue; // only for Ppq0[os]
            linalg::xaxpy(N, int2e.get(p,q,ka,kb), qt2Css0.data(), opP.data());
         }
      }

   // Qps = <pq||sr> aq^+ar
   template <typename Tm>
      void oper_dot_opQ(opersu2_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opQ" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         const auto& qt2Bpp0 = qops('B')[oper_pack(ka,ka)];
         const auto& qt2Bpp1 = qops('B')[oper_pack(ka,kb)];
         for(auto& pr : qops('Q')){
            int ps = pr.first;
            auto& opQ = pr.second;
            auto ups = oper_unpack(ps);
            int p = ups.first, s = ups.second;
            int N = opQ.size();
            if(p%2 == s%2){ // Qps0
               // <pk||sk> + <pk'||sk'> = 2[ps|kk] - [pk|ks]
               auto erifac = int2e.get(p,ka,s,ka) + int2e.get(p,kb,s,kb);
               linalg::xaxpy(N, erifac, qt2Bpp0.data(), opQ.data());
            }else{ // Qps1
               // <pk'||s'k> = -[pk|ks]
               auto erifac = int2e.get(p,kb,s,ka);
               linalg::xaxpy(N, erifac, qt2Bpp1.data(), opQ.data());
            }
         }
      }

} // ctns

#endif
