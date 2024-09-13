#ifndef OPER_DOT_H
#define OPER_DOT_H

/*
   Dot operators: CABPQSH

   The local basis is {|0>,|2>,|a>,|b>} in consistent with ctns_phys.h

   We use the convention that p1+*p2+*q2*q1 where p1<p2 and q2>q1, i.e., 
   The index in the middle is larger than that close to the boundary.
   This is different from the ordering used in onstate.h
*/

#include "init_phys.h"
#include "oper_dict.h"
#include "oper_dot_local.h"

namespace ctns{

   const bool debug_oper_dot = false;
   extern const bool debug_oper_dot;

   // init local operators on dot
   template <bool ifab, typename Tm>
      void oper_init_dot(qoper_dict<ifab,Tm>& qops,
            const int isym,
            const bool ifkr,
            const int kp,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const int size,
            const int rank){
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
      void oper_dot_opC(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opC" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0] = kA^+
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 1.]
         //  [1. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(1,3) = 1;
         mat(2,0) = 1;
         qops('C')[ka].from_matrix(mat);
         if(debug_oper_dot) qops('C')[ka].to_matrix().print("c0+");
         // c[1] = kB^+ 
         if(not qops.ifkr){
            // also store c[1] = kB^+
            // [[ 0.  0.  0.  0.]
            //  [ 0.  0. -1.  0.]
            //  [ 0.  0.  0.  0.]
            //  [ 1.  0.  0.  0.]]
            linalg::matrix<Tm> mat(4,4);
            mat(1,2) = -1;
            mat(3,0) = 1;
            qops('C')[kb].from_matrix(mat);
            if(debug_oper_dot) qops('C')[kb].to_matrix().print("c1+");
         }
      }

   // A[kA,kB] = kA^+kB^+
   template <typename Tm>
      void oper_dot_opA(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opA" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(c[1])
         // [[0. 0. 0. 0.]
         //  [1. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(1,0) = 1;
         qops('A')[oper_pack(ka,kb)].from_matrix(mat);
         if(debug_oper_dot) qops('A')[oper_pack(ka,kb)].to_matrix().print("c0^+c1^+");
      }

   // B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
   // B[kB,kA] = kB^+kA = B[kA,kB].K(1)
   // B[kB,kB] = kB^+kB = B[kA,kA].K(2)
   template <typename Tm>
      void oper_dot_opB(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opB" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(a[0])
         // [[0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 1. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mataa(4,4);
         mataa(1,1) = 1;
         mataa(2,2) = 1;
         qops('B')[oper_pack(ka,ka)].from_matrix(mataa);
         if(debug_oper_dot) qops('B')[oper_pack(ka,ka)].to_matrix().print("c0^+c0");
         // c[0].dot(a[1])
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 1.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> matab(4,4);
         matab(2,3) = 1;
         qops('B')[oper_pack(ka,kb)].from_matrix(matab);
         if(debug_oper_dot) qops('B')[oper_pack(ka,kb)].to_matrix().print("c0^+c1");
         // NOTE: even in this case, we can still use Hermicity to reduce Bba = b^+a!
         if(not qops.ifkr){
            // Bbb:
            // c[1].dot(a[1])
            // [[0. 0. 0. 0.]
            //  [0. 1. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 1.]]
            linalg::matrix<Tm> matbb(4,4);
            matbb(1,1) = 1;
            matbb(3,3) = 1;
            qops('B')[oper_pack(kb,kb)].from_matrix(matbb);
            if(debug_oper_dot) qops('B')[oper_pack(kb,kb)].to_matrix().print("c1^+c1");
         }
         if(!qops.ifhermi){
            // c[1].dot(a[0])
            // [[0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 1. 0.]]
            linalg::matrix<Tm> matba(4,4);
            matba(3,2) = 1;
            qops('B')[oper_pack(kb,ka)].from_matrix(matba);
            if(debug_oper_dot) qops('B')[oper_pack(kb,ka)].to_matrix().print("c1^+c0");
         }
      }

   // build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   template <typename Tm>
      void oper_dot_opH(oper_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opH" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(1,1) = 1;
         stensor2<Tm> qt2abba(qsym(qops.isym,0,0), qops.qbra, qops.qket);
         qt2abba.from_matrix(mat);
         const auto& qBaa = qops('B')[oper_pack(ka,ka)];
         const auto& qBab = qops('B')[oper_pack(ka,kb)];
         int N = qops('H')[0].size();
         Tm* ptr = qops('H')[0].data();
         linalg::xaxpy(N, int2e.get(ka,kb,ka,kb), qt2abba.data(), ptr);
         linalg::xaxpy(N, int1e.get(ka,ka), qBaa.data(), ptr);
         if(qops.ifkr){
            const auto& qBbb = qBaa.K(2);
            const auto& qBba = qBab.K(1);
            /*
               qops('H')[0] = int2e.get(ka,kb,ka,kb)*qt2abba
               + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBaa.K(2);
               + int1e.get(ka,kb)*qBab + int1e.get(kb,ka)*qBab.K(1);
               */
            linalg::xaxpy(N, int1e.get(ka,kb), qBab.data(), ptr);
            linalg::xaxpy(N, int1e.get(kb,kb), qBbb.data(), ptr);
            linalg::xaxpy(N, int1e.get(kb,ka), qBba.data(), ptr);
         }else{
            const auto& qBbb = qops('B')[oper_pack(kb,kb)];
            const auto& qBba = qops('B')[oper_pack(ka,kb)].H(); // recovered by Hermicity
            /*
               if(qops.isym == 0 or qops.isym == 1){
               qops('H')[0] = int2e.get(ka,kb,ka,kb)*qt2abba
               + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBbb
               + int1e.get(ka,kb)*qBab + int1e.get(kb,ka)*qBba;
               }else if(qops.isym == 2){
               qops('H')[0] = int2e.get(ka,kb,ka,kb)*qt2abba
               + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBbb;
               }
               */
            linalg::xaxpy(N, int1e.get(kb,kb), qBbb.data(), ptr);
            if(qops.isym == 0 or qops.isym == 1){
               linalg::xaxpy(N, int1e.get(ka,kb), qBab.data(), ptr);
               linalg::xaxpy(N, int1e.get(kb,ka), qBba.data(), ptr);
            }
         } // ifkr
      }

   // build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   template <typename Tm>
      void oper_dot_opS(oper_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opS" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(a[1].dot(a[0]))
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(2,1) = 1;
         auto sym_Sb = get_qsym_opS(qops.isym,1);
         stensor2<Tm> qt2aba(sym_Sb, qops.qbra, qops.qket); // ka^+ kb ka
         qt2aba.from_matrix(mat);
         // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
         if(qops.ifkr){
            const auto& akA = qops('C')[ka].H();
            const auto& akB = qops('C')[ka].H().K(1);
            const auto& qt2bba = -qt2aba.K(2); // [b^+ba] = (K[b^+ba])^* = [a^+ab]* = -[a^+ba]*
            /*
               for(int kp : krest){
               int pa = 2*kp, pb = pa+1;
               qops('S')[pa] = 0.5*int1e.get(pa,ka)*akA
               + 0.5*int1e.get(pa,kb)*akB
               + int2e.get(pa,ka,ka,kb)*qt2aba  // <pa||ab> p^+aba  		 
               + int2e.get(pa,kb,ka,kb)*qt2bba; // <pb||ab> p^+bba 
               } // kp
               */
            for(auto& pr : qops('S')){
               int pa = pr.first;
               auto& opS = pr.second;
               int N = opS.size();
               linalg::xaxpy(N, 0.5*int1e.get(pa,ka), akA.data(), opS.data());
               linalg::xaxpy(N, 0.5*int1e.get(pa,kb), akB.data(), opS.data());
               linalg::xaxpy(N, int2e.get(pa,ka,ka,kb), qt2aba.data(), opS.data());
               linalg::xaxpy(N, int2e.get(pa,kb,ka,kb), qt2bba.data(), opS.data());
            }
         }else{
            const auto& akA = qops('C')[ka].H();
            const auto& akB = qops('C')[kb].H();
            // c[1].dot(a[1].dot(a[0]))
            // [[0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 1. 0. 0.]]
            linalg::matrix<Tm> mat(4,4);
            mat(3,1) = 1;
            auto sym_Sa = get_qsym_opS(qops.isym,0);
            stensor2<Tm> qt2bba(sym_Sa, qops.qbra, qops.qket);
            qt2bba.from_matrix(mat);
            if(qops.isym == 0 or qops.isym == 1){
               /*
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
                  qops('S')[pa] = 0.5*int1e.get(pa,ka)*akA
                  + 0.5*int1e.get(pa,kb)*akB       // zero in NR
                  + int2e.get(pa,ka,ka,kb)*qt2aba  // zero in NR
                  + int2e.get(pa,kb,ka,kb)*qt2bba;
                  qops('S')[pb] = 0.5*int1e.get(pb,ka)*akA       // zero in NR
                  + 0.5*int1e.get(pb,kb)*akB
                  + int2e.get(pb,ka,ka,kb)*qt2aba 
                  + int2e.get(pb,kb,ka,kb)*qt2bba; // zero in NR
                  } // kp
                  */
               for(auto& pr : qops('S')){
                  int p = pr.first;
                  auto& opS = pr.second;
                  int N = opS.size();
                  linalg::xaxpy(N, 0.5*int1e.get(p,ka), akA.data(), opS.data());
                  linalg::xaxpy(N, 0.5*int1e.get(p,kb), akB.data(), opS.data());
                  linalg::xaxpy(N, int2e.get(p,ka,ka,kb), qt2aba.data(), opS.data());
                  linalg::xaxpy(N, int2e.get(p,kb,ka,kb), qt2bba.data(), opS.data());
               }
            }else if(qops.isym == 2){
               /*
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
               // S_{pA} += <pAkA||kAkB> kA^+kBkA + <pAkB||kAkB> kB^+kBkA 
               qops('S')[pa] = 0.5*int1e.get(pa,ka)*akA
               + int2e.get(pa,kb,ka,kb)*qt2bba;
               // S_{pB} += <pBkA||kAkB> kA^+kBkA + <pBkB||kAkB> kB^+kBkA 
               qops('S')[pb] = 0.5*int1e.get(pb,kb)*akB
               + int2e.get(pb,ka,ka,kb)*qt2aba; 
               } // kp
               */
               for(auto& pr : qops('S')){
                  int p = pr.first;
                  auto& opS = pr.second;
                  int N = opS.size();
                  if(p%2 == 0){
                     linalg::xaxpy(N, 0.5*int1e.get(p,ka), akA.data(), opS.data());
                     linalg::xaxpy(N, int2e.get(p,kb,ka,kb), qt2bba.data(), opS.data());
                  }else{
                     linalg::xaxpy(N, 0.5*int1e.get(p,kb), akB.data(), opS.data());
                     linalg::xaxpy(N, int2e.get(p,ka,ka,kb), qt2aba.data(), opS.data());
                  }
               }
            } // isym
         } // ifkr
      }

   // Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
   // 				   <pq||ab> ba = (a^+b^+)^+
   template <typename Tm>
      void oper_dot_opP(oper_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opP" << std::endl;   
         int ka = 2*k0, kb = ka+1;
         const auto& qt2ba = qops('A')[oper_pack(ka,kb)].H();
         if(qops.ifkr){
            /*
               for(int kp : krest){
               int pa = 2*kp, pb = pa+1;
               for(int kq : krest){
               int qa = 2*kq, qb = qa+1;
            // storage scheme for Ppq: according to spatial orbital
            if(kp < kq){
            // P[pA,qA] = <pA,qA||kA,kB> A[kA,kB]^+ (zero in NR)
            qops('P')[oper_pack(pa,qa)] = int2e.get(pa,qa,ka,kb)*qt2ba; 
            // P[pA,qB] = <pA,qB||kA,kB> A[kA,kB]^+
            qops('P')[oper_pack(pa,qb)] = int2e.get(pa,qb,ka,kb)*qt2ba;
            }else if(kp == kq){
            qops('P')[oper_pack(pa,pb)] = int2e.get(pa,pb,ka,kb)*qt2ba;
            }
            } // kq
            } // kp
            */
            for(auto& pr : qops('P')){
               int pq = pr.first;
               auto& opP = pr.second;
               auto upq = oper_unpack(pq);
               int p = upq.first, q = upq.second; 
               int N = opP.size();
               linalg::xaxpy(N, int2e.get(p,q,ka,kb), qt2ba.data(), opP.data());
            }
         }else{     
            if(qops.isym == 0 or qops.isym == 1){
               /*
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
                  for(int kq : krest){
                  int qa = 2*kq, qb = qa+1;
                  if(kp < kq){
                  qops('P')[oper_pack(pa,qa)] = int2e.get(pa,qa,ka,kb)*qt2ba; 
                  qops('P')[oper_pack(pa,qb)] = int2e.get(pa,qb,ka,kb)*qt2ba;
                  qops('P')[oper_pack(pb,qa)] = int2e.get(pb,qa,ka,kb)*qt2ba; 
                  qops('P')[oper_pack(pb,qb)] = int2e.get(pb,qb,ka,kb)*qt2ba;
                  }else if(kp == kq){                                          
                  qops('P')[oper_pack(pa,pb)] = int2e.get(pa,pb,ka,kb)*qt2ba;
                  }
                  } // kq
                  } // kp
                  */
               for(auto& pr : qops('P')){
                  int pq = pr.first;
                  auto& opP = pr.second;
                  auto upq = oper_unpack(pq);
                  int p = upq.first, q = upq.second; 
                  int N = opP.size();
                  linalg::xaxpy(N, int2e.get(p,q,ka,kb), qt2ba.data(), opP.data());
               }
            }else if(qops.isym == 2){
               /*
                  auto qphys = get_qbond_phys(isym);
                  stensor2<Tm> paa_zero(qsym(isym,-2,-2),qphys,qphys);
                  stensor2<Tm> pbb_zero(qsym(isym,-2, 2),qphys,qphys);
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
                  for(int kq : krest){
                  int qa = 2*kq, qb = qa+1;
                  if(kp < kq){
                  qops('P')[oper_pack(pa,qa)] = paa_zero; // because the integral <AA||AB> is zero
                  qops('P')[oper_pack(pa,qb)] = int2e.get(pa,qb,ka,kb)*qt2ba;
                  qops('P')[oper_pack(pb,qa)] = int2e.get(pb,qa,ka,kb)*qt2ba; 
                  qops('P')[oper_pack(pb,qb)] = pbb_zero;
                  }else if(kp == kq){
                  qops('P')[oper_pack(pa,pb)] = int2e.get(pa,pb,ka,kb)*qt2ba;
                  }
                  } // kq
                  } // kp
                  */
               for(auto& pr : qops('P')){
                  int pq = pr.first;
                  auto& opP = pr.second;
                  auto upq = oper_unpack(pq);
                  int p = upq.first, q = upq.second; 
                  int N = opP.size();
                  if(p%2 == q%2) continue;
                  linalg::xaxpy(N, int2e.get(p,q,ka,kb), qt2ba.data(), opP.data());
               }
            } // isym
         } // ifkr
      }

   // Qps = <pq||sr> aq^+ar
   template <typename Tm>
      void oper_dot_opQ(oper_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opQ" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         const auto& qt2aa = qops('B')[oper_pack(ka,ka)];
         const auto& qt2ab = qops('B')[oper_pack(ka,kb)];
         if(qops.ifkr){
            const auto& qt2ba = qt2ab.K(1); 
            const auto& qt2bb = qt2aa.K(2);
            /*
               for(int kp : krest){
               int pa = 2*kp, pb = pa+1;
               for(int ks : krest){
               int sa = 2*ks, sb = sa+1;
            // storage scheme for Qps: according to spatial orbital
            if(kp <= ks){
            // Q[pA,sA] = <pA,q||sA,r> B[q,r]
            // 	    = <pA,kA||sA,kA> B[kA,kA] + <pA,kB||sA,kB> B[kB,kB]
            //	    + <pA,kA||sA,kB> B[kA,kB] + <pA,kB||sA,kA> B[kB,kA] (zero in NR)
            qops('Q')[oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
            + int2e.get(pa,kb,sa,kb)*qt2bb
            + int2e.get(pa,ka,sa,kb)*qt2ab
            + int2e.get(pa,kb,sa,ka)*qt2ba;
            // Q[pA,sB] = <pA,q||sB,r> B[q,r]
            // 	    = <pA,kA||sB,kA> B[kA,kA] + <pA,kB||sB,kB> B[kB,kB] (zero in NR) 
            //	    + <pA,kA||sB,kB> B[kA,kB] + <pA,kB||sB,kA> B[kB,kA] 
            qops('Q')[oper_pack(pa,sb)] = int2e.get(pa,ka,sb,ka)*qt2aa
            + int2e.get(pa,kb,sb,kb)*qt2bb
            + int2e.get(pa,ka,sb,kb)*qt2ab
            + int2e.get(pa,kb,sb,ka)*qt2ba;
            }
            } // ks
            } // kp
            */
            for(auto& pr : qops('Q')){
               int ps = pr.first;
               auto& opQ = pr.second;
               auto ups = oper_unpack(ps);
               int p = ups.first, s = ups.second; 
               int N = opQ.size();
               linalg::xaxpy(N, int2e.get(p,ka,s,ka), qt2aa.data(), opQ.data());
               linalg::xaxpy(N, int2e.get(p,kb,s,kb), qt2bb.data(), opQ.data());
               linalg::xaxpy(N, int2e.get(p,ka,s,kb), qt2ab.data(), opQ.data());
               linalg::xaxpy(N, int2e.get(p,kb,s,ka), qt2ba.data(), opQ.data());
            }
         }else{
            const auto& qt2ba = qops('B')[oper_pack(ka,kb)].H();
            const auto& qt2bb = qops('B')[oper_pack(kb,kb)];
            if(qops.isym == 0 or qops.isym == 1){
               /*
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
                  for(int ks : krest){
                  int sa = 2*ks, sb = sa+1;
               // storage scheme for Qps: according to spatial orbital
               if(kp <= ks){
               qops('Q')[oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
               + int2e.get(pa,kb,sa,kb)*qt2bb
               + int2e.get(pa,ka,sa,kb)*qt2ab
               + int2e.get(pa,kb,sa,ka)*qt2ba;
               qops('Q')[oper_pack(pa,sb)] = int2e.get(pa,ka,sb,ka)*qt2aa
               + int2e.get(pa,kb,sb,kb)*qt2bb
               + int2e.get(pa,ka,sb,kb)*qt2ab
               + int2e.get(pa,kb,sb,ka)*qt2ba;
               if(kp != ks){ // recovered by Hermicity for kp=ks, see oper_combine.h 
               qops('Q')[oper_pack(pb,sa)] = int2e.get(pb,ka,sa,ka)*qt2aa 
               + int2e.get(pb,kb,sa,kb)*qt2bb
               + int2e.get(pb,ka,sa,kb)*qt2ab
               + int2e.get(pb,kb,sa,ka)*qt2ba;
               }
               qops('Q')[oper_pack(pb,sb)] = int2e.get(pb,ka,sb,ka)*qt2aa
               + int2e.get(pb,kb,sb,kb)*qt2bb
               + int2e.get(pb,ka,sb,kb)*qt2ab
               + int2e.get(pb,kb,sb,ka)*qt2ba;
               }
               } // ks
               } // kp
               */
               for(auto& pr : qops('Q')){
                  int ps = pr.first;
                  auto& opQ = pr.second;
                  auto ups = oper_unpack(ps);
                  int p = ups.first, s = ups.second; 
                  int N = opQ.size();
                  linalg::xaxpy(N, int2e.get(p,ka,s,ka), qt2aa.data(), opQ.data());
                  linalg::xaxpy(N, int2e.get(p,kb,s,kb), qt2bb.data(), opQ.data());
                  linalg::xaxpy(N, int2e.get(p,ka,s,kb), qt2ab.data(), opQ.data());
                  linalg::xaxpy(N, int2e.get(p,kb,s,ka), qt2ba.data(), opQ.data());
               }
            }else if(qops.isym == 2){
               /*
                  for(int kp : krest){
                  int pa = 2*kp, pb = pa+1;
                  for(int ks : krest){
                  int sa = 2*ks, sb = sa+1;
               // storage scheme for Qps: according to spatial orbital
               if(kp <= ks){
               // NOTE: zero terms need to be removed, otherwise qsym does not match!
               qops('Q')[oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
               + int2e.get(pa,kb,sa,kb)*qt2bb;
               qops('Q')[oper_pack(pa,sb)] = int2e.get(pa,kb,sb,ka)*qt2ba;
               if(kp != ks){ // recovered by Hermicity for kp=ks, see oper_combine.h 
               qops('Q')[oper_pack(pb,sa)] = int2e.get(pb,ka,sa,kb)*qt2ab;
               }
               qops('Q')[oper_pack(pb,sb)] = int2e.get(pb,ka,sb,ka)*qt2aa
               + int2e.get(pb,kb,sb,kb)*qt2bb;
               }
               } // ks
               } // kp
               */
               for(auto& pr : qops('Q')){
                  int ps = pr.first;
                  auto& opQ = pr.second;
                  auto ups = oper_unpack(ps);
                  int p = ups.first, s = ups.second;
                  int N = opQ.size();
                  if(p%2 == s%2){
                     linalg::xaxpy(N, int2e.get(p,ka,s,ka), qt2aa.data(), opQ.data());
                     linalg::xaxpy(N, int2e.get(p,kb,s,kb), qt2bb.data(), opQ.data());
                  }else if(p%2 == 0 && s%2 == 1){
                     linalg::xaxpy(N, int2e.get(p,kb,s,ka), qt2ba.data(), opQ.data());
                  }else if(p%2 == 1 && s%2 == 0){
                     linalg::xaxpy(N, int2e.get(p,ka,s,kb), qt2ab.data(), opQ.data());
                  }
               }
            } // isym
         } // ifkr
      }

   // --- for RDMs ---
   // Identity
   template <typename Tm>
      void oper_dot_opI(oper_dict<Tm>& qops){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opI" << std::endl; 
         // [[1. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 1. 0.]
         //  [0. 0. 0. 1.]]
         linalg::matrix<Tm> mat = linalg::identity_matrix<Tm>(4);
         qops('I')[0].from_matrix(mat);
      }

   // F = a^+b^+ba 
   template <typename Tm>
      void oper_dot_opF(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opF" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(1,1) = 1;
         qops('F')[ka].from_matrix(mat);
         if(debug_oper_dot) qops('F')[ka].to_matrix().print("c0+c1+c1c0");  
      }

   // T = {a^+ba,b^+ba} 
   template <typename Tm>
      void oper_dot_opT(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opT" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(a[1].dot(a[0]))
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 1. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(2,1) = 1;
         qops('T')[ka].from_matrix(mat); // a^+ba
         if(not qops.ifkr){
            // c[1].dot(a[1].dot(a[0]))
            // [[0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 0. 0. 0.]
            //  [0. 1. 0. 0.]]
            linalg::matrix<Tm> mat(4,4);
            mat(3,1) = 1;
            qops('T')[kb].from_matrix(mat); // b^+ba
         }
      }

   // for RDM: kA
   template <typename Tm>
      void oper_dot_opD(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opD" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // dagger of c[0] = kA^+
         // [[0. 0. 0. 0.]
         //  [0. 0. 0. 1.]
         //  [1. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(3,1) = 1;
         mat(0,2) = 1;
         qops('D')[ka].from_matrix(mat);
         if(debug_oper_dot) qops('D')[ka].to_matrix().print("c0");
         // dagger of c[1] = kB^+ 
         if(not qops.ifkr){
            // also store dagger of c[1] = kB^+
            // [[ 0.  0.  0.  0.]
            //  [ 0.  0. -1.  0.]
            //  [ 0.  0.  0.  0.]
            //  [ 1.  0.  0.  0.]]
            linalg::matrix<Tm> mat(4,4);
            mat(2,1) = -1;
            mat(0,3) = 1;
            qops('D')[kb].from_matrix(mat);
            if(debug_oper_dot) qops('D')[kb].to_matrix().print("c1");
         }
      }

   // M[kA,kB] = kA kB
   template <typename Tm>
      void oper_dot_opM(oper_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot) std::cout << "ctns::oper_dot_opM" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // dagger of -c[0].dot(c[1])
         // [[0. 0. 0. 0.]
         //  [1. 0. 0. 0.]
         //  [0. 0. 0. 0.]
         //  [0. 0. 0. 0.]]
         linalg::matrix<Tm> mat(4,4);
         mat(0,1) = -1;
         qops('M')[oper_pack(ka,kb)].from_matrix(mat);
         if(debug_oper_dot) qops('M')[oper_pack(ka,kb)].to_matrix().print("c0c1");
      }

} // ctns

#endif
