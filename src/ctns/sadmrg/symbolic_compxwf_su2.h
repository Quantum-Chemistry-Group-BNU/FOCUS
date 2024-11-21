#ifndef SYMBOLIC_COMPXWF_SU2_H
#define SYMBOLIC_COMPXWF_SU2_H

#include "../symbolic_task.h"
#include "../oper_partition.h"
#include "symbolic_compxwf_opS_su2.h"

namespace ctns{

   const bool debug_compxwf_su2 = true;
   extern const bool debug_compxwf_su2;

   //
   // opxwf = (sum_{ij} oij*a1^(d1)[i]*a2^(d2)[i])^d * wf
   //
   template <typename Tm>
      void symbolic_op1op2xwf_su2(symbolic_task<Tm>& formulae,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int ts,
            const std::map<int,Tm>& oij,
            const bool ifdagger1,
            const bool ifdagger2){
         if(cindex1.size() <= cindex2.size()){
            // sum_i a1[i] * (sum_j oij a2[j])
            for(const auto& i : cindex1){
               auto op1C = symbolic_oper(block1,'C',i);      
               auto op1 = ifdagger1? op1C : op1C.H(); // default is [a^+]^+ = a
               // top2 = sum_j oij a2[j]
               symbolic_sum<Tm> top2;
               for(const auto& j : cindex2){
                  auto op2C = symbolic_oper(block2,'C',j);		 
                  auto op2 = ifdagger2? op2C : op2C.H();
                  top2.sum(oij.at(oper_pack(i,j)), op2);
               } // j
               auto o12 = symbolic_prod(op1,top2);
               o12.ispins.push_back(std::make_tuple(1,1,ts));
               formulae.append(o12);
            } // i
         }else{
            // this part appears when the branch is larger 
            // sum_j (sum_i oij a1[i]) * a2[j]
            for(const auto& j : cindex2){
               auto op2C = symbolic_oper(block2,'C',j);
               auto op2 = ifdagger2? op2C : op2C.H();
               // tmp_op1 = sum_i oij a1[i]
               symbolic_sum<Tm> top1;
               for(const auto& i : cindex1){
                  auto op1C = symbolic_oper(block1,'C',i);
                  auto op1 = ifdagger1? op1C : op1C.H();
                  top1.sum(oij.at(oper_pack(i,j)), op1);
               } // i
               auto o12 = symbolic_prod(top1,op2);
               o12.ispins.push_back(std::make_tuple(1,1,ts));
               formulae.append(o12);
            } // j
         } // cindex1.size() <= cindex2.size() 
      }

   // kernel for computing renormalized P|ket> or P^+|ket> 
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opP_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index){
         symbolic_task<Tm> formulae; 
         auto pq = oper_unpack(index);
         int p = pq.first, q = pq.second;
         int sp = p%2, sq = q%2;
         int ts = (sp!=sq)? 0 : 2;
         // 
         // Ppq = 1/2<pq||sr> aras  (p<q)
         //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
         //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
         //     + <pq||s1r2> ar2*as1	  => -<pq||s1r2> as1*ar2
         //
         // 1. P1*I2
         auto P1pq = symbolic_prod<Tm>(symbolic_oper(block1,'P',index),
               symbolic_oper(block2,'I',0));
         P1pq.ispins.push_back(std::make_tuple(ts,0,ts));
         formulae.append(P1pq);
         // 2. I1*P2
         auto P2pq = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
               symbolic_oper(block2,'P',index));
         P2pq.ispins.push_back(std::make_tuple(0,ts,ts));
         formulae.append(P2pq);
         // 3. sum_{s1} sum_{r2} -<pq||s1r2> as1*ar2
         std::map<int,Tm> oij;
         for(const auto& s1a : cindex1){
            int s1b = s1a+1;
            for(const auto& r2a : cindex2){
               int r2b = r2a+1;
               if(ts == 0){
                  int pa = p, qb = q;
                  oij[oper_pack(s1a,r2a)] =  int2e.get(pa,qb,s1a,r2b) + int2e.get(pa,qb,r2a,s1b);
               }else{
                  int pa = p, qa = q, qb = qa+1;
                  oij[oper_pack(s1a,r2a)] = -int2e.get(pa,qb,s1a,r2b) + int2e.get(pa,qb,r2a,s1b);
               }
            }
         }
         symbolic_op1op2xwf_su2<Tm>(formulae,block1,block2,cindex1,cindex2,
               ts,oij,0,0); // as1*ar2
         return formulae;
      }

   // kernel for computing renormalized Q|ket> or Q^+|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opQ_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index){
         symbolic_task<Tm> formulae; 
         auto ps = oper_unpack(index);
         int p = ps.first, s = ps.second;
         int sp = p%2, ss = s%2;
         int ts = (sp!=ss)? 2 : 0;
         //
         // Qps = <pq||sr> aq^+ar
         //     = <pq1||sr1> Bq1r1 	=> Qps^1
         // 	 + <pq2||sr2> Bq2r2 	=> Qps^2
         //     + <pq1||sr2> aq1^+ar2 => <pq1||sr2> aq1^+*ar2 
         //     + <pq2||sr1> aq2^+ar1 => -<pq2||sr1> ar1*aq2^+
         //
         // 1. Q1*I2
         auto Q1ps = symbolic_prod<Tm>(symbolic_oper(block1,'Q',index),
               symbolic_oper(block2,'I',0));
         Q1ps.ispins.push_back(std::make_tuple(ts,0,ts));
         formulae.append(Q1ps);
         // 2. I1*Q2
         auto Q2ps = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
               symbolic_oper(block2,'Q',index));
         Q2ps.ispins.push_back(std::make_tuple(0,ts,ts));
         formulae.append(Q2ps);
         // 3. <pq1||sr2> aq1^+*ar2 &  4. -<pr2||sq1> aq1*ar2^+
         std::map<int,Tm> o1ij, o2ij;
         for(const auto& q1a : cindex1){
            int q1b = q1a+1;
            for(const auto& r2a : cindex2){
               int r2b = r2a+1;
               if(ts == 0){
                  int pa = p, sa = s;
                  o1ij[oper_pack(q1a,r2a)] =  int2e.get(pa,q1a,sa,r2a) + int2e.get(pa,q1b,sa,r2b);
                  o2ij[oper_pack(q1a,r2a)] =  int2e.get(pa,r2a,sa,q1a) + int2e.get(pa,r2b,sa,q1b); // (-1)^0=1
               }else{
                  int pa = p, sb = s;
                  o1ij[oper_pack(q1a,r2a)] =  int2e.get(pa,q1b,sb,r2a);
                  o2ij[oper_pack(q1a,r2a)] = -int2e.get(pa,r2b,sb,q1a);
               }
            }
         }
         symbolic_op1op2xwf_su2<Tm>(formulae,block1,block2,cindex1,cindex2,
               ts,o1ij,1,0); // aq1^+*ar2
         symbolic_op1op2xwf_su2<Tm>(formulae,block1,block2,cindex1,cindex2,
               ts,o2ij,0,1); // aq1*ar2^+
         return formulae;
      }

   // kernel for computing renormalized Sp|ket> [6 terms]
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opS_su2(const std::string oplist1,
            const std::string oplist2, 
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index,
            const bool ifkr,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifdistc){
         assert(ifkr == true);
         symbolic_task<Tm> formulae;
         int p = index, kp = p/2;
         //
         // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
         //    = Sp^1 + Sp^2 (S exists in both blocks)
         //    + <pq1||s2r2> aq[1]^+ar[2]as[2] 
         //    + <pq2||s1r2> aq[2]^+ar[2]as[1] 
         //    + <pq2||s1r1> aq[2]^+ar[1]as[1] 
         //    + <pq1||s1r2> aq[1]^+ar[2]as[1] 
         //
         int iproc = distribute1(ifkr,size,p);
         if(!ifdist1 or iproc==rank){
            // 1. S1*I2
            auto S1p = symbolic_prod<Tm>(symbolic_oper(block1,'S',index),
                  symbolic_oper(block2,'I',0));
            S1p.ispins.push_back(std::make_tuple(1,0,1)); 
            formulae.append(S1p);
            // 2. I1*S2
            auto S2p = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'S',index));
            S2p.ispins.push_back(std::make_tuple(0,1,1));
            formulae.append(S2p);
         }

         // cross terms [revised count the no. of contractions]
         int k1 = cindex1.size(), kc1 = 2*k1, kA1 = k1*k1, kB1 = 2*kA1;
         int k2 = cindex2.size(), kc2 = 2*k2, kA2 = k2*k2, kB2 = 2*kA2;

         // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]	   
         int formula3 = -1;
         bool exist2A = ifexistQ(oplist2,'A');
         bool exist2P = ifexistQ(oplist2,'P');
         bool outer3s = kc1<=kA2; // outer sum is single index
         if(exist2P and (!exist2A or (exist2A and outer3s))){
            formula3 = 0;
         }else if(exist2A and !outer3s){
            formula3 = 1;
         }else if(exist2A and !exist2P and outer3s){
            formula3 = 2;
         }else{
            tools::exit("error: no such case for opS3");
         }  
         auto size3 = formulae.size();
         if(formula3 == 0){
            symbolic_compxwf_opS3a_su2(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula3 == 1){
            // In the case of MPS with configuration lc, this formula can be calculated in two ways,
            // since the centeral operators are replicated in all processors. In the general CTNS case,
            // where ifdistc=false, central operators are also storaged distributedly, then only the 
            // second branch is correct.
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+ 
                  auto aindex2 = oper_index_opA(cindex2, ifkr);
                  assert(aindex2.size() == 1);
                  symbolic_compxwf_opS3b_su2(block1, block2, cindex1, cindex2, int2e, p, 
                        aindex2, formulae);
               }
            }else{
               // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
               auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS3b_su2(block1, block2, cindex1, cindex2, int2e, p,
                     aindex2_dist, formulae);
            }
         }else if(formula3 == 2){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  auto aindex2 = oper_index_opA(cindex2, ifkr);
                  assert(aindex2.size() == 1);
                  symbolic_compxwf_opS3c_su2(block1, block2, cindex1, cindex2, int2e, p, 
                        aindex2, formulae);
               }
            }else{
               auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS3c_su2(block1, block2, cindex1, cindex2, int2e, p,
                     aindex2_dist, formulae);
            }
         }
         if(debug_compxwf_su2){
            size3 = formulae.size()-size3;
            std::cout << "formula3=" << formula3 << " size=" << size3 
               << " exist2A,2P=" << exist2A << "," << exist2P
               << " outer3s=" << outer3s
               << std::endl;
         }
 
         // 4. <pq2||s1r2> aq[2]^+ar[2]as[1]    
         int formula4 = -1;
         bool exist2B = ifexistQ(oplist2,'B');
         bool exist2Q = ifexistQ(oplist2,'Q');
         bool outer4s = kc1<=kB2;
         if(exist2Q and (!exist2B or (exist2B and outer4s))){
            formula4 = 0;
         }else if(exist2B and !outer4s){
            formula4 = 1;
         }else if(exist2B and !exist2Q and outer4s){
            formula4 = 2;
         }else{
            tools::exit("error: no such case for opS4");
         }
         auto size4 = formulae.size(); 
         if(formula4 == 0){
            // sum_q aq[1]*Qpq[2]
            symbolic_compxwf_opS4a_su2(block1, block2, cindex1, cindex2, p, ifkr,
                  int2e.sorb, size, rank, formulae);
         }else if(formula4 == 1){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
                  auto bindex2 = oper_index_opB(cindex2, ifkr);
                  symbolic_compxwf_opS4b_su2(block1, block2, cindex1, cindex2, int2e, p,
                        bindex2, formulae);
               }
            }else{
               auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS4b_su2(block1, block2, cindex1, cindex2, int2e, p, 
                        bindex2_dist, formulae);
            }
         }else if(formula4 == 2){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
                  auto bindex2 = oper_index_opB(cindex2, ifkr);
                  symbolic_compxwf_opS4c_su2(block1, block2, cindex1, cindex2, int2e, p,
                        bindex2, formulae);
               }
            }else{
               auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS4c_su2(block1, block2, cindex1, cindex2, int2e, p, 
                        bindex2_dist, formulae);
            }
         }
         if(debug_compxwf_su2){
            size4 = formulae.size()-size4;
            std::cout << "formula4=" << formula4 << " size=" << size 
               << " exist2B,2Q=" << exist2B << "," << exist2Q
               << " outer4s=" << outer4s
               << std::endl;
         }

         // 5. <pq2||s1r1> aq[2]^+ar[1]as[1]
         int formula5 = -1;
         bool exist1A = ifexistQ(oplist1,'A');
         bool exist1P = ifexistQ(oplist1,'P');
         bool outer5s = kc2<=kA1;
         if(exist1P and (!exist1A or (exist1A and outer5s))){
            formula5 = 0;
         }else if(exist1A and !outer5s){
            formula5 = 1;
         }else if(exist1A and !exist1P and outer5s){
            formula5 = 2;
         }else{
            tools::exit("error: no such case for op5");
         }
         auto size5 = formulae.size();
         if(formula5 == 0){
            // sum_q Ppq[1]*aq^+[2]
            symbolic_compxwf_opS5a_su2(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula5 == 1){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
                  auto aindex1 = oper_index_opA(cindex1, ifkr);
                  assert(aindex1.size() == 1);
                  symbolic_compxwf_opS5b_su2(block1, block2, cindex1, cindex2, int2e, p,
                        aindex1, formulae);
               }
            }else{
               auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS5b_su2(block1, block2, cindex1, cindex2, int2e, p,
                     aindex1_dist, formulae);
            }
         }else if(formula5 == 2){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  auto aindex1 = oper_index_opA(cindex1, ifkr);
                  assert(aindex1.size() == 1);
                  symbolic_compxwf_opS5c_su2(block1, block2, cindex1, cindex2, int2e, p,
                        aindex1, formulae);
               }
            }else{
               auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS5c_su2(block1, block2, cindex1, cindex2, int2e, p,
                     aindex1_dist, formulae);
            }
         }
         if(debug_compxwf_su2){
            size5 = formulae.size()-size5;
            std::cout << "formula5=" << formula5 << " size=" << size5 
               << " exist1A,1P=" << exist1A << "," << exist1P
               << " outer5s=" << outer5s
               << std::endl;
         }

         // 6. <pq1||s1r2> aq[1]^+ar[2]as[1]  
         int formula6 = -1;
         bool exist1B = ifexistQ(oplist1,'B');
         bool exist1Q = ifexistQ(oplist1,'Q');
         bool outer6s = kc2<=kB1;
         if(exist1Q and (!exist1B or (exist1B and outer6s))){
            formula6 = 0;
         }else if(exist1B and !outer6s){
            formula6 = 1;
         }else if(exist1B and !exist1Q and outer6s){
            formula6 = 2;
         }else{
            tools::exit("error: no such case for opS6");
         }
         auto size6 = formulae.size();
         if(formula6 == 0){
            // sum_q Qpq^[1]*aq[2]
            symbolic_compxwf_opS6a_su2(block1, block2, cindex1, cindex2, p, ifkr,
                  int2e.sorb, size, rank, formulae);
         }else if(formula6 == 1){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
                  auto bindex1 = oper_index_opB(cindex1, ifkr);
                  symbolic_compxwf_opS6b_su2(block1, block2, cindex1, cindex2, int2e, p,
                        bindex1, formulae);
               }
            }else{
               auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS6b_su2(block1, block2, cindex1, cindex2, int2e, p,
                     bindex1_dist, formulae);
            }
         }else if(formula6 == 2){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
                  auto bindex1 = oper_index_opB(cindex1, ifkr);
                  symbolic_compxwf_opS6c_su2(block1, block2, cindex1, cindex2, int2e, p,
                        bindex1, formulae);
               }
            }else{
               auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, size, rank, int2e.sorb);
               symbolic_compxwf_opS6c_su2(block1, block2, cindex1, cindex2, int2e, p,
                     bindex1_dist, formulae);
            }
         }
         if(debug_compxwf_su2){
            size6 = formulae.size()-size6;
            std::cout << "formula6=" << formula6 << " size=" << size6 
               << " exist1B,1Q=" << exist1B << "," << exist1Q
               << " outer6s=" << outer6s
               << std::endl;
         }
         return formulae;
      }

   // kernel for computing renormalized H|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opH_su2(const std::string oplist1,
            const std::string oplist2,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifdistc){
         assert(ifkr == true);
         symbolic_task<Tm> formulae;
         //
         // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
         //   = H1 + H2
         //   + p1^+*Sp1^2 + h.c.
         //   + q2^+*Sq2^1 + h.c.
         //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
         //   + <p1q2||s1r2> p1^+q2^+r2s1 
         //
         const double scale = 0.5; 
         if(!ifdist1 or rank==0){ 
            // 1. H1*I2
            auto H1 = symbolic_prod<Tm>(symbolic_oper(block1,'H',0), scale);
            auto I2 = symbolic_prod<Tm>(symbolic_oper(block2,'I',0));
            auto H1_I2 = H1.product(I2);
            H1_I2.ispins.push_back(std::make_tuple(0,0,0));
            formulae.append(H1_I2);
            // 2. I1*H2
            auto I1 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0));
            auto H2 = symbolic_prod<Tm>(symbolic_oper(block2,'H',0), scale);
            auto I1_H2 = I1.product(H2);
            I1_H2.ispins.push_back(std::make_tuple(0,0,0));
            formulae.append(I1_H2);
         }
         // One-index operators
         // 3. sum_p1 p1^+ Sp1^2 + h.c. 
         for(const auto& p1 : cindex1){
            int iproc = distribute1(ifkr,size,p1);
            if(!ifdist1 or iproc==rank){
               auto op1C = symbolic_oper(block1,'C',p1);
               auto op2S = symbolic_oper(block2,'S',p1);
               auto C1S2 = symbolic_prod<Tm>(op1C, op2S, std::sqrt(2.0)); // su2
               C1S2.ispins.push_back(std::make_tuple(1,1,0));
               formulae.append(C1S2);
            }
         }
         // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
         for(const auto& q2 : cindex2){
            int iproc = distribute1(ifkr,size,q2);
            if(!ifdist1 or iproc==rank){
               auto op1S = symbolic_oper(block1,'S',q2);
               auto op2C = symbolic_oper(block2,'C',q2);
               auto S1C2 = symbolic_prod<Tm>(op1S, op2C, std::sqrt(2.0)); // in su2 case, the sign is +1
               S1C2.ispins.push_back(std::make_tuple(1,1,0));
               formulae.append(S1C2);
            }
         }
         // Two-index operators
         int formula = 0;
         if(ifdistc and block1[0]=='l' and block2[0]=='c' and !ifexistQ(oplist1,'P')){ // lc 
            formula = 1; // AL*AC
         }else if(ifdistc and block1[0]=='c' and block2[0]=='r' and !ifexistQ(oplist2,'P')){ // cr
            formula = 2; // AC*AR
         }else{
            formula = 0; // default AP or PA
         }
         size_t tsize = formulae.size();
         if(formula == 0){
            // general case 
            const bool ifNC = determine_NCorCN_Ham(oplist1, oplist2, cindex1.size(), cindex2.size());
            auto AP1 = ifNC? 'A' : 'P';
            auto AP2 = ifNC? 'P' : 'A';
            auto BQ1 = ifNC? 'B' : 'Q';
            auto BQ2 = ifNC? 'Q' : 'B';
            const auto& cindex = ifNC? cindex1 : cindex2;
            auto aindex_dist = oper_index_opA_dist(cindex, ifkr, size, rank, sorb);
            auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, sorb);
            // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
            for(const auto& index : aindex_dist){
               auto pr = oper_unpack(index);
               int i = pr.first, ki = i/2, spin_i = i%2;
               int j = pr.second, kj = j/2, spin_j = j%2;
               int ts = (spin_i!=spin_j)? 0 : 2;
               double wfac = (ki==kj)? 0.5 : 1.0;
               const double wt = (ts==0)? -wfac : std::sqrt(3.0);
               auto op1 = symbolic_oper(block1,AP1,index);
               auto op2 = symbolic_oper(block2,AP2,index);
               auto AP = symbolic_prod<Tm>(op1, op2, wt);
               AP.ispins.push_back(std::make_tuple(ts,ts,0));
               formulae.append(AP);
            }
            // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
            for(const auto& index : bindex_dist){
               auto pr = oper_unpack(index);
               int i = pr.first, ki = i/2, spin_i = i%2;
               int j = pr.second, kj = j/2, spin_j = j%2;
               int ts = (spin_i!=spin_j)? 2 : 0;
               double wfac = (ki==kj)? 0.5 : 1.0;
               const double wt = ((ts==0)? 1.0 : -std::sqrt(3.0))*wfac;
               auto op1 = symbolic_oper(block1,BQ1,index);
               auto op2 = symbolic_oper(block2,BQ2,index);
               auto BQ = symbolic_prod<Tm>(op1, op2, wt);
               BQ.ispins.push_back(std::make_tuple(ts,ts,0));
               formulae.append(BQ);
            }
         }else if(formula == 1){ // AL*PC = {AL*x}*AC[all are available for dot!]
            auto aindex2 = oper_index_opA(cindex2, ifkr);
            auto bindex2 = oper_index_opB(cindex2, ifkr);
            auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, size, rank, sorb);
            auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, size, rank, sorb);
            for(const auto& isr : aindex2){
               auto sr = oper_unpack(isr); // sr [Abar_sr]^0
               int s2 = sr.first, ks = s2/2, spin_s = s2%2;
               int r2 = sr.second, kr = s2/2, spin_r = r2%2; 
               int ts = (spin_s!=spin_r)? 0 : 2;
               auto op2 = symbolic_oper(block2,'A',isr).H();
               symbolic_sum<Tm> top1;
               for(const auto& ipq : aindex1_dist){ 
                  auto pq = oper_unpack(ipq);
                  int p2 = pq.first, kp = p2/2, spin_p = p2%2;
                  int q2 = pq.second, kq = q2/2, spin_q = q2%2;
                  int ts1 = (spin_p!=spin_q)? 0 : 2;
                  if(ts1 != ts) continue;
                  auto op1 = symbolic_oper(block1,'A',ipq);
                  double wpq = (kp==kq)? 0.5 : 1.0;
                  const double wt = (ts==0)? -wpq: std::sqrt(3.0);
                  top1.sum(wt*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op1);
               }
               if(top1.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1, op2);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
            }
            for(const auto& iqr : bindex2){
               auto qr = oper_unpack(iqr); 
               int q2 = qr.first, kq = q2/2, spin_q = q2%2;
               int r2 = qr.second, kr = r2/2, spin_r = r2%2;
               int ts = (spin_q!=spin_r)? 2 : 0;
               double wqr = (kq==kr)? 0.5 : 1.0;
               const double wt = ((ts==0)? 1.0 : -std::sqrt(3.0))*wqr;
               auto op2 = symbolic_oper(block2,'B',iqr); 
               auto op2H = symbolic_oper(block2,'B',iqr).H(); 
               symbolic_sum<Tm> top1, top1H;
               for(const auto& ips : bindex1_dist){ 
                  auto ps = oper_unpack(ips);
                  int p2 = ps.first, kp = p2/2, spin_p = p2%2;
                  int s2 = ps.second, ks = s2/2, spin_s = s2%2;
                  int ts1 = (spin_p!=spin_s)? 2 : 0;
                  if(ts1 != ts) continue;
                  auto op1 = symbolic_oper(block1,'B',ips);
                  double wps = (kp==ks)? 0.5 : 1.0;
                  Tm fac1 = wps*wt;
                  Tm fac2 = fac1;
                  if(ts==2) fac2 = -fac1;
                  top1.sum(fac1*get_vint2e_su2(int2e,ts,kp,kq,ks,kr), op1);
                  top1H.sum(fac2*get_vint2e_su2(int2e,ts,kp,kr,ks,kq), op1);
               }
               if(top1.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1, op2);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
               if(top1H.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1H, op2H);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
            }
         }else if(formula == 2){ // PC*AR = AC*{x*AR}
            auto aindex1 = oper_index_opA(cindex1, ifkr);
            auto bindex1 = oper_index_opB(cindex1, ifkr);
            auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, size, rank, sorb);
            auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, size, rank, sorb);
            for(const auto& isr : aindex1){
               auto sr = oper_unpack(isr); // sr [Abar_sr]^0
               int s2 = sr.first, ks = s2/2, spin_s = s2%2;
               int r2 = sr.second, kr = s2/2, spin_r = r2%2; 
               int ts = (spin_s!=spin_r)? 0 : 2;
               auto op1 = symbolic_oper(block1,'A',isr).H();
               symbolic_sum<Tm> top2;
               for(const auto& ipq : aindex2_dist){ 
                  auto pq = oper_unpack(ipq);
                  int p2 = pq.first, kp = p2/2, spin_p = p2%2;
                  int q2 = pq.second, kq = q2/2, spin_q = q2%2;
                  int ts1 = (spin_p!=spin_q)? 0 : 2;
                  if(ts1 != ts) continue;
                  auto op2 = symbolic_oper(block2,'A',ipq);
                  double wpq = (kp==kq)? 0.5 : 1.0;
                  const double wt = (ts==0)? -wpq: std::sqrt(3.0);
                  top2.sum(wt*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
               }
               if(top2.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1, top2);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
            }
            for(const auto& iqr : bindex1){
               auto qr = oper_unpack(iqr); 
               int q2 = qr.first, kq = q2/2, spin_q = q2%2;
               int r2 = qr.second, kr = r2/2, spin_r = r2%2;
               int ts = (spin_q!=spin_r)? 2 : 0;
               double wqr = (kq==kr)? 0.5 : 1.0;
               const double wt = ((ts==0)? 1.0 : -std::sqrt(3.0))*wqr;
               auto op1 = symbolic_oper(block1,'B',iqr); 
               auto op1H = symbolic_oper(block1,'B',iqr).H(); 
               symbolic_sum<Tm> top2, top2H;
               for(const auto& ips : bindex2_dist){ 
                  auto ps = oper_unpack(ips);
                  int p2 = ps.first, kp = p2/2, spin_p = p2%2;
                  int s2 = ps.second, ks = s2/2, spin_s = s2%2;
                  int ts1 = (spin_p!=spin_s)? 2 : 0;
                  if(ts1 != ts) continue;
                  auto op2 = symbolic_oper(block2,'B',ips);
                  double wps = (kp==ks)? 0.5 : 1.0;
                  Tm fac1 = wps*wt;
                  Tm fac2 = fac1;
                  if(ts==2) fac2 = -fac1;
                  top2.sum(fac1*get_vint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
                  top2H.sum(fac2*get_vint2e_su2(int2e,ts,kp,kr,ks,kq), op2);
               }
               if(top2.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1, top2);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
               if(top2H.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1H, top2H);
                  op12.ispins.push_back(std::make_tuple(ts,ts,0));
                  formulae.append(op12);
               }
            }
         } 
         if(debug_compxwf_su2){
            tsize = formulae.size()-tsize;
            std::cout << "formula(two-index)=" << formula << " size=" << tsize 
               << " ifdistc=" << ifdistc << " block1=" << block1 << " block2=" << block2
               << std::endl;
         }
         return formulae;
      }

} // ctns

#endif
