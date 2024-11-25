#ifndef SYMBOLIC_COMPXWF_H
#define SYMBOLIC_COMPXWF_H

#include "oper_partition.h"
#include "symbolic_task.h"
#include "symbolic_op1op2xwf.h"
#include "symbolic_compxwf_opS.h"

namespace ctns{

   const bool debug_compxwf = false;
   extern const bool debug_compxwf;

   // kernel for computing renormalized P|ket> or P^+|ket> 
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opP(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index,
            const int isym,
            const bool ifkr){
         symbolic_task<Tm> formulae; 
         auto pq = oper_unpack(index);
         int p = pq.first, q = pq.second;
         auto sym_op = get_qsym_opP(isym, p, q);
         // 
         // Ppq = 1/2<pq||sr> aras  (p<q)
         //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
         //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
         //     + <pq||s1r2> ar2*as1	  => -<pq||s1r2> as1*ar2
         //
         // 1. P1*I2
         auto P1pq = symbolic_prod<Tm>(symbolic_oper(block1,'P',index),
               symbolic_oper(block2,'I',0));
         formulae.append(P1pq);
         // 2. I1*P2
         auto P2pq = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
               symbolic_oper(block2,'P',index));
         formulae.append(P2pq);
         // 3. sum_{s1} sum_{r2} -<pq||s1r2> as1*ar2
         std::map<int,Tm> oij;
         if(!ifkr){
            for(const auto& s1 : cindex1){
               for(const auto& r2 : cindex2){
                  oij[oper_pack(s1,r2)] = -int2e.get(p,q,s1,r2);
               }
            }
         }else{
            for(const auto& s1a : cindex1){
               int s1b = s1a+1;
               for(const auto& r2a : cindex2){
                  int r2b = r2a+1;
                  oij[oper_pack(s1a,r2a)] = -int2e.get(p,q,s1a,r2a);
                  oij[oper_pack(s1a,r2b)] = -int2e.get(p,q,s1a,r2b);
                  oij[oper_pack(s1b,r2a)] = -int2e.get(p,q,s1b,r2a);
                  oij[oper_pack(s1b,r2b)] = -int2e.get(p,q,s1b,r2b);
               }
            }
         }
         symbolic_op1op2xwf<Tm>(ifkr,formulae,block1,block2,cindex1,cindex2,
               isym,sym_op,oij,0,0,0); // as1*ar2
         return formulae;
      }

   // kernel for computing renormalized Q|ket> or Q^+|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opQ(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index,
            const int isym,
            const bool ifkr){
         symbolic_task<Tm> formulae; 
         auto ps = oper_unpack(index);
         int p = ps.first, s = ps.second;
         auto sym_op = get_qsym_opQ(isym, p, s);
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
         formulae.append(Q1ps);
         // 2. I1*Q2
         auto Q2ps = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
               symbolic_oper(block2,'Q',index));
         formulae.append(Q2ps);
         // 3. <pq1||sr2> aq1^+*ar2 &  4. -<pr2||sq1> aq1*ar2^+
         std::map<int,Tm> o1ij, o2ij;
         if(!ifkr){
            for(const auto& q1 : cindex1){
               for(const auto& r2 : cindex2){
                  o1ij[oper_pack(q1,r2)] =  int2e.get(p,q1,s,r2);
                  o2ij[oper_pack(q1,r2)] = -int2e.get(p,r2,s,q1);
               }
            }
         }else{
            for(const auto& q1a : cindex1){
               int q1b = q1a+1;
               for(const auto& r2a : cindex2){
                  int r2b = r2a+1;
                  o1ij[oper_pack(q1a,r2a)] =  int2e.get(p,q1a,s,r2a);
                  o1ij[oper_pack(q1a,r2b)] =  int2e.get(p,q1a,s,r2b);
                  o1ij[oper_pack(q1b,r2a)] =  int2e.get(p,q1b,s,r2a);
                  o1ij[oper_pack(q1b,r2b)] =  int2e.get(p,q1b,s,r2b);
                  o2ij[oper_pack(q1a,r2a)] = -int2e.get(p,r2a,s,q1a);
                  o2ij[oper_pack(q1a,r2b)] = -int2e.get(p,r2b,s,q1a);
                  o2ij[oper_pack(q1b,r2a)] = -int2e.get(p,r2a,s,q1b);
                  o2ij[oper_pack(q1b,r2b)] = -int2e.get(p,r2b,s,q1b);
               }
            }
         }
         symbolic_op1op2xwf<Tm>(ifkr,formulae,block1,block2,cindex1,cindex2,
               isym,sym_op,o1ij,1,0,0); // aq1^+*ar2
         symbolic_op1op2xwf<Tm>(ifkr,formulae,block1,block2,cindex1,cindex2,
               isym,sym_op,o2ij,0,1,0); // aq1*ar2^+
         return formulae;
      }

   // kernel for computing renormalized Sp|ket> [6 terms]
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opS(const std::string oplist1,
            const std::string oplist2,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int index,
            const int isym,
            const bool ifkr,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifdistc){
         symbolic_task<Tm> formulae;
         int p = index, kp = p/2;
         auto sym_op = get_qsym_opS(isym, p);
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
            formulae.append(S1p);
            // 2. I1*S2
            auto S2p = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'S',index));
            formulae.append(S2p);
         }

         // cross terms
         int kc1 = ifkr? 2*cindex1.size() : cindex1.size();
         int kA1 = kc1*(kc1-1)/2;
         int kB1 = kc1*kc1;
         int kc2 = ifkr? 2*cindex2.size() : cindex2.size();
         int kA2 = kc2*(kc2-1)/2;
         int kB2 = kc2*kc2;
         
         // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]	(see eq109 in R_CTNS.pdf) 
         int formula3 = get_formula_opS3(oplist2, kc1, kA2);
         auto size3 = formulae.size();
         if(formula3 == 0){ 
            symbolic_compxwf_opS3a(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula3 == 1){ 
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
                  auto aindex2 = oper_index_opA(cindex2, ifkr, isym);
                  symbolic_compxwf_opS3b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        aindex2, formulae);
               }
            }else{
               auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS3b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     aindex2_dist, formulae);
            }
         }else if(formula3 == 2){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_q aq[1]^+ (sum_sr <pq1||s2r2> Asr[2]^+)
                  auto aindex2 = oper_index_opA(cindex2, ifkr, isym);
                  symbolic_compxwf_opS3c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        aindex2, formulae);
               }
            }else{
               auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS3c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     aindex2_dist, formulae);
            }
         }
         if(debug_compxwf){
            size3 = formulae.size()-size3;
            std::cout << "formula3=" << formula3 << " size=" << size3 << std::endl;
         }

         // 4. <pq2||s1r2> aq[2]^+ar[2]as[1] (see eq110 in R_CTNS.pdf)
         int formula4 = get_formula_opS4(oplist2, kc1, kB2);
         auto size4 = formulae.size(); 
         if(formula4 == 0){
            // sum_q aq[1]*Qpq[2]
            symbolic_compxwf_opS4a(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula4 == 1){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
                  auto bindex2 = oper_index_opB(cindex2, ifkr, isym);
                  symbolic_compxwf_opS4b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        bindex2, formulae);
               }
            }else{ 
               auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS4b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     bindex2_dist, formulae);
            }
         }else if(formula4 == 2){
            if(ifdistc && block2[0]=='c'){ // lc
               if(iproc == rank){
                  // sum_s as[1] (sum_qr <pq2||s1r2> aq[2]^+ar[2])
                  auto bindex2 = oper_index_opB(cindex2, ifkr, isym);
                  symbolic_compxwf_opS4c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        bindex2, formulae);
               }
            }else{ 
               auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS4c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     bindex2_dist, formulae);
            }
         }
         if(debug_compxwf){
            size4 = formulae.size()-size4;
            std::cout << "formula4=" << formula4 << " size=" << size << std::endl;
         }

         // 5. <pq2||s1r1> aq[2]^+ar[1]as[1] (see eq111 in R_CTNS.pdf)
         int formula5 = get_formula_opS5(oplist1, kc2, kA1);
         auto size5 = formulae.size();
         if(formula5 == 0){
            // sum_q Ppq[1]*aq^+[2]
            symbolic_compxwf_opS5a(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula5 == 1){ 
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
                  auto aindex1 = oper_index_opA(cindex1, ifkr, isym);
                  symbolic_compxwf_opS5b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        aindex1, formulae);
               }
            }else{ 
               auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS5b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     aindex1_dist, formulae);
            }
         }else if(formula5 == 2){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_q (sum_sr Asr[1]^+ <pq2||s1r1>) aq[2]^+
                  auto aindex1 = oper_index_opA(cindex1, ifkr, isym);
                  symbolic_compxwf_opS5c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        aindex1, formulae);
               }
            }else{ 
               auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS5c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     aindex1_dist, formulae);
            }
         }
         if(debug_compxwf){
            size5 = formulae.size()-size5;
            std::cout << "formula5=" << formula5 << " size=" << size5 << std::endl;
         }

         // 6. <pq1||s1r2> aq[1]^+ar[2]as[1] (see eq112 in R_CTNS.pdf) 
         int formula6 = get_formula_opS6(oplist1, kc2, kB1);
         auto size6 = formulae.size();
         if(formula6 == 0){
            // sum_q Qpq^[1]*aq[2]
            symbolic_compxwf_opS6a(block1, block2, cindex1, cindex2, p, ifkr, 
                  int2e.sorb, size, rank, formulae);
         }else if(formula6 == 1){ 
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
                  auto bindex1 = oper_index_opB(cindex1, ifkr, isym);
                  symbolic_compxwf_opS6b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        bindex1, formulae);
               }
            }else{
               auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS6b(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     bindex1_dist, formulae);
            }
         }else if(formula6 == 2){
            if(ifdistc && block1[0]=='c'){ // cr
               if(iproc == rank){
                  // sum_r (sum_qs aq[1]^+as[1] -<pq1||s1r2>) ar[2]
                  auto bindex1 = oper_index_opB(cindex1, ifkr, isym);
                  symbolic_compxwf_opS6c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                        bindex1, formulae);
               }
            }else{
               auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, isym, size, rank, int2e.sorb);
               symbolic_compxwf_opS6c(block1, block2, cindex1, cindex2, int2e, p, isym, ifkr, 
                     bindex1_dist, formulae);
            }
         }
         if(debug_compxwf){
            size6 = formulae.size()-size6;
            std::cout << "formula6=" << formula6 << " size=" << size6 << std::endl;
         }
         return formulae;
      }

   // kernel for computing renormalized H|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_compxwf_opH(const std::string oplist1,
            const std::string oplist2,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int isym,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifdistc){
         symbolic_task<Tm> formulae;
         //
         // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
         //   = H1 + H2
         //   + p1^+*Sp1^2 + h.c.
         //   + q2^+*Sq2^1 + h.c.
         //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
         //   + <p1q2||s1r2> p1^+q2^+r2s1 
         //
         const double scale = ifkr? 0.25 : 0.5;
         if(!ifdist1 or rank==0){ 
            // 1. H1*I2
            auto H1 = symbolic_prod<Tm>(symbolic_oper(block1,'H',0), scale);
            auto I2 = symbolic_prod<Tm>(symbolic_oper(block2,'I',0));
            auto H1_I2 = H1.product(I2);
            formulae.append(H1_I2);
            // 2. I1*H2
            auto I1 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0));
            auto H2 = symbolic_prod<Tm>(symbolic_oper(block2,'H',0), scale); 
            auto I1_H2 = I1.product(H2);
            formulae.append(I1_H2);
         }
         // One-index operators
         // 3. sum_p1 p1^+ Sp1^2 + h.c. 
         for(const auto& p1 : cindex1){
            int iproc = distribute1(ifkr,size,p1);
            if(!ifdist1 or iproc==rank){
               auto op1C = symbolic_oper(block1,'C',p1);
               auto op2S = symbolic_oper(block2,'S',p1);
               auto C1S2 = symbolic_prod<Tm>(op1C, op2S);
               formulae.append(C1S2);
            }
         }
         // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
         for(const auto& q2 : cindex2){
            int iproc = distribute1(ifkr,size,q2);
            if(!ifdist1 or iproc==rank){
               auto op1S = symbolic_oper(block1,'S',q2);
               auto op2C = symbolic_oper(block2,'C',q2);
               auto S1C2 = symbolic_prod<Tm>(op1S, op2C, -1.0);
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
            // for AP,BQ terms: to ensure the optimal scaling!
            const bool ifNC = determine_NCorCN_Ham(oplist1, oplist2, cindex1.size(), cindex2.size()); 
            auto AP1 = ifNC? 'A' : 'P';
            auto AP2 = ifNC? 'P' : 'A';
            auto BQ1 = ifNC? 'B' : 'Q';
            auto BQ2 = ifNC? 'Q' : 'B';
            const auto& cindex = ifNC? cindex1 : cindex2;
            auto aindex_dist = oper_index_opA_dist(cindex, ifkr, isym, size, rank, sorb);
            auto bindex_dist = oper_index_opB_dist(cindex, ifkr, isym, size, rank, sorb);
            // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c. (see eq25 in R_CTNS.pdf)
            for(const auto& index : aindex_dist){
               const double wt = ifkr? wfacAP(index) : 1.0;
               auto op1 = symbolic_oper(block1,AP1,index);
               auto op2 = symbolic_oper(block2,AP2,index);
               auto AP = symbolic_prod<Tm>(op1, op2, wt);
               formulae.append(AP);
            }
            // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
            for(const auto& index : bindex_dist){
               const double wt = ifkr? wfacBQ(index) : wfac(index);
               auto op1 = symbolic_oper(block1,BQ1,index);
               auto op2 = symbolic_oper(block2,BQ2,index);
               auto BQ = symbolic_prod<Tm>(op1, op2, wt);
               formulae.append(BQ);
            }
         }else if(formula == 1){
            if(ifkr) tools::exit("error: formula=1 in opH does not support ifkr=true yet!"); 
            // AL*PC = {AL*x}*AC[all are available for dot!]
            auto aindex2 = oper_index_opA(cindex2, ifkr, isym);
            auto bindex2 = oper_index_opB(cindex2, ifkr, isym);
            auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, isym, size, rank, sorb);
            auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, isym, size, rank, sorb);
            // \sum_{pL<qL} A^L_{pq}*P^C_{pq} (ZL@2024/11/23 see eq24 in R_CTNS.pdf) 
            // = \sum_{pL<qL} A^L_{pq}*(\sum_{sC<rC} eri(p,q,s,r)*A^C_{sr}^dagger)
            // = \sum_{sC<rC} (\sum_{pL<qL} A^L_{pq}*eri(p,q,s,r)) * A^C_{sr}^dagger
            for(const auto& isr : aindex2){
               auto sr = oper_unpack(isr);
               int s = sr.first;
               int r = sr.second;
               auto op2 = symbolic_oper(block2,'A',isr).H();
               auto sym_op2 = op2.get_qsym(isym);
               symbolic_sum<Tm> top1;
               for(const auto& ipq : aindex1_dist){
                  auto pq = oper_unpack(ipq);
                  int p = pq.first;
                  int q = pq.second;
                  auto op1 = symbolic_oper(block1,'A',ipq);
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op1 == -sym_op2) top1.sum(int2e.get(p,q,s,r), op1);
               }
               if(top1.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1, op2);
                  formulae.append(op12);
               }
            }
            // \sum_{pL<=sL} w_{ps} B^L_{ps} Q^C_{ps} (ZL@2024/11/23 see eq25 in R_CTNS.pdf)
            // = \sum_{pL<=sL} w_{ps} B^L_{ps} \sum_{q_Cr_C} eri(p,q,s,r) B^C_{qr}
            // = \sum_{pL<=sL} w_{ps} B^L_{ps} \sum_{q_C<=r_C} w_{qr} [eri(p,q,s,r) B^C_{qr} + eri(p,r,s,q) B^C_{qr}^dagger]
            // = \sum_{qC<=rC} w_{qr} [\sum_{pL<=sL} w_{ps} B^L_{ps} eri(p,q,s,r)] * B^C_{qr}
            // + \sum_{qC<=rC} w_{qr} [\sum_{pL<=sL} w_{ps} B^L_{ps} eri(p,r,s,q)] * B^C_{qr}^dagger
            for(const auto& iqr : bindex2){
               auto qr = oper_unpack(iqr);
               int q = qr.first;
               int r = qr.second;
               double wqr = (q==r)? 0.5 : 1.0;
               auto op2 = symbolic_oper(block2,'B',iqr);
               auto op2H = symbolic_oper(block2,'B',iqr).H();
               auto sym_op2 = op2.get_qsym(isym);
               auto sym_op2H = op2H.get_qsym(isym);
               symbolic_sum<Tm> top1, top1H;
               for(const auto& ips : bindex1_dist){
                  auto ps = oper_unpack(ips);
                  int p = ps.first;
                  int s = ps.second;
                  double wps = (p==s)? 0.5 : 1.0;
                  auto op1 = symbolic_oper(block1,'B',ips);
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op1 == -sym_op2) top1.sum(wps*wqr*int2e.get(p,q,s,r), op1);
                  if(sym_op1 == -sym_op2H) top1H.sum(wps*wqr*int2e.get(p,r,s,q), op1);
               }
               if(top1.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1, op2);
                  formulae.append(op12);
               }
               if(top1H.size()>0){
                  auto op12 = symbolic_prod<Tm>(top1H, op2H);
                  formulae.append(op12);
               }
            }
         }else if(formula == 2){
            if(ifkr) tools::exit("error: formula=2 in opH does not support ifkr=true yet!"); 
            // PC*AR = AC*{x*AR}
            auto aindex1 = oper_index_opA(cindex1, ifkr, isym);
            auto bindex1 = oper_index_opB(cindex1, ifkr, isym);
            auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, isym, size, rank, sorb);
            auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, isym, size, rank, sorb);
            for(const auto& isr : aindex1){
               auto sr = oper_unpack(isr); // sr [Abar_sr]^0
               int s = sr.first;
               int r = sr.second;
               auto op1 = symbolic_oper(block1,'A',isr).H();
               auto sym_op1 = op1.get_qsym(isym);
               symbolic_sum<Tm> top2;
               for(const auto& ipq : aindex2_dist){ 
                  auto pq = oper_unpack(ipq);
                  int p = pq.first;
                  int q = pq.second;
                  auto op2 = symbolic_oper(block2,'A',ipq);
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op2 == -sym_op1) top2.sum(int2e.get(p,q,s,r), op2);
               }
               if(top2.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1, top2);
                  formulae.append(op12);
               }
            }
            for(const auto& iqr : bindex1){
               auto qr = oper_unpack(iqr); 
               int q = qr.first;
               int r = qr.second;
               double wqr = (q==r)? 0.5 : 1.0;
               auto op1 = symbolic_oper(block1,'B',iqr); 
               auto op1H = symbolic_oper(block1,'B',iqr).H(); 
               auto sym_op1 = op1.get_qsym(isym);
               auto sym_op1H = op1H.get_qsym(isym);
               symbolic_sum<Tm> top2, top2H;
               for(const auto& ips : bindex2_dist){ 
                  auto ps = oper_unpack(ips);
                  int p = ps.first;
                  int s = ps.second;
                  double wps = (p==s)? 0.5 : 1.0;
                  auto op2 = symbolic_oper(block2,'B',ips);
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op2 == -sym_op1) top2.sum(wps*wqr*int2e.get(p,q,s,r), op2);
                  if(sym_op2 == -sym_op1H) top2H.sum(wps*wqr*int2e.get(p,r,s,q), op2);
               }
               if(top2.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1, top2);
                  formulae.append(op12);
               }
               if(top2H.size()>0){
                  auto op12 = symbolic_prod<Tm>(op1H, top2H);
                  formulae.append(op12);
               }
            }
         }
         if(debug_compxwf){
            tsize = formulae.size()-tsize;
            std::cout << "formula(two-index)=" << formula << " size=" << tsize 
               << " ifdistc=" << ifdistc << " block1=" << block1 << " block2=" << block2
               << std::endl;
         }
         return formulae;
      }

} // ctns

#endif
