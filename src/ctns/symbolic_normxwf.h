#ifndef SYMBOLIC_NORMXWF_H
#define SYMBOLIC_NORMXWF_H

#include "symbolic_task.h"

namespace ctns{

   // kernel for computing Cp|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opC(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'C',index),
                  symbolic_oper(block2,'I',0));
            formulae.append(op1);
         }else if(iformula == 2){
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'C',index));
            formulae.append(op2);
         } // iformula
         return formulae;
      }

   // kernel for computing Apq|ket> 
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opA(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula,
            const bool ifkr){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            // A[p1q1]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'A',index),
                  symbolic_oper(block2,'I',0));
            formulae.append(op1);
         }else if(iformula == 2){
            // A[p2q2]
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'A',index));
            formulae.append(op2);
         }else if(iformula == 3){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            // A[p1<q2] = p1+q2+
            auto op1 = symbolic_oper(block1,'C',p);
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op2 = ifnot_kros? symbolic_oper(block2,'C',q) : 
               symbolic_oper(block2,'C',q-1).K(1);
            auto op12 = symbolic_prod<Tm>(op1,op2);
            formulae.append(op12);
         }else if(iformula == 4){
            auto qp = oper_unpack(index);	
            int p = qp.second, sp = p%2;
            int q = qp.first, sq = q%2;
            // A[q2<p1] = q2+p1+ = -p1+q2+
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op1 = ifnot_kros? symbolic_oper(block1,'C',p) :
               symbolic_oper(block1,'C',p-1).K(1);
            auto op2 = symbolic_oper(block2,'C',q);
            auto op12 = symbolic_prod<Tm>(op1,op2,-1.0);
            formulae.append(op12);
         } // iformula
         return formulae;
      }

   // kernel for computing Bps|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opB(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula,
            const bool ifkr,
            const bool ifDop=false){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            // B[p1q1]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'B',index),
                  symbolic_oper(block2,'I',0));
            formulae.append(op1);
         }else if(iformula == 2){
            // B[p2q2]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'B',index));
            formulae.append(op1);
         }else if(iformula == 3){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            // B[p1q2] = p1+q2
            auto op1 = symbolic_oper(block1,'C',p);
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op2 = ifDop? symbolic_oper(block2,'D',q,false) :
                  (ifnot_kros? symbolic_oper(block2,'C',q,true) : symbolic_oper(block2,'C',q-1,true).K(1));
            auto op12 = symbolic_prod<Tm>(op1,op2);
            formulae.append(op12);
         }else if(iformula == 4){
            auto qp = oper_unpack(index);	
            int p = qp.second, sp = p%2;
            int q = qp.first, sq = q%2;
            // B[q2p1] = q2+p1 = -p1q2+
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op1 = ifDop? symbolic_oper(block1,'D',p,false) : 
               (ifnot_kros? symbolic_oper(block1,'C',p,true) : symbolic_oper(block1,'C',p-1,true).K(1));
            auto op2 = symbolic_oper(block2,'C',q);
            auto op12 = symbolic_prod<Tm>(op1,op2,-1.0);
            formulae.append(op12);
         } // iformula
         return formulae;
      }

   // ----- ZL@20240906: for RDM calculations -----
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opI(const std::string block1,
            const std::string block2){
         symbolic_task<Tm> formulae;
         auto op1 = symbolic_oper(block1,'I',0);
         auto op2 = symbolic_oper(block2,'I',0);
         auto op12 = symbolic_prod<Tm>(op1, op2); 
         formulae.append(op12);
         return formulae;
      }
   // kernel for computing Dp|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opD(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'D',index),
                  symbolic_oper(block2,'I',0));
            formulae.append(op1);
         }else if(iformula == 2){
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'D',index));
            formulae.append(op2);
         } // iformula
         return formulae;
      }
   // kernel for computing Mpq|ket> 
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opM(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula,
            const bool ifkr){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            // A[p1q1]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'M',index),
                  symbolic_oper(block2,'I',0));
            formulae.append(op1);
         }else if(iformula == 2){
            // A[p2q2]
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'M',index));
            formulae.append(op2);
         }else if(iformula == 3){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            // A[p1<q2] = p1+q2+
            auto op1 = symbolic_oper(block1,'D',p);
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op2 = ifnot_kros? symbolic_oper(block2,'D',q) : 
               symbolic_oper(block2,'D',q-1).K(1);
            auto op12 = symbolic_prod<Tm>(op1,op2);
            formulae.append(op12);
         }else if(iformula == 4){
            auto qp = oper_unpack(index);	
            int p = qp.second, sp = p%2;
            int q = qp.first, sq = q%2;
            // A[q2<p1] = q2+p1+ = -p1+q2+
            const bool ifnot_kros = !(ifkr && sp != sq);
            auto op1 = ifnot_kros? symbolic_oper(block1,'D',p) :
               symbolic_oper(block1,'D',p-1).K(1);
            auto op2 = symbolic_oper(block2,'D',q);
            auto op12 = symbolic_prod<Tm>(op1,op2,-1.0);
            formulae.append(op12);
         } // iformula
         return formulae;
      }

} // ctns

#endif
