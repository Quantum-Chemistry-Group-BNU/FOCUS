#ifndef SYMBOLIC_NORMXWF_SU2_H
#define SYMBOLIC_NORMXWF_SU2_H

#include "../symbolic_task.h"

namespace ctns{

   // kernel for computing Cp|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opC_su2(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'C',index),
                  symbolic_oper(block2,'I',0));
            op1.ispins.push_back(std::make_tuple(1,0,1)); // ts1,ts2,ts12
            formulae.append(op1);
         }else if(iformula == 2){
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'C',index));
            op2.ispins.push_back(std::make_tuple(0,1,1));
            formulae.append(op2);
         } // iformula
         return formulae;
      }

   // kernel for computing Apq|ket> 
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opA_su2(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 0 : 2; // we use opposite spin case to store singlet
            // A[p1q1]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'A',index),
                  symbolic_oper(block2,'I',0));
            op1.ispins.push_back(std::make_tuple(ts,0,ts));
            formulae.append(op1);
         }else if(iformula == 2){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 0 : 2;
            // A[p2q2]
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'A',index));
            op2.ispins.push_back(std::make_tuple(0,ts,ts));
            formulae.append(op2);
         }else if(iformula == 3){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 0 : 2;
            // A[p1<q2] = p1+q2+
            auto op1 = symbolic_oper(block1,'C',p);
            auto op2 = (sp==sq)? symbolic_oper(block2,'C',q) : 
               symbolic_oper(block2,'C',q-1);
            auto op12 = symbolic_prod<Tm>(op1,op2);
            op12.ispins.push_back(std::make_tuple(1,1,ts));
            formulae.append(op12);
         }else if(iformula == 4){
            auto qp = oper_unpack(index); // in this case: qApA, qApB is stored	
            int p = qp.second, sp = p%2;
            int q = qp.first, sq = q%2;
            int ts = (sp!=sq)? 0 : 2;
            // A[q2<p1] = q2+p1+ = -p1+q2+
            auto op1 = (sp==sq)? symbolic_oper(block1,'C',p) :
               symbolic_oper(block1,'C',p-1);
            auto op2 = symbolic_oper(block2,'C',q);
            double fac = (ts==0)? 1 : -1; // su2 case: permutation sgn depending on K
            auto op12 = symbolic_prod<Tm>(op1,op2,fac);
            op12.ispins.push_back(std::make_tuple(1,1,ts));
            formulae.append(op12);
         } // iformula
         return formulae;
      }

   // kernel for computing Bps|ket>
   template <typename Tm>
      symbolic_task<Tm> symbolic_normxwf_opB_su2(const std::string block1,
            const std::string block2,
            const int index,
            const int iformula){
         symbolic_task<Tm> formulae;
         if(iformula == 1){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 2 : 0; 
            // B[p1q1]
            auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'B',index),
                  symbolic_oper(block2,'I',0));
            op1.ispins.push_back(std::make_tuple(ts,0,ts));
            formulae.append(op1);
         }else if(iformula == 2){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 2 : 0; 
            // B[p2q2]
            auto op2 = symbolic_prod<Tm>(symbolic_oper(block1,'I',0),
                  symbolic_oper(block2,'B',index));
            op2.ispins.push_back(std::make_tuple(0,ts,ts));
            formulae.append(op2);
         }else if(iformula == 3){
            auto pq = oper_unpack(index);	
            int p = pq.first, sp = p%2;
            int q = pq.second, sq = q%2;
            int ts = (sp!=sq)? 2 : 0; 
            // B[p1q2] = p1+q2
            auto op1 = symbolic_oper(block1,'C',p);
            auto op2 = (sp==sq)? symbolic_oper(block2,'C',q,true) : // dagger=true [a^+]^+ = a
               symbolic_oper(block2,'C',q-1,true);
            auto op12 = symbolic_prod<Tm>(op1,op2);
            op12.ispins.push_back(std::make_tuple(1,1,ts));
            formulae.append(op12);
         }else if(iformula == 4){
            auto qp = oper_unpack(index);	
            int p = qp.second, sp = p%2;
            int q = qp.first, sq = q%2;
            int ts = (sp!=sq)? 2 : 0; 
            // B[q2p1] = q2+p1 = -p1q2+
            auto op1 = (sp==sq)? symbolic_oper(block1,'C',p,true) : 
               symbolic_oper(block1,'C',p-1,true);
            auto op2 = symbolic_oper(block2,'C',q);
            double fac = (ts==0)? 1 : -1; // su2 case: permutation sgn depending on K
            auto op12 = symbolic_prod<Tm>(op1,op2,fac);
            op12.ispins.push_back(std::make_tuple(1,1,ts));
            formulae.append(op12);
         } // iformula
         return formulae;
      }

} // ctns

#endif
