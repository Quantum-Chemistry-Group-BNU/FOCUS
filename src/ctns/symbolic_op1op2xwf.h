#ifndef SYMBOLIC_OP1OP2XWF_H
#define SYMBOLIC_OP1OP2XWF_H

#include "symbolic_task.h"

namespace ctns{

   //
   // opxwf = (sum_{ij} oij*a1^(d1)[i]*a2^(d2)[i])^d * wf
   //

   template <typename Tm>
      void symbolic_op1op2xwf_nkr(symbolic_task<Tm>& formulae,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int isym,
            const qsym& sym_op,
            const std::map<int,Tm>& oij,
            const bool ifdagger1,
            const bool ifdagger2,
            const bool ifdagger){
         symbolic_prod<Tm> term;
         if(cindex1.size() <= cindex2.size()){
            // sum_i a1[i] * (sum_j oij a2[j])
            for(const auto& i : cindex1){
               auto op1C = symbolic_oper(block1,'C',i);      
               auto op1 = ifdagger1? op1C : op1C.H();
               auto sym_op1 = op1.get_qsym(isym); 
               // top2 = sum_j oij a2[j]
               symbolic_sum<Tm> top2;
               for(const auto& j : cindex2){
                  auto op2C = symbolic_oper(block2,'C',j);		 
                  auto op2 = ifdagger2? op2C : op2C.H();
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top2.sum(oij.at(oper_pack(i,j)), op2);
               } // j
               formulae.append(op1,top2,ifdagger);
            } // i
         }else{
            // this part appears when the branch is larger 
            // sum_j (sum_i oij a1[i]) * a2[j]
            for(const auto& j : cindex2){
               auto op2C = symbolic_oper(block2,'C',j);
               auto op2 = ifdagger2? op2C : op2C.H();
               auto sym_op2 = op2.get_qsym(isym);
               // tmp_op1 = sum_i oij a1[i]
               symbolic_sum<Tm> top1;
               for(const auto& i : cindex1){
                  auto op1C = symbolic_oper(block1,'C',i);
                  auto op1 = ifdagger1? op1C : op1C.H();
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top1.sum(oij.at(oper_pack(i,j)), op1);
               } // i
               formulae.append(top1,op2,ifdagger);
            } // j
         } // cindex1.size() <= cindex2.size() 
      }

   // TRS version: only unbar part of creation ops is stored
   template <typename Tm>
      void symbolic_op1op2xwf_kr(symbolic_task<Tm>& formulae,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int isym,
            const qsym& sym_op,
            const std::map<int,Tm>& oij,
            const bool ifdagger1,
            const bool ifdagger2,
            const bool ifdagger){
         symbolic_prod<Tm> term;
         if(cindex1.size() <= cindex2.size()){
            // sum_i a1[i] * (sum_j oij a2[j])
            for(const auto& ia : cindex1){
               int ib = ia+1;
               auto op1Ca = symbolic_oper(block1,'C',ia);
               auto op1a = ifdagger1? op1Ca : op1Ca.H();
               auto op1b = op1a.K(1);
               // top2 = sum_j oij a2[j]
               symbolic_sum<Tm> top2a, top2b;
               for(const auto& ja : cindex2){
                  int jb = ja+1;
                  auto op2Ca = symbolic_oper(block2,'C',ja);
                  auto op2a = ifdagger2? op2Ca : op2Ca.H();
                  auto op2b = op2a.K(1);
                  top2a.sum(oij.at(oper_pack(ia,ja)), op2a);
                  top2a.sum(oij.at(oper_pack(ia,jb)), op2b);
                  top2b.sum(oij.at(oper_pack(ib,ja)), op2a);
                  top2b.sum(oij.at(oper_pack(ib,jb)), op2b);
               } // ja
               formulae.append(op1a,top2a,ifdagger);
               formulae.append(op1b,top2b,ifdagger);
            } // ia
         }else{
            // this part appears when the branch is larger 
            // sum_j (sum_i oij a1[i]) * a2[j]
            for(const auto& ja : cindex2){
               int jb = ja+1;
               auto op2Ca = symbolic_oper(block2,'C',ja);
               auto op2a = ifdagger2? op2Ca : op2Ca.H();
               auto op2b = op2a.K(1);
               // top1 = sum_i oij a1[i]
               symbolic_sum<Tm> top1a, top1b;
               for(const auto& ia : cindex1){
                  int ib = ia+1;
                  auto op1Ca = symbolic_oper(block1,'C',ia);
                  auto op1a = ifdagger1? op1Ca : op1Ca.H();
                  auto op1b = op1a.K(1);
                  top1a.sum(oij.at(oper_pack(ia,ja)), op1a);
                  top1a.sum(oij.at(oper_pack(ib,ja)), op1b);
                  top1b.sum(oij.at(oper_pack(ia,jb)), op1a);
                  top1b.sum(oij.at(oper_pack(ib,jb)), op1b);
               } // ia
               formulae.append(top1a,op2a,ifdagger);
               formulae.append(top1b,op2b,ifdagger);	 
            } // ja 
         }  // cindex1.size() <= cindex2.size()
      }

   // opxwf = (sum_{ij}oij*a1^(d)[i]*a2^(d)[i])*wf
   template <typename Tm>
      void symbolic_op1op2xwf(const bool ifkr,
            symbolic_task<Tm>& formulae,
            const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int isym,
            const qsym& sym_op,
            const std::map<int,Tm>& oij,
            const bool ifdagger1,
            const bool ifdagger2,
            const bool ifdagger){
         if(!ifkr){
            symbolic_op1op2xwf_nkr(formulae,block1,block2,cindex1,cindex2,
                  isym,sym_op,oij,ifdagger1,ifdagger2,ifdagger);
         }else{
            symbolic_op1op2xwf_kr(formulae,block1,block2,cindex1,cindex2,
                  isym,sym_op,oij,ifdagger1,ifdagger2,ifdagger);
         }
      }

} // ctns

#endif
