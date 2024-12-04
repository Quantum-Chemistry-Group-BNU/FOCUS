#ifndef SYMBOLIC_FORMULAE_TWODOT_SU2_H
#define SYMBOLIC_FORMULAE_TWODOT_SU2_H

#include "../../core/tools.h"
#include "../oper_dict.h"
#include "../symbolic_oper.h"
#include "symbolic_normxwf_su2.h"
#include "symbolic_compxwf_su2.h"

namespace ctns{

   template <typename Tm>
      void symbolic_Slc1_Cc2r_form(const symbolic_task<Tm>& Slc1,
            const symbolic_task<Tm>& Cc2r,
            const bool ifsave,
            const int print_level,
            size_t& idx,
            std::map<std::string,int>& counter,
            symbolic_task<Tm>& formulae,
            const std::string msg){
         auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
         Slc1_Cc2r.scale(std::sqrt(2.0));
         Slc1_Cc2r.append_ispins(std::make_tuple(1,1,0));
         formulae.join(Slc1_Cc2r);
         counter["SC"] += Slc1_Cc2r.size();
         if(ifsave){ 
            std::cout << "idx=" << idx++;
            Slc1_Cc2r.display(msg, print_level);
         }
      }

   template <typename Tm>
      void symbolic_Slc1_Cc2r_fromAB(const std::string oplist_l,
            const std::string oplist_r,
            const std::string oplist_c1,
            const std::string oplist_c2,
            const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c1,
            const std::vector<int>& cindex_c2,
            const int isym, // for uniform interface only
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            const int print_level,
            size_t& idx,
            std::map<std::string,int>& counter,
            symbolic_task<Tm>& formulae){
         // c[c2R] = c[c2]*I[R]
         for(const auto& index : cindex_c2){
            auto Slc1_task = symbolic_compxwf_opS_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1,
                  int2e, index, ifkr, size, rank, ifdist1, ifdistc);
            if(Slc1_task.size() == 0) continue;
            int iformula = 1;
            auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
            symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                  "Slc1_Cc2r["+std::to_string(index)+"]");

         }
         // c[c2R] = I[c2]*c[R]
         // S[LC1]_1, S[LC1]_2
         for(const auto& index: cindex_r){
            symbolic_task<Tm> Slc1_task;
            int iproc = distribute1(ifkr,size,index);
            if(!ifdist1 or iproc==rank){
               // 1. S1*I2
               auto S1p = symbolic_prod<Tm>(symbolic_oper("l",'S',index),
                     symbolic_oper("c1",'I',0));
               S1p.ispins.push_back(std::make_tuple(1,0,1)); 
               Slc1_task.append(S1p);
               // 2. I1*S2
               auto S2p = symbolic_prod<Tm>(symbolic_oper("l",'I',0),
                     symbolic_oper("c1",'S',index));
               S2p.ispins.push_back(std::make_tuple(0,1,1));
               Slc1_task.append(S2p);
            }
            if(Slc1_task.size() == 0) continue;
            int iformula = 2;
            auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
            symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                  "Slc1_Cc2r(1,2)["+std::to_string(index)+"]");
         }
         // S[LC1]_3 and S[LC1]_4 (adapted from S3b and S4b)
         auto aindex_c1 = oper_index_opA(cindex_c1, ifkr, isym);
         for(const auto& q1 : cindex_l){
            int iproc = distribute1(ifkr,size,q1);
            if(iproc != rank) continue;
            // loop over Asr[c1]
            auto op1 = symbolic_oper("l",'C',q1);
            for(const auto& isr : aindex_c1){
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               int spin_s2 = s2%2, spin_r2 = r2%2;
               int ts = (spin_s2!=spin_r2)? 0 : 2;
               auto op2 = symbolic_oper("c1",'A',isr).H();
               auto Slc1 = symbolic_prod<Tm>(op1,op2);
               Slc1.ispins.push_back(std::make_tuple(1,ts,1));
               // op_c2r
               symbolic_sum<Tm> top_r;
               for(const auto& p : cindex_r){
                  auto op_r = symbolic_oper("r",'C',p); 
                  double fac = (ts==0)? -1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
                  top_r.sum(fac*get_xint2e_su2(int2e,ts,p/2,q1/2,ks,kr), op_r);
               }
               auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
               Cc2r.ispins.push_back(std::make_tuple(0,1,1));
               auto Slc1_task = symbolic_task<Tm>(Slc1);
               auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
               symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                     "Slc1_Cc2r(3)["+std::to_string(q1)+"]");
            }
         }
         auto bindex_c1 = oper_index_opB(cindex_c1, ifkr, isym);
         for(const auto& s1 : cindex_l){
            int iproc = distribute1(ifkr,size,s1);
            if(iproc != rank) continue;
            // loop over Bqr[c1]
            auto op1 = symbolic_oper("l",'C',s1).H();
            for(const auto& iqr : bindex_c1){
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq2 = q2/2;
               int r2 = qr.second, kr2 = r2/2;
               int spin_q2 = q2%2, spin_r2 = r2%2;
               int ts = (spin_q2!=spin_r2)? 2 : 0;
               auto op2 = symbolic_oper("c1",'B',iqr);
               {
                  auto Slc1 = symbolic_prod<Tm>(op1,op2);
                  Slc1.ispins.push_back(std::make_tuple(1,ts,1));
                  // op_c2r
                  symbolic_sum<Tm> top_r;
                  for(const auto& p : cindex_r){
                     auto op1 = symbolic_oper("r",'C',p);
                     double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                     top_r.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq2,s1/2,kr2), op1);
                  }
                  auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
                  Cc2r.ispins.push_back(std::make_tuple(0,1,1));
                  auto Slc1_task = symbolic_task<Tm>(Slc1);
                  auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
                  symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                       "Slc1_Cc2r(4)["+std::to_string(s1)+"]");
               }
               // Hermitian part: q2<->r2
               if(kq2 == kr2) continue;
               {
                  // We use [Brq]^k = (-1)^k*[Bqr]^k
                  auto op2H = op2.H();
                  auto Slc1 = symbolic_prod<Tm>(op1,op2H);
                  Slc1.ispins.push_back(std::make_tuple(1,ts,1));
                  // op_c2r
                  symbolic_sum<Tm> top_r;
                  for(const auto& p : cindex_r){
                     auto op1 = symbolic_oper("r",'C',p);
                     double fac = (ts==0)? 1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
                     top_r.sum(fac*get_vint2e_su2(int2e,ts,p/2,kr2,s1/2,kq2), op1);
                  }
                  auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
                  Cc2r.ispins.push_back(std::make_tuple(0,1,1));
                  auto Slc1_task = symbolic_task<Tm>(Slc1);
                  auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
                  symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                        "Slc1_Cc2r(4H)["+std::to_string(s1)+"]");
               }
            }
         }
         // S[LC1]_5 
         auto aindex_l_dist = oper_index_opA_dist(cindex_l, ifkr, isym, size, rank, int2e.sorb);
         if(2*cindex_r.size() <= aindex_l_dist.size()){
            for(const auto& index: cindex_r){
               symbolic_task<Tm> Slc1_task;
               symbolic_compxwf_opS5c_su2("l", "c1", cindex_l, cindex_c1, int2e, index, aindex_l_dist, Slc1_task);
               if(Slc1_task.size() == 0) continue;
               int iformula = 2;
               auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
               symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                     "Slc1_Cc2r(5c)["+std::to_string(index)+"]");
            }
         }else{
            // adapted from S5b
            for(const auto& q2a : cindex_c1){
               auto op2 = symbolic_oper("c1",'C',q2a);
               // loop over Asr[l]
               for(const auto& isr : aindex_l_dist){
                  auto sr = oper_unpack(isr);
                  int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2;
                  int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2;
                  int ts = (spin_s1!=spin_r1)? 0 : 2;
                  auto op1 = symbolic_oper("l",'A',isr).H();
                  auto Slc1 = symbolic_prod<Tm>(op1,op2);
                  Slc1.ispins.push_back(std::make_tuple(ts,1,1));
                  // op_c2r = sum_q <pq2||s1r1> ap[R]^+
                  symbolic_sum<Tm> top_r;
                  for(const auto& p : cindex_r){
                     auto op_r = symbolic_oper("r",'C',p);
                     double fac = (ts==0)? -1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                     top_r.sum(fac*get_xint2e_su2(int2e,ts,p/2,q2a/2,ks1,kr1), op_r);
                  }
                  auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
                  Cc2r.ispins.push_back(std::make_tuple(0,1,1));
                  auto Slc1_task = symbolic_task<Tm>(Slc1);
                  auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
                  symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                        "Slc1_Cc2r(5d)["+std::to_string(isr)+"]");
               }
            }
         }
         // S[LC1]_6
         auto bindex_l_dist = oper_index_opB_dist(cindex_l, ifkr, isym, size, rank, int2e.sorb);
         if(4*cindex_r.size() <= 2*bindex_l_dist.size()){
            for(const auto& index: cindex_r){
               symbolic_task<Tm> Slc1_task;
               symbolic_compxwf_opS6c_su2("l", "c1", cindex_l, cindex_c1, int2e, index, bindex_l_dist, Slc1_task);
               if(Slc1_task.size() == 0) continue;
               int iformula = 2;
               auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
               symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                        "Slc1_Cc2r(6c)["+std::to_string(index)+"]");
            }
         }else{
            // adapted from S6b
            for(const auto& r2a : cindex_c1){
               auto op2 = symbolic_oper("c1",'C',r2a).H();
               // loop over Bqs[l]
               for(const auto& iqs : bindex_l_dist){
                  auto qs = oper_unpack(iqs);
                  int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2;
                  int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2;
                  int ts = (spin_q1!=spin_s1)? 2 : 0;
                  auto op1 = symbolic_oper("l",'B',iqs);
                  {
                     auto Slc1 = symbolic_prod<Tm>(op1,op2);
                     Slc1.ispins.push_back(std::make_tuple(ts,1,1));
                     // op_c2r 
                     symbolic_sum<Tm> top_r;
                     for(const auto& p : cindex_r){
                        auto op_r = symbolic_oper("r",'C',p);
                        double fac = (ts==0)? 1.0/std::sqrt(2.0) : std::sqrt(3.0/2.0);
                        top_r.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq1,r2a/2,ks1), op_r);
                     }
                     auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
                     Cc2r.ispins.push_back(std::make_tuple(0,1,1));
                     auto Slc1_task = symbolic_task<Tm>(Slc1);
                     auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
                     symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                           "Slc1_Cc2r(6d)["+std::to_string(iqs)+"]");
                  }
                  // Hermitian part: q1<->s1
                  if(kq1 == ks1) continue;
                  {
                     // We use [Bsq]^k = (-1)^k*[Bqs]^k
                     auto op1H = op1.H();
                     auto Slc1 = symbolic_prod<Tm>(op1H,op2);
                     Slc1.ispins.push_back(std::make_tuple(ts,1,1));
                     // op_c2r
                     symbolic_sum<Tm> top_r;
                     for(const auto& p : cindex_r){
                        auto op_r = symbolic_oper("r",'C',p);
                        double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                        top_r.sum(fac*get_vint2e_su2(int2e,ts,p/2,ks1,r2a/2,kq1), op_r); // s<->q
                     }
                     auto Cc2r = symbolic_prod(symbolic_oper("c2",'I',0),top_r);
                     Cc2r.ispins.push_back(std::make_tuple(0,1,1));
                     auto Slc1_task = symbolic_task<Tm>(Slc1);
                     auto Cc2r_task = symbolic_task<Tm>(Cc2r);  
                     symbolic_Slc1_Cc2r_form(Slc1_task, Cc2r_task, ifsave, print_level, idx, counter, formulae,
                           "Slc1_Cc2r(6dH)["+std::to_string(iqs)+"]");
                  }
               }
            }
         }
      }

   template <typename Tm>
      symbolic_task<Tm> gen_formulae_twodot_su2(const std::string oplist_l,
            const std::string oplist_r,
            const std::string oplist_c1,
            const std::string oplist_c2,
            const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c1,
            const std::vector<int>& cindex_c2,
            const int isym, // for uniform interface only
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            std::map<std::string,int>& counter){
         assert(isym == 3);
         const int print_level = 1;
         const size_t csize_lc1 = cindex_l.size() + cindex_c1.size();
         const size_t csize_c2r = cindex_c2.size() + cindex_r.size();
         const bool ifNC = determine_NCorCN_Ham(oplist_l, oplist_r, csize_lc1, csize_c2r);
         const bool ifhermi = true;

         symbolic_task<Tm> formulae;
         size_t idx = 0;

         // Local terms:
         // H[lc1]
         auto Hlc1 = symbolic_compxwf_opH_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1, 
               int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
         counter["H1"] = Hlc1.size();
         if(Hlc1.size() > 0){
            auto op2 = symbolic_prod<Tm>(symbolic_oper("c2",'I',0),symbolic_oper("r",'I',0));
            op2.ispins.push_back(std::make_tuple(0,0,0));
            symbolic_task<Tm> Ic2r;
            Ic2r.append(op2);
            auto Hlc1_Ic2r = Hlc1.outer_product(Ic2r);
            Hlc1_Ic2r.append_ispins(std::make_tuple(0,0,0));
            formulae.join(Hlc1_Ic2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++; 
               Hlc1_Ic2r.display("Hlc1_Ic2r", print_level);
            }
         }
         // H[c2r]
         auto Hc2r = symbolic_compxwf_opH_su2<Tm>(oplist_c2, oplist_r, "c2", "r", cindex_c2, cindex_r, 
               int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
         counter["H2"] = Hc2r.size();
         if(Hc2r.size() > 0){
            auto op1 = symbolic_prod<Tm>(symbolic_oper("l",'I',0),symbolic_oper("c1",'I',0));
            op1.ispins.push_back(std::make_tuple(0,0,0));
            symbolic_task<Tm> Ilc1;
            Ilc1.append(op1);
            auto Ilc1_Hc2r = Ilc1.outer_product(Hc2r);
            Ilc1_Hc2r.append_ispins(std::make_tuple(0,0,0)); 
            formulae.join(Ilc1_Hc2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Ilc1_Hc2r.display("Ilc1_Hc2r", print_level);
            }
         }

         // One-index terms:
         // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
         counter["CS"] = 0;
         assert(ifexistQ(oplist_r,'P') == ifexistQ(oplist_r,'Q'));
         if(ifexistQ(oplist_r,'P')){
            auto infoC1 = oper_combine_opC(cindex_l, cindex_c1);
            for(const auto& pr : infoC1){
               int index = pr.first;
               int iformula = pr.second;
               // p1^L1C1+*Sp1^C2R & -p1^L1C1*Sp1^C2R+
               auto Clc1 = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
               auto Sc2r = symbolic_compxwf_opS_su2<Tm>(oplist_c2, oplist_r, "c2", "r", cindex_c2, cindex_r,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Sc2r.size() == 0) continue;
               auto Clc1_Sc2r = Clc1.outer_product(Sc2r);
               Clc1_Sc2r.scale(std::sqrt(2.0));
               Clc1_Sc2r.append_ispins(std::make_tuple(1,1,0));
               formulae.join(Clc1_Sc2r);
               counter["CS"] += Clc1_Sc2r.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Clc1_Sc2r.display("Clc1_Sc2r["+std::to_string(index)+"]", print_level);
               }
            }
         }else{
            // right block only contains A,B operators
            assert(ifexistQ(oplist_r,'A') and ifexistQ(oplist_r,'B'));
            std::cout << "not implemented yet!" << std::endl;
            exit(1);
         }

         // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
         counter["SC"] = 0;
         assert(ifexistQ(oplist_l,'P') == ifexistQ(oplist_l,'Q'));
         if(ifexistQ(oplist_l,'P')){
            auto infoC2 = oper_combine_opC(cindex_c2, cindex_r);
            for(const auto& pr : infoC2){
               int index = pr.first;
               int iformula = pr.second;
               // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
               auto Slc1 = symbolic_compxwf_opS_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Slc1.size() == 0) continue;
               auto Cc2r = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
               auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
               Slc1_Cc2r.scale(std::sqrt(2.0));
               Slc1_Cc2r.append_ispins(std::make_tuple(1,1,0));
               formulae.join(Slc1_Cc2r);
               counter["SC"] += Slc1_Cc2r.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Slc1_Cc2r.display("Slc1_Cc2r["+std::to_string(index)+"]", print_level);
               }
            }
         }else{
            // left block only contains A,B operators
            assert(ifexistQ(oplist_l,'A') and ifexistQ(oplist_l,'B'));
            assert(csize_lc1 <= csize_c2r and ifdist1 and ifdistc);
            symbolic_Slc1_Cc2r_fromAB(oplist_l,oplist_r,oplist_c1,oplist_c2,
                  cindex_l,cindex_r,cindex_c1,cindex_c2,isym,ifkr,
                  int2e,size,rank,ifdist1,ifdistc,ifsave,print_level,
                  idx,counter,formulae);
         }

         // Two-index terms:
         if(ifNC){

            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["AP"] = 0;
            auto ainfo = oper_combine_opA(cindex_l, cindex_c1, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2;
               int iformula = pr.second;
               int iproc = distribute2('A',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Alc1 = symbolic_normxwf_opA_su2<Tm>("l", "c1", index, iformula);
                  auto Pc2r = symbolic_compxwf_opP_su2<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index);
                  auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  Alc1_Pc2r.scale(fac);
                  Alc1_Pc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Alc1_Pc2r);
                  counter["AP"] += Alc1_Pc2r.size();
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Alc1_Pc2r.display("Alc1_Pc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["BQ"] = 0;
            auto binfo = oper_combine_opB(cindex_l, cindex_c1, ifkr, ifhermi);
            for(const auto& pr : binfo){
               int index = pr.first;
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               int iformula = pr.second;
               int iproc = distribute2('B',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  auto Blc1 = symbolic_normxwf_opB_su2<Tm>("l", "c1", index, iformula);
                  auto Qc2r = symbolic_compxwf_opQ_su2<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index);
                  auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  Blc1_Qc2r.scale(fac);
                  Blc1_Qc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Blc1_Qc2r);
                  counter["BQ"] += Blc1_Qc2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Blc1_Qc2r.display("Blc1_Qc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }

         }else{

            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["PA"] = 0;
            auto ainfo = oper_combine_opA(cindex_c2, cindex_r, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2;
               int iformula = pr.second;
               int iproc = distribute2('A',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Plc1 = symbolic_compxwf_opP_su2<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index);
                  auto Ac2r = symbolic_normxwf_opA_su2<Tm>("c2", "r", index, iformula);
                  auto Plc1_Ac2r = Plc1.outer_product(Ac2r);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  Plc1_Ac2r.scale(fac);
                  Plc1_Ac2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Plc1_Ac2r);
                  counter["PA"] += Plc1_Ac2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Plc1_Ac2r.display("Plc1_Ac2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["QB"] = 0;
            auto binfo = oper_combine_opB(cindex_c2, cindex_r, ifkr, ifhermi);
            for(const auto& pr : binfo){
               int index = pr.first;
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               int iformula = pr.second;
               int iproc = distribute2('B',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  auto Qlc1 = symbolic_compxwf_opQ_su2<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index);
                  auto Bc2r = symbolic_normxwf_opB_su2<Tm>("c2", "r", index, iformula);
                  auto Qlc1_Bc2r = Qlc1.outer_product(Bc2r);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  Qlc1_Bc2r.scale(fac);
                  Qlc1_Bc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Qlc1_Bc2r);
                  counter["QB"] += Qlc1_Bc2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Qlc1_Bc2r.display("Qlc1_Bc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }

         } // ifNC
         return formulae;
      }

} // ctns

#endif
