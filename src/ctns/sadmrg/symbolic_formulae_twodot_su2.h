#ifndef SYMBOLIC_FORMULAE_TWODOT_SU2_H
#define SYMBOLIC_FORMULAE_TWODOT_SU2_H

#include "../../core/tools.h"
#include "../oper_dict.h"
#include "../symbolic_oper.h"
#include "symbolic_normxwf_su2.h"
#include "symbolic_compxwf_su2.h"

namespace ctns{

   // for constructing Hamiltonian
   template <typename Tm>
      void symbolic_Hops1ops2_su2(const symbolic_task<Tm>& ops1,
            const symbolic_task<Tm>& ops2,
            const std::string type,
            const int ts,
            const double fac,
            const bool ifsave,
            const int print_level,
            size_t& idx,
            std::map<std::string,int>& counter,
            symbolic_task<Tm>& formulae,
            const std::string msg){
         auto ops1_ops2 = ops1.outer_product(ops2);
         ops1_ops2.scale(fac);
         ops1_ops2.append_ispins(std::make_tuple(ts,ts,0));
         formulae.join(ops1_ops2);
         counter[type] += ops1_ops2.size();
         if(ifsave){ 
            std::cout << "idx=" << idx++;
            ops1_ops2.display(msg, print_level);
         }
      }

   // ZL@2024/12/04 introduced for better scalability for ifab2pq=true
   template <typename Tm>
      void symbolic_Clc1_Sc2r_fromAB(const std::string oplist_l,
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
         const double fac = std::sqrt(2.0);
         // c[LC1] = I[L]*c[C1]
         for(const auto& index : cindex_c1){
            auto Sc2r_task = symbolic_compxwf_opS_su2<Tm>(oplist_c2, oplist_r, "c2", "r", cindex_c2, cindex_r,
                  int2e, index, ifkr, size, rank, ifdist1, ifdistc);
            if(Sc2r_task.size() == 0) continue;
            int iformula = 2;
            auto Clc1_task = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
            symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level, 
                  idx, counter, formulae, "Clc1_Sc2r["+std::to_string(index)+"]");
         }
         // c[LC1] = c[L]*c[C1]
         // S[C2R]_1, S[C2R]_2
         for(const auto& index : cindex_l){
            symbolic_task<Tm> Sc2r_task;
            int iproc = distribute1(ifkr,size,index);
            if(!ifdist1 or iproc==rank){
               // 1. S1*I2
               auto S1p = symbolic_prod<Tm>(symbolic_oper("c2",'S',index), symbolic_oper("r",'I',0));
               S1p.ispins.push_back(std::make_tuple(1,0,1)); 
               Sc2r_task.append(S1p);
               // 2. I1*S2
               auto S2p = symbolic_prod<Tm>(symbolic_oper("c2",'I',0), symbolic_oper("r",'S',index));
               S2p.ispins.push_back(std::make_tuple(0,1,1));
               Sc2r_task.append(S2p);
            }
            if(Sc2r_task.size() == 0) continue;
            int iformula = 1;
            auto Clc1_task = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
            symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                  idx, counter, formulae, "Clc1_Sc2r["+std::to_string(index)+"]");
         }
         // S[C2R]_5 and S[C2R]_6 (adapted from S5b and S6b)
         auto aindex_c2 = oper_index_opA(cindex_c2, ifkr, isym);
         for(const auto& q2 : cindex_r){
            int iproc = distribute1(ifkr,size,q2);
            if(iproc != rank) continue;
            // loop over Asr
            auto op2c = symbolic_oper("r",'C',q2);
            for(const auto& isr : aindex_c2){
               auto sr = oper_unpack(isr);
               int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2;
               int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2;
               int ts = (spin_s1!=spin_r1)? 0 : 2;
               auto op1 = symbolic_oper("c2",'A',isr).H();
               auto Sc2r = symbolic_prod<Tm>(op1,op2c);
               Sc2r.ispins.push_back(std::make_tuple(ts,1,1));
               // op_lc1
               symbolic_sum<Tm> top_l;
               for(const auto& p : cindex_l){
                  auto op_l = symbolic_oper("l",'C',p);
                  double fac = (ts==0)? -1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                  top_l.sum(fac*get_xint2e_su2(int2e,ts,p/2,q2/2,ks1,kr1), op_l);
               }
               auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
               Clc1.ispins.push_back(std::make_tuple(1,0,1));
               auto Clc1_task = symbolic_task<Tm>(Clc1);
               auto Sc2r_task = symbolic_task<Tm>(Sc2r);
               symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                     idx, counter, formulae, "Clc1_Sc2r(5)["+std::to_string(q2)+"]");
            }
         }
         auto bindex_c2 = oper_index_opB(cindex_c2, ifkr, isym);
         for(const auto& r2 : cindex_r){
            int iproc = distribute1(ifkr,size,r2);
            if(iproc != rank) continue;
            // loop over Bqs
            auto op2 = symbolic_oper("r",'C',r2).H();
            for(const auto& iqs : bindex_c2){
               auto qs = oper_unpack(iqs);
               int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2;
               int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2;
               int ts = (spin_q1!=spin_s1)? 2 : 0;
               auto op1 = symbolic_oper("c2",'B',iqs);
               {
                  auto Sc2r = symbolic_prod<Tm>(op1,op2);
                  Sc2r.ispins.push_back(std::make_tuple(ts,1,1));
                  // op_lc1
                  symbolic_sum<Tm> top_l;
                  for(const auto& p : cindex_l){
                     auto op_l = symbolic_oper("l",'C',p);
                     double fac = (ts==0)? 1.0/std::sqrt(2.0) : std::sqrt(3.0/2.0);
                     top_l.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq1,r2/2,ks1), op_l);
                  }
                  auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
                  Clc1.ispins.push_back(std::make_tuple(1,0,1));
                  auto Clc1_task = symbolic_task<Tm>(Clc1);
                  auto Sc2r_task = symbolic_task<Tm>(Sc2r);
                  symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                        idx, counter, formulae, "Clc1_Sc2r(6)["+std::to_string(r2)+"]");
               }
               // Hermitian part: q1<->s1
               if(kq1 == ks1) continue;
               {
                  // We use [Bsq]^k = (-1)^k*[Bqs]^k
                  auto op1H = op1.H();
                  auto Sc2r = symbolic_prod<Tm>(op1H,op2);
                  Sc2r.ispins.push_back(std::make_tuple(ts,1,1));
                  // op_lc1
                  symbolic_sum<Tm> top_l;
                  for(const auto& p : cindex_l){
                     auto op_l = symbolic_oper("l",'C',p);
                     double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                     top_l.sum(fac*get_vint2e_su2(int2e,ts,p/2,ks1,r2/2,kq1), op_l); // s<->q
                  }
                  auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
                  Clc1.ispins.push_back(std::make_tuple(1,0,1));
                  auto Clc1_task = symbolic_task<Tm>(Clc1);
                  auto Sc2r_task = symbolic_task<Tm>(Sc2r);
                  symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                        idx, counter, formulae, "Clc1_Sc2r(6H)["+std::to_string(r2)+"]");
               }
            }
         }
         // S[C2R]_3 
         auto aindex_r_dist = oper_index_opA_dist(cindex_r, ifkr, isym, size, rank, int2e.sorb);
         if(2*cindex_l.size() <= aindex_r_dist.size()){
            for(const auto& index: cindex_l){
               symbolic_task<Tm> Sc2r_task;
               symbolic_compxwf_opS3c_su2<Tm>("c2", "r", cindex_c2, cindex_r, int2e, index, aindex_r_dist, Sc2r_task);
               if(Sc2r_task.size() == 0) continue;
               int iformula = 1;
               auto Clc1_task = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
               symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                     idx, counter, formulae, "Clc1_Sc2r(3c)["+std::to_string(index)+"]");
            }
         }else{
            // adapted from S3b
            for(const auto& q1 : cindex_c2){
               auto op1 = symbolic_oper("c2",'C',q1);
               // loop over Asr[r]
               for(const auto& isr : aindex_r_dist){
                  auto sr = oper_unpack(isr);
                  int s2 = sr.first, ks = s2/2;
                  int r2 = sr.second, kr = r2/2;
                  int spin_s2 = s2%2, spin_r2 = r2%2;
                  int ts = (spin_s2!=spin_r2)? 0 : 2;
                  auto op2 = symbolic_oper("r",'A',isr).H();
                  auto Sc2r = symbolic_prod<Tm>(op1,op2);
                  Sc2r.ispins.push_back(std::make_tuple(1,ts,1));
                  // op_lc1 = sum_q <pq1||s2r2> aq[L]^+
                  symbolic_sum<Tm> top_l;
                  for(const auto& p : cindex_l){
                     auto op_l = symbolic_oper("l",'C',p);
                     double fac = (ts==0)? -1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
                     top_l.sum(fac*get_xint2e_su2(int2e,ts,p/2,q1/2,ks,kr), op_l);
                  }
                  auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
                  Clc1.ispins.push_back(std::make_tuple(1,0,1));
                  auto Clc1_task = symbolic_task<Tm>(Clc1);
                  auto Sc2r_task = symbolic_task<Tm>(Sc2r);
                  symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                        idx, counter, formulae, "Clc1_Sc2r(3d)["+std::to_string(isr)+"]");
               }
            }
         }
         // S[C2R]_4
         auto bindex_r_dist = oper_index_opB_dist(cindex_r, ifkr, isym, size, rank, int2e.sorb);
         if(4*cindex_l.size() <= 2*bindex_r_dist.size()){
            for(const auto& index: cindex_l){
               symbolic_task<Tm> Sc2r_task;
               symbolic_compxwf_opS4c_su2<Tm>("c2", "r", cindex_c2, cindex_r, int2e, index, bindex_r_dist, Sc2r_task);
               if(Sc2r_task.size() == 0) continue;
               int iformula = 1;
               auto Clc1_task = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
               symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                     idx, counter, formulae, "Clc1_Sc2r(4c)["+std::to_string(index)+"]");
            }
         }else{
            // adapted from S4b
            for(const auto& s1 : cindex_c2){
               auto op1 = symbolic_oper("c2",'C',s1).H();
               // loop over Bqr[r]
               for(const auto& iqr : bindex_r_dist){
                  auto qr = oper_unpack(iqr);
                  int q2 = qr.first, kq2 = q2/2;
                  int r2 = qr.second, kr2 = r2/2;
                  int spin_q2 = q2%2, spin_r2 = r2%2;
                  int ts = (spin_q2!=spin_r2)? 2 : 0;
                  auto op2 = symbolic_oper("r",'B',iqr);
                  {
                     auto Sc2r = symbolic_prod<Tm>(op1,op2);
                     Sc2r.ispins.push_back(std::make_tuple(1,ts,1));
                     // top_l sum_s <pq2||s1r2> ap[1]
                     symbolic_sum<Tm> top_l;
                     for(const auto& p : cindex_l){
                        auto op_l = symbolic_oper("l",'C',p);
                        double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
                        top_l.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq2,s1/2,kr2), op_l);
                     }
                     auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
                     Clc1.ispins.push_back(std::make_tuple(1,0,1));
                     auto Clc1_task = symbolic_task<Tm>(Clc1);
                     auto Sc2r_task = symbolic_task<Tm>(Sc2r);
                     symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                           idx, counter, formulae, "Clc1_Sc2r(4d)["+std::to_string(iqr)+"]");
                  }
                  // Hermitian part: q2<->r2
                  if(kq2 == kr2) continue;
                  {
                     // We use [Brq]^k = (-1)^k*[Bqr]^k
                     auto op2H = op2.H();
                     auto Sc2r = symbolic_prod<Tm>(op1,op2H);
                     Sc2r.ispins.push_back(std::make_tuple(1,ts,1));
                     // top_l sum_s <pq2||s1r2> ap[1]
                     symbolic_sum<Tm> top_l;
                     for(const auto& p : cindex_l){
                        auto op_l = symbolic_oper("l",'C',p);
                        double fac = (ts==0)? 1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
                        top_l.sum(fac*get_vint2e_su2(int2e,ts,p/2,kr2,s1/2,kq2), op_l);
                     }
                     auto Clc1 = symbolic_prod(top_l,symbolic_oper("c1",'I',0));
                     Clc1.ispins.push_back(std::make_tuple(1,0,1));
                     auto Clc1_task = symbolic_task<Tm>(Clc1);
                     auto Sc2r_task = symbolic_task<Tm>(Sc2r);
                     symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                           idx, counter, formulae, "Clc1_Sc2r(4dH)["+std::to_string(iqr)+"]");
                  }
               }
            }
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
         const double fac = std::sqrt(2.0);
         // c[c2R] = c[c2]*I[R]
         for(const auto& index : cindex_c2){
            auto Slc1_task = symbolic_compxwf_opS_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1,
                  int2e, index, ifkr, size, rank, ifdist1, ifdistc);
            if(Slc1_task.size() == 0) continue;
            int iformula = 1;
            auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
            symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                  idx, counter, formulae, "Slc1_Cc2r["+std::to_string(index)+"]");
         }
         // c[c2R] = I[c2]*c[R]
         // S[LC1]_1, S[LC1]_2
         for(const auto& index: cindex_r){
            symbolic_task<Tm> Slc1_task;
            int iproc = distribute1(ifkr,size,index);
            if(!ifdist1 or iproc==rank){
               // 1. S1*I2
               auto S1p = symbolic_prod<Tm>(symbolic_oper("l",'S',index), symbolic_oper("c1",'I',0));
               S1p.ispins.push_back(std::make_tuple(1,0,1)); 
               Slc1_task.append(S1p);
               // 2. I1*S2
               auto S2p = symbolic_prod<Tm>(symbolic_oper("l",'I',0), symbolic_oper("c1",'S',index));
               S2p.ispins.push_back(std::make_tuple(0,1,1));
               Slc1_task.append(S2p);
            }
            if(Slc1_task.size() == 0) continue;
            int iformula = 2;
            auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
            symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                  idx, counter, formulae, "Slc1_Cc2r(1,2)["+std::to_string(index)+"]");
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
               symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                     idx, counter, formulae, "Slc1_Cc2r(3)["+std::to_string(q1)+"]");
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
                  symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                        idx, counter, formulae, "Slc1_Cc2r(4)["+std::to_string(s1)+"]");
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
                  symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                        idx, counter, formulae, "Slc1_Cc2r(4H)["+std::to_string(s1)+"]");
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
               symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                     idx, counter, formulae, "Slc1_Cc2r(5c)["+std::to_string(index)+"]");
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
                  symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                        idx, counter, formulae, "Slc1_Cc2r(5d)["+std::to_string(isr)+"]");
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
               symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                     idx, counter, formulae, "Slc1_Cc2r(6c)["+std::to_string(index)+"]");
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
                     symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                           idx, counter, formulae, "Slc1_Cc2r(6d)["+std::to_string(iqs)+"]");
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
                     symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level, 
                           idx, counter, formulae, "Slc1_Cc2r(6dH)["+std::to_string(iqs)+"]");
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
         counter["H1"] = 0;
         auto Hlc1_task = symbolic_compxwf_opH_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1, 
               int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
         if(Hlc1_task.size() > 0){
            auto op2 = symbolic_prod<Tm>(symbolic_oper("c2",'I',0),symbolic_oper("r",'I',0));
            op2.ispins.push_back(std::make_tuple(0,0,0));
            auto Ic2r_task = symbolic_task<Tm>(op2);
            symbolic_Hops1ops2_su2(Hlc1_task, Ic2r_task, "H1", 0, 1.0, ifsave, print_level, 
                idx, counter, formulae, "Hlc1_Ic2r");   
         }
         // H[c2r]
         counter["H2"] = 0;
         auto Hc2r_task = symbolic_compxwf_opH_su2<Tm>(oplist_c2, oplist_r, "c2", "r", cindex_c2, cindex_r, 
               int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
         if(Hc2r_task.size() > 0){
            auto op1 = symbolic_prod<Tm>(symbolic_oper("l",'I',0),symbolic_oper("c1",'I',0));
            op1.ispins.push_back(std::make_tuple(0,0,0));
            auto Ilc1_task = symbolic_task<Tm>(op1);
            symbolic_Hops1ops2_su2(Ilc1_task, Hc2r_task, "H2", 0, 1.0, ifsave, print_level,
                  idx, counter, formulae, "Ilc1_Hc2r"); 
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
               auto Clc1_task = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
               auto Sc2r_task = symbolic_compxwf_opS_su2<Tm>(oplist_c2, oplist_r, "c2", "r", cindex_c2, cindex_r,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Sc2r_task.size() == 0) continue;
               double fac = std::sqrt(2.0);
               symbolic_Hops1ops2_su2(Clc1_task, Sc2r_task, "CS", 1, fac, ifsave, print_level,
                     idx, counter, formulae, "Clc1_Sc2r["+std::to_string(index)+"]");
            }
         }else{
            // right block only contains A,B operators
            assert(ifexistQ(oplist_r,'A') and ifexistQ(oplist_r,'B'));
            assert(csize_lc1 >= csize_c2r and ifdist1 and ifdistc);
            symbolic_Clc1_Sc2r_fromAB(oplist_l,oplist_r,oplist_c1,oplist_c2,
                  cindex_l,cindex_r,cindex_c1,cindex_c2,isym,ifkr,
                  int2e,size,rank,ifdist1,ifdistc,ifsave,print_level,
                  idx,counter,formulae);
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
               auto Slc1_task = symbolic_compxwf_opS_su2<Tm>(oplist_l, oplist_c1, "l", "c1", cindex_l, cindex_c1,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Slc1_task.size() == 0) continue;
               auto Cc2r_task = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
               double fac = std::sqrt(2.0);
               symbolic_Hops1ops2_su2(Slc1_task, Cc2r_task, "SC", 1, fac, ifsave, print_level,
                     idx, counter, formulae, "Slc1_Cc2r["+std::to_string(index)+"]");
            }
         }else{
            // left block only contains A,B operators
            assert(ifexistQ(oplist_l,'A') and ifexistQ(oplist_l,'B'));
            assert(csize_lc1 < csize_c2r and ifdist1 and ifdistc);
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
                  auto Alc1_task = symbolic_normxwf_opA_su2<Tm>("l", "c1", index, iformula);
                  auto Pc2r_task = symbolic_compxwf_opP_su2<Tm>("c2", "r", cindex_c2, cindex_r, int2e, index);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  symbolic_Hops1ops2_su2(Alc1_task, Pc2r_task, "AP", ts, fac, ifsave, print_level,
                     idx, counter, formulae, "Alc1_Pc2r["+std::to_string(index)+"]");
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
                  auto Blc1_task = symbolic_normxwf_opB_su2<Tm>("l", "c1", index, iformula);
                  auto Qc2r_task = symbolic_compxwf_opQ_su2<Tm>("c2", "r", cindex_c2, cindex_r, int2e, index);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  symbolic_Hops1ops2_su2(Blc1_task, Qc2r_task, "BQ", ts, fac, ifsave, print_level,
                       idx, counter, formulae, "Blc1_Qc2r["+std::to_string(index)+"]");
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
                  auto Plc1_task = symbolic_compxwf_opP_su2<Tm>("l", "c1", cindex_l, cindex_c1, int2e, index);
                  auto Ac2r_task = symbolic_normxwf_opA_su2<Tm>("c2", "r", index, iformula);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  symbolic_Hops1ops2_su2(Plc1_task, Ac2r_task, "PA", ts, fac, ifsave, print_level,
                       idx, counter, formulae, "Plc1_Ac2r["+std::to_string(index)+"]");
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
                  auto Qlc1_task = symbolic_compxwf_opQ_su2<Tm>("l", "c1", cindex_l, cindex_c1, int2e, index);
                  auto Bc2r_task = symbolic_normxwf_opB_su2<Tm>("c2", "r", index, iformula);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  symbolic_Hops1ops2_su2(Qlc1_task, Bc2r_task, "QB", ts, fac, ifsave, print_level,
                        idx, counter, formulae, "Qlc1_Bc2r["+std::to_string(index)+"]");
               } // iproc
            }

         } // ifNC
         return formulae;
      }

} // ctns

#endif
