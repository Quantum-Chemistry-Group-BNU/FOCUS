#ifndef SYMBOLIC_COMPXWF_OPS_H
#define SYMBOLIC_COMPXWF_OPS_H

namespace ctns{

   inline int get_formula_opS3(const std::string oplist2, const int kc1, const int kA2){
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
      return formula3;
   }

   inline int get_formula_opS4(const std::string oplist2, const int kc1, const int kB2){
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
      return formula4;
   }
   
   inline int get_formula_opS5(const std::string oplist1, const int kc2, const int kA1){
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
      return formula5;
   }

   inline int get_formula_opS6(const std::string oplist1, const int kc2, const int kB1){
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
      return formula6;
   }

   //------
   // opS3
   //------
   // sum_q aq^+[1]*Ppq[2]
   template <typename Tm>
      void symbolic_compxwf_opS3a(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            for(const auto& q : cindex1){
               int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
               int iproc = distribute2('P',ifkr,size,ipq,sorb);
               if(iproc == rank){
                  auto op1c = symbolic_oper(block1,'C',q);
                  auto op2P = symbolic_oper(block2,'P',ipq);
                  auto c1P2 = (p<q)? symbolic_prod<Tm>(op1c,op2P) : 
                     symbolic_prod<Tm>(op1c,op2P,-1.0);
                  formulae.append(c1P2);
               }
            } // q
         }else{
            // Kramers symmetry-adapted version 
            int kp = p/2, pa = p, pb = pa+1;
            for(const auto& qa : cindex1){
               int qb = qa+1, kq = qa/2;
               auto op1c_A = symbolic_oper(block1,'C',qa);
               auto op1a_A = op1c_A.H();
               auto op1c_B = op1c_A.K(1);
               auto op1a_B = op1a_A.K(1);
               int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
               int iproc_aa = distribute2('P',ifkr,size,ipq_aa,sorb);
               if(iproc_aa == rank){
                  auto op2P_AA = symbolic_oper(block2,'P',ipq_aa);
                  auto c1P2_AA = (kp<kq)? symbolic_prod<Tm>(op1c_A,op2P_AA) : 
                     symbolic_prod<Tm>(op1c_A,op2P_AA,-1.0); 
                  formulae.append(c1P2_AA);
               }
               int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
               int iproc_ab = distribute2('P',ifkr,size,ipq_ab,sorb);
               if(iproc_ab == rank){
                  auto op2P_AB = symbolic_oper(block2,'P',ipq_ab);
                  auto c1P2_AB = (kp<kq)? symbolic_prod<Tm>(op1c_B,op2P_AB) :
                     symbolic_prod<Tm>(op1c_B,op2P_AB.K(1),-1.0);
                  formulae.append(c1P2_AB);
               }
            } // qa
         } // ifkr
      }

   // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
   template <typename Tm>
      void symbolic_compxwf_opS3b(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& aindex2,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            for(const auto& isr : aindex2){
               auto sr = oper_unpack(isr);
               int s2 = sr.first;
               int r2 = sr.second;
               auto op2 = symbolic_oper(block2,'A',isr).H();
               auto sym_op2 = op2.get_qsym(isym);
               // sum_q <pq1||s2r2> aq[1]^+
               symbolic_sum<Tm> top1;
               for(const auto& q1 : cindex1){
                  auto op1 = symbolic_oper(block1,'C',q1);
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top1.sum(int2e.get(p,q1,s2,r2), op1);
               }
               formulae.append(top1,op2);
            }
         }else{
            // Kramers symmetry-adapted version 
            for(const auto& isr : aindex2){
               double wt = wfacAP(isr);
               auto sr = oper_unpack(isr);
               int s2 = sr.first , ks2 = s2/2, spin_s2 = s2%2, s2K = s2+1-2*spin_s2;
               int r2 = sr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;
               auto op2 = symbolic_oper(block2,'A',isr).H();
               auto op2K = op2.K(2-spin_s2-spin_r2);
               // sum_q <pq1||s2r2> aq[1]^+
               symbolic_sum<Tm> top1, top1K;
               for(const auto& q1a : cindex1){
                  int q1b = q1a+1;
                  auto op1c_A = symbolic_oper(block1,'C',q1a);
                  auto op1c_B = op1c_A.K(1);
                  top1.sum(int2e.get(p,q1a,s2,r2), op1c_A);
                  top1.sum(int2e.get(p,q1b,s2,r2), op1c_B);
                  top1K.sum(int2e.get(p,q1a,s2K,r2K), op1c_A);
                  top1K.sum(int2e.get(p,q1b,s2K,r2K), op1c_B);
               }
               top1.scale(wt);
               top1K.scale(wt);
               formulae.append(top1,op2);
               formulae.append(top1K,op2K);
            }
         } // ifkr
      }

   // sum_q aq[1]^+ (sum_sr <pq1||s2r2> Asr[2]^+)
   template <typename Tm>
      void symbolic_compxwf_opS3c(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& aindex2,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            // sum_q aq[1]^+
            for(const auto& q1 : cindex1){
               auto op1 = symbolic_oper(block1,'C',q1);
               auto sym_op1 = op1.get_qsym(isym);
               // sum_sr <pq1||s2r2> Asr[2]^+
               symbolic_sum<Tm> top2; 
               for(const auto& isr : aindex2){
                  auto sr = oper_unpack(isr);
                  int s2 = sr.first;
                  int r2 = sr.second;
                  auto op2 = symbolic_oper(block2,'A',isr).H();
                  auto sym_op2 = op2.get_qsym(isym);
                  // sum_q <pq1||s2r2> Asr[2]^+
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top2.sum(int2e.get(p,q1,s2,r2), op2);
               } // isr
               if(top2.size()>0) formulae.append(op1,top2);
            }
         }else{
            tools::exit("error: opS3c does not support ifkr=true yet!");
         } // ifkr
      }

   //------
   // opS4
   //------
   // sum_q aq[1]*Qpq[2]
   template <typename Tm>
      void symbolic_compxwf_opS4a(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         if(!ifkr){ 
            for(const auto& q : cindex1){
               int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
               int iproc = distribute2('Q',ifkr,size,ipq,sorb);
               if(iproc == rank){
                  auto op1c = symbolic_oper(block1,'C',q);
                  auto op1a = op1c.H();
                  auto op2Q = symbolic_oper(block2,'Q',ipq);
                  auto a1Q2 = (p<q)? symbolic_prod<Tm>(op1a,op2Q) : 
                     symbolic_prod<Tm>(op1a,op2Q.H());
                  formulae.append(a1Q2);
               }
            } // q
         }else{
            // Kramers symmetry-adapted version 
            int kp = p/2, pa = p, pb = pa+1;
            for(const auto& qa : cindex1){
               int qb = qa+1, kq = qa/2;
               auto op1c_A = symbolic_oper(block1,'C',qa);
               auto op1a_A = op1c_A.H();
               auto op1c_B = op1c_A.K(1);
               auto op1a_B = op1a_A.K(1);
               int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
               int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,sorb);
               if(iproc_aa == rank){
                  auto op2Q_AA = symbolic_oper(block2,'Q',ipq_aa);
                  auto a1Q2_AA = (kp<kq)? symbolic_prod<Tm>(op1a_A,op2Q_AA) : 
                     symbolic_prod<Tm>(op1a_A,op2Q_AA.H());
                  formulae.append(a1Q2_AA);
               }
               int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
               int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,sorb);
               if(iproc_ab == rank){
                  auto op2Q_AB = symbolic_oper(block2,'Q',ipq_ab);
                  auto a1Q2_AB = (kp<kq)? symbolic_prod<Tm>(op1a_B,op2Q_AB) :
                     symbolic_prod<Tm>(op1a_B,op2Q_AB.K(1).H());
                  formulae.append(a1Q2_AB);
               }
            } // qa
         } // ifkr
      }

   // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
   template <typename Tm>
      void symbolic_compxwf_opS4b(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& bindex2,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            for(const auto& iqr : bindex2){
               auto qr = oper_unpack(iqr);
               int q2 = qr.first;
               int r2 = qr.second;
               auto op2 = symbolic_oper(block2,'B',iqr);
               auto sym_op2 = op2.get_qsym(isym);
               // sum_s <pq2||s1r2> as[1]
               symbolic_sum<Tm> top1;
               for(const auto& s1 : cindex1){
                  auto op1 = symbolic_oper(block1,'C',s1).H();
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top1.sum(int2e.get(p,q2,s1,r2), op1);
               }
               formulae.append(top1,op2);
               // Hermitian part: q2<->r2
               if(q2 == r2) continue;	    
               auto op2H = op2.H();
               auto sym_op2H = op2H.get_qsym(isym);
               symbolic_sum<Tm> top1H;
               for(const auto& s1 : cindex1){
                  auto op1 = symbolic_oper(block1,'C',s1).H();
                  auto sym_op1 = op1.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2H) continue;
                  top1H.sum(int2e.get(p,r2,s1,q2), op1);
               }
               formulae.append(top1H,op2H);
            }
         }else{
            // Kramers symmetry-adapted version 
            for(const auto& iqr : bindex2){
               auto qr = oper_unpack(iqr);
               int q2 = qr.first , kq2 = q2/2, spin_q2 = q2%2, q2K = q2+1-2*spin_q2;
               int r2 = qr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;;
               auto op2 = symbolic_oper(block2,'B',iqr);
               auto op2K =  op2.K(2-spin_q2-spin_r2);
               // sum_s <pq2||s1r2> as[1]
               symbolic_sum<Tm> top1, top1K;
               for(const auto& s1a : cindex1){
                  int s1b = s1a+1;
                  auto op1a_A = symbolic_oper(block1,'C',s1a).H();
                  auto op1a_B = op1a_A.K(1);
                  top1.sum(int2e.get(p,q2,s1a,r2), op1a_A);
                  top1.sum(int2e.get(p,q2,s1b,r2), op1a_B);
                  top1K.sum(int2e.get(p,q2K,s1a,r2K), op1a_A);
                  top1K.sum(int2e.get(p,q2K,s1b,r2K), op1a_B);
               }
               formulae.append(top1,op2);
               formulae.append(top1K,op2K);
               // Hermitian part: q2<->r2
               if(kq2 == kr2) continue;
               symbolic_sum<Tm> top1H, top1KH;
               for(const auto& s1a : cindex1){
                  int s1b = s1a+1;
                  auto op1a_A = symbolic_oper(block1,'C',s1a).H();
                  auto op1a_B = op1a_A.K(1);
                  top1H.sum(int2e.get(p,r2,s1a,q2), op1a_A);
                  top1H.sum(int2e.get(p,r2,s1b,q2), op1a_B);
                  top1KH.sum(int2e.get(p,r2K,s1a,q2K), op1a_A);
                  top1KH.sum(int2e.get(p,r2K,s1b,q2K), op1a_B);
               }
               formulae.append(top1H,op2.H());
               formulae.append(top1KH,op2K.H());
            }
         } // ifkr
      }

   // sum_s as[1] (sum_qr <pq2||s1r2> aq[2]^+ar[2])
   template <typename Tm>
      void symbolic_compxwf_opS4c(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& bindex2,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            // sum_s as[1] 
            for(const auto& s1 : cindex1){
               auto op1 = symbolic_oper(block1,'C',s1).H();
               auto sym_op1 = op1.get_qsym(isym);
               // sum_qr <pq2||s1r2> aq[2]^+ar[2]
               symbolic_sum<Tm> top2, top2H;
               for(const auto& iqr : bindex2){
                  auto qr = oper_unpack(iqr);
                  int q2 = qr.first;
                  int r2 = qr.second;
                  double wqr = (q2==r2)? 0.5 : 1.0;
                  auto op2 = symbolic_oper(block2,'B',iqr);
                  auto sym_op2 = op2.get_qsym(isym);
                  auto op2H = op2.H();
                  auto sym_op2H = op2H.get_qsym(isym);
                  // sum_qr <pq2||s1r2> aq[2]^+ar[2]
                  if(sym_op == sym_op1 + sym_op2){
                     top2.sum(wqr*int2e.get(p,q2,s1,r2), op2);
                  }
                  if(sym_op == sym_op1 + sym_op2H){
                     top2H.sum(wqr*int2e.get(p,r2,s1,q2), op2H);
                  }
               }
               if(top2.size()>0) formulae.append(op1,top2);
               if(top2H.size()>0) formulae.append(op1,top2H);
            }
         }else{
            tools::exit("error: opS4c does not support ifkr=true yet!");
         } // ifkr
      }

   //------
   // opS5
   //------
   // sum_q Ppq[1]*aq^+[2]
   template <typename Tm>
      void symbolic_compxwf_opS5a(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            for(const auto& q : cindex2){
               int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
               int iproc = distribute2('P',ifkr,size,ipq,sorb);
               if(iproc == rank){
                  auto op2c = symbolic_oper(block2,'C',q);
                  auto op1P = symbolic_oper(block1,'P',ipq);
                  auto P1c2 = (p<q)? symbolic_prod<Tm>(op1P,op2c) : 
                     symbolic_prod<Tm>(op1P,op2c,-1.0);
                  formulae.append(P1c2);
               }
            } // q
         }else{
            // Kramers symmetry-adapted version 
            int kp = p/2, pa = p, pb = pa+1;
            for(const auto& qa : cindex2){
               int qb = qa+1, kq = qa/2;
               auto op2c_A = symbolic_oper(block2,'C',qa);
               auto op2a_A = op2c_A.H();
               auto op2c_B = op2c_A.K(1);
               auto op2a_B = op2a_A.K(1);
               int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
               int iproc_aa = distribute2('P',ifkr,size,ipq_aa,sorb);
               if(iproc_aa == rank){
                  auto op1P_AA = symbolic_oper(block1,'P',ipq_aa);
                  auto P1c2_AA = (kp<kq)? symbolic_prod<Tm>(op1P_AA,op2c_A) : 
                     symbolic_prod<Tm>(op1P_AA,op2c_A,-1.0); 
                  formulae.append(P1c2_AA);
               } 
               int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
               int iproc_ab = distribute2('P',ifkr,size,ipq_ab,sorb);
               if(iproc_ab == rank){
                  auto op1P_AB = symbolic_oper(block1,'P',ipq_ab);
                  auto P1c2_AB = (kp<kq)? symbolic_prod<Tm>(op1P_AB,op2c_B) :
                     symbolic_prod<Tm>(op1P_AB.K(1),op2c_B,-1.0);
                  formulae.append(P1c2_AB);
               }
            } // qa
         } // ifkr
      }

   // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
   template <typename Tm>
      void symbolic_compxwf_opS5b(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& aindex1,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            for(const auto& isr : aindex1){
               auto sr = oper_unpack(isr);
               int s1 = sr.first;
               int r1 = sr.second;
               auto op1 = symbolic_oper(block1,'A',isr).H();
               auto sym_op1 = op1.get_qsym(isym);
               // sum_q <pq2||s1r1> aq[2]^+
               symbolic_sum<Tm> top2;
               for(const auto& q2 : cindex2){
                  auto op2 = symbolic_oper(block2,'C',q2);
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top2.sum(int2e.get(p,q2,s1,r1), op2);
               }
               formulae.append(op1,top2);
            }
         }else{
            // Kramers symmetry-adapted version 
            for(const auto& isr : aindex1){
               double wt =  wfacAP(isr);
               auto sr = oper_unpack(isr);
               int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
               int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2, r1K = r1+1-2*spin_r1;
               auto op1 = symbolic_oper(block1,'A',isr).H();
               auto op1K = op1.K(2-spin_s1-spin_r1);
               // sum_q <pq2||s1r1> aq[2]^+
               symbolic_sum<Tm> top2, top2K;
               for(const auto& q2a : cindex2){
                  int q2b = q2a+1;
                  auto op2c_A = symbolic_oper(block2,'C',q2a);
                  auto op2c_B = op2c_A.K(1);
                  top2.sum(int2e.get(p,q2a,s1,r1), op2c_A);
                  top2.sum(int2e.get(p,q2b,s1,r1), op2c_B);
                  top2K.sum(int2e.get(p,q2a,s1K,r1K), op2c_A);
                  top2K.sum(int2e.get(p,q2b,s1K,r1K), op2c_B);
               }
               top2.scale(wt);
               top2K.scale(wt);
               formulae.append(op1,top2);
               formulae.append(op1K,top2K);
            }
         } // ifkr
      }

   // sum_q (sum_sr Asr[1]^+ <pq2||s1r1>) aq[2]^+
   template <typename Tm>
      void symbolic_compxwf_opS5c(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& aindex1,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            // sum_q aq[2]^+
            for(const auto& q2 : cindex2){
               auto op2 = symbolic_oper(block2,'C',q2);
               auto sym_op2 = op2.get_qsym(isym);
               // sum_sr Asr[1]^+ <pq2||s1r1>
               symbolic_sum<Tm> top1;
               for(const auto& isr : aindex1){
                  auto sr = oper_unpack(isr);
                  int s1 = sr.first;
                  int r1 = sr.second;
                  auto op1 = symbolic_oper(block1,'A',isr).H();
                  auto sym_op1 = op1.get_qsym(isym);
                  // sum_sr Asr[1]^+ <pq2||s1r1>
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top1.sum(int2e.get(p,q2,s1,r1), op1);
               } // isr
               if(top1.size()>0) formulae.append(top1,op2);
            }
         }else{
            tools::exit("error: opS5c does not support ifkr=true yet!");
         } // ifkr
      }

   //------
   // opS6
   //------
   // sum_q Qpq^[1]*aq[2]
   template <typename Tm>
      void symbolic_compxwf_opS6a(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            for(const auto& q : cindex2){
               int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
               int iproc = distribute2('Q',ifkr,size,ipq,sorb);
               if(iproc == rank){
                  auto op2c = symbolic_oper(block2,'C',q);
                  auto op2a = op2c.H();
                  auto op1Q = symbolic_oper(block1,'Q',ipq);
                  auto Q1a2 = (p<q)? symbolic_prod<Tm>(op1Q,op2a) : 
                     symbolic_prod<Tm>(op1Q.H(),op2a);
                  formulae.append(Q1a2);
               }
            } // q
         }else{
            // Kramers symmetry-adapted version 
            int kp = p/2, pa = p, pb = pa+1;
            for(const auto& qa : cindex2){
               int qb = qa+1, kq = qa/2;
               auto op2c_A = symbolic_oper(block2,'C',qa);
               auto op2a_A = op2c_A.H();
               auto op2c_B = op2c_A.K(1);
               auto op2a_B = op2a_A.K(1);
               int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
               int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,sorb);
               if(iproc_aa == rank){
                  auto op1Q_AA = symbolic_oper(block1,'Q',ipq_aa);
                  auto Q1a2_AA = (kp<kq)? symbolic_prod<Tm>(op1Q_AA,op2a_A) : 
                     symbolic_prod<Tm>(op1Q_AA.H(),op2a_A);
                  formulae.append(Q1a2_AA);
               }
               int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
               int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,sorb);
               if(iproc_ab == rank){
                  auto op1Q_AB = symbolic_oper(block1,'Q',ipq_ab);
                  auto Q1a2_AB = (kp<kq)? symbolic_prod<Tm>(op1Q_AB,op2a_B) :
                     symbolic_prod<Tm>(op1Q_AB.K(1).H(),op2a_B);
                  formulae.append(Q1a2_AB);
               }
            } // qa
         } // ifkr
      }

   // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
   template <typename Tm>
      void symbolic_compxwf_opS6b(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& bindex1,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            for(const auto& iqs : bindex1){
               auto qs = oper_unpack(iqs);
               int q1 = qs.first;
               int s1 = qs.second;
               auto op1 = symbolic_oper(block1,'B',iqs);
               auto sym_op1 = op1.get_qsym(isym);
               // sum_r -<pq1||s1r2> ar[2]
               symbolic_sum<Tm> top2;
               for(const auto& r2 : cindex2){
                  auto op2 = symbolic_oper(block2,'C',r2).H();
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op != sym_op1 + sym_op2) continue;
                  top2.sum(-int2e.get(p,q1,s1,r2), op2);
               }
               formulae.append(op1,top2);
               // Hermitian part: q1<->r1
               if(q1 == s1) continue;
               auto op1H = op1.H();
               auto sym_op1H = op1H.get_qsym(isym);
               symbolic_sum<Tm> top2H;
               for(const auto& r2 : cindex2){
                  auto op2 = symbolic_oper(block2,'C',r2).H();
                  auto sym_op2 = op2.get_qsym(isym);
                  if(sym_op != sym_op1H + sym_op2) continue;
                  top2H.sum(-int2e.get(p,s1,q1,r2), op2);
               }
               formulae.append(op1H,top2H);
            }
         }else{
            // Kramers symmetry-adapted version 
            for(const auto& iqs : bindex1){
               auto qs = oper_unpack(iqs);
               int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2, q1K = q1+1-2*spin_q1;
               int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
               auto op1 = symbolic_oper(block1,'B',iqs);
               auto op1K = op1.K(2-spin_q1-spin_s1);
               // sum_r -<pq1||s1r2> ar[2]
               symbolic_sum<Tm> top2, top2K;
               for(const auto& r2a : cindex2){
                  int r2b = r2a+1;
                  auto op2a_A = symbolic_oper(block2,'C',r2a).H();
                  auto op2a_B = op2a_A.K(1);
                  top2.sum(-int2e.get(p,q1,s1,r2a), op2a_A);
                  top2.sum(-int2e.get(p,q1,s1,r2b), op2a_B);
                  top2K.sum(-int2e.get(p,q1K,s1K,r2a), op2a_A);
                  top2K.sum(-int2e.get(p,q1K,s1K,r2b), op2a_B);
               }
               formulae.append(op1,top2);
               formulae.append(op1K,top2K);
               // Hermitian part: q1<->s1
               if(kq1 == ks1) continue;
               symbolic_sum<Tm> top2H, top2KH;
               for(const auto& r2a : cindex2){
                  int r2b = r2a+1;
                  auto op2a_A = symbolic_oper(block2,'C',r2a).H();
                  auto op2a_B = op2a_A.K(1);
                  top2H.sum(-int2e.get(p,s1,q1,r2a), op2a_A);
                  top2H.sum(-int2e.get(p,s1,q1,r2b), op2a_B);
                  top2KH.sum(-int2e.get(p,s1K,q1K,r2a), op2a_A);
                  top2KH.sum(-int2e.get(p,s1K,q1K,r2b), op2a_B);
               }
               formulae.append(op1.H(),top2H);
               formulae.append(op1K.H(),top2KH);
            }
         } // ifkr
      }

   // sum_r (sum_qs aq[1]^+as[1] -<pq1||s1r2>) ar[2]
   template <typename Tm>
      void symbolic_compxwf_opS6c(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const int isym,
            const bool ifkr,
            const std::vector<int>& bindex1,
            symbolic_task<Tm>& formulae){
         if(!ifkr){
            auto sym_op = get_qsym_opS(isym, p);
            // sum_r ar[2]
            for(const auto& r2 : cindex2){
               auto op2 = symbolic_oper(block2,'C',r2).H();
               auto sym_op2 = op2.get_qsym(isym);
               // sum_qs aq[1]^+as[1] -<pq1||s1r2>
               symbolic_sum<Tm> top1, top1H; 
               for(const auto& iqs : bindex1){
                  auto qs = oper_unpack(iqs);
                  int q1 = qs.first;
                  int s1 = qs.second;
                  double wqs = (q1==s1)? 0.5 : 1.0;
                  auto op1 = symbolic_oper(block1,'B',iqs);
                  auto sym_op1 = op1.get_qsym(isym);
                  auto op1H = op1.H();
                  auto sym_op1H = op1H.get_qsym(isym);
                  // sum_qs aq[1]^+as[1] -<pq1||s1r2>
                  if(sym_op == sym_op1 + sym_op2){
                     top1.sum(-wqs*int2e.get(p,q1,s1,r2), op1);
                  }
                  if(sym_op == sym_op1H + sym_op2){
                     top1H.sum(-wqs*int2e.get(p,s1,q1,r2), op1H);
                  }
               }
               if(top1.size()>0) formulae.append(top1,op2);
               if(top1H.size()>0) formulae.append(top1H,op2);
            }
         }else{
            tools::exit("error: opS6c does not support ifkr=true yet!");
         } // ifkr
      }

} // ctns

#endif
