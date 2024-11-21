#ifndef SYMBOLIC_COMPXWF_OPS_SU2_H
#define SYMBOLIC_COMPXWF_OPS_SU2_H

namespace ctns{

   // integrals for Ppq
   template <typename Tm>
      Tm get_xint2e_su2(const integral::two_body<Tm>& int2e,
            const int ts,
            const int kp,
            const int kq,
            const int ks,
            const int kr){
         if(ts == 0){
            Tm fac = (ks==kr)? 0.5 : 1.0;
            return -fac*(int2e.get(2*kp,2*kq+1,2*ks,2*kr+1) + int2e.get(2*kp,2*kq+1,2*kr,2*ks+1));
         }else{
            return int2e.get(2*kp,2*kq,2*ks,2*kr);
         }
      }

   // integrals for Qps
   template <typename Tm>
      Tm get_vint2e_su2(const integral::two_body<Tm>& int2e,
            const int ts,
            const int kp,
            const int kq,
            const int ks,
            const int kr){
         if(ts == 0){
            return int2e.get(2*kp,2*kq,2*ks,2*kr) + int2e.get(2*kp,2*kq+1,2*ks,2*kr+1);
         }else{
            return int2e.get(2*kp,2*kq+1,2*ks+1,2*kr);
         }
      }

   //------
   // opS3
   //------
   // sum_q aq^+[1]*Ppq[2]
   template <typename Tm>
      void symbolic_compxwf_opS3a_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         // sum_q aq^+[1]*Ppq[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex1){
            int qb = qa+1, kq = qa/2;
            auto op1c = symbolic_oper(block1,'C',qa);
            // triplet Ppq:
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
            int iproc_aa = distribute2('P',ifkr,size,ipq_aa,sorb);
            if(iproc_aa == rank){
               auto op2P_AA = symbolic_oper(block2,'P',ipq_aa);
               double fac = (kp<kq)? std::sqrt(3.0/2.0) : -std::sqrt(3.0/2.0); // Ppq1 = -Pqp1
               auto c1P2_AA = symbolic_prod<Tm>(op1c,op2P_AA,fac); 
               c1P2_AA.ispins.push_back(std::make_tuple(1,2,1)); 
               formulae.append(c1P2_AA);
            }
            // singlet Ppq:
            int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
            int iproc_ab = distribute2('P',ifkr,size,ipq_ab,sorb);
            if(iproc_ab == rank){
               auto op2P_AB = symbolic_oper(block2,'P',ipq_ab);
               double fac = -1.0/std::sqrt(2.0); // Ppq0 = Pqp0
               auto c1P2_AB = symbolic_prod<Tm>(op1c,op2P_AB,fac);
               c1P2_AB.ispins.push_back(std::make_tuple(1,0,1));
               formulae.append(c1P2_AB);
            }
         } // qa
      }

   // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
   template <typename Tm>
      void symbolic_compxwf_opS3b_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& aindex2,
            symbolic_task<Tm>& formulae){
         // loop over Asr
         for(const auto& isr : aindex2){
            auto sr = oper_unpack(isr);
            int s2 = sr.first, ks = s2/2;
            int r2 = sr.second, kr = r2/2;
            int spin_s2 = s2%2, spin_r2 = r2%2;
            int ts = (spin_s2!=spin_r2)? 0 : 2;
            auto op2 = symbolic_oper(block2,'A',isr).H();
            // sum_q <pq1||s2r2> aq[1]^+
            symbolic_sum<Tm> top1;
            for(const auto& q1 : cindex1){
               auto op1 = symbolic_oper(block1,'C',q1);
               double fac = (ts==0)? -1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
               top1.sum(fac*get_xint2e_su2(int2e,ts,p/2,q1/2,ks,kr), op1);
            }
            auto op12 = symbolic_prod(top1,op2);
            op12.ispins.push_back(std::make_tuple(1,ts,1));
            formulae.append(op12);
         }
      }

   // sum_q aq[1]^+ sum_sr <pq1||s2r2> Asr[2]^+
   template <typename Tm>
      void symbolic_compxwf_opS3c_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& aindex2,
            symbolic_task<Tm>& formulae){
         // sum_q aq[1]^+ sum_sr <pq1||s2r2> Asr[2]^+
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex1){
            int qb = qa+1, kq = qa/2;
            auto op1c = symbolic_oper(block1,'C',qa);
            // sum_sr <pq1||s2r2> Asr[2]^+
            symbolic_sum<Tm> top2s, top2t;
            for(const auto& isr : aindex2){
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               int spin_s2 = s2%2, spin_r2 = r2%2;
               int ts = (spin_s2!=spin_r2)? 0 : 2;
               auto op2 = symbolic_oper(block2,'A',isr).H();
               // sum_sr <pq1||s2r2> Asr[2]^+
               if(ts == 0){
                  double fac = -1.0/std::sqrt(2.0);
                  top2s.sum(fac*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
               }else{
                  double fac = std::sqrt(3.0/2.0);
                  top2t.sum(fac*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
               }
            }
            if(top2s.size()>0){ 
               auto op12 = symbolic_prod(op1c,top2s);
               op12.ispins.push_back(std::make_tuple(1,0,1));
               formulae.append(op12);
            }
            if(top2t.size()>0){
               auto op12 = symbolic_prod(op1c,top2t);
               op12.ispins.push_back(std::make_tuple(1,2,1));
               formulae.append(op12);
            }
         } // qa
      }

   //------
   // opS4
   //------
   // sum_q aq[1]*Qpq[2]
   template <typename Tm>
      void symbolic_compxwf_opS4a_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         // sum_q aq[1]*Qpq[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex1){
            int qb = qa+1, kq = qa/2;
            auto op1a = symbolic_oper(block1,'C',qa).H();
            // singlet Qpq
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
            int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,sorb);
            if(iproc_aa == rank){
               auto op2Q_AA = symbolic_oper(block2,'Q',ipq_aa);
               double fac = 1.0/std::sqrt(2.0); // singlet case
               auto a1Q2_AA = (kp<kq)? symbolic_prod<Tm>(op1a,op2Q_AA,fac) : 
                  symbolic_prod<Tm>(op1a,op2Q_AA.H(),fac);
               a1Q2_AA.ispins.push_back(std::make_tuple(1,0,1));
               formulae.append(a1Q2_AA);
            }
            // triplet Qpq
            int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
            int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,sorb);
            if(iproc_ab == rank){
               auto op2Q_AB = symbolic_oper(block2,'Q',ipq_ab);
               double fac = -std::sqrt(3.0/2.0); // triplet case: additional sign exist - different from nonSU2 case !
               auto a1Q2_AB = (kp<kq)? symbolic_prod<Tm>(op1a,op2Q_AB,fac) :
                  symbolic_prod<Tm>(op1a,op2Q_AB.H(),-fac); // Qpq^k = (-1)^k (Qqp^k)^d
               a1Q2_AB.ispins.push_back(std::make_tuple(1,2,1));
               formulae.append(a1Q2_AB);
            }
         } // qa
      }

   // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
   template <typename Tm>
      void symbolic_compxwf_opS4b_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& bindex2,
            symbolic_task<Tm>& formulae){
         // loop over Bqr
         for(const auto& iqr : bindex2){
            auto qr = oper_unpack(iqr);
            int q2 = qr.first, kq2 = q2/2;
            int r2 = qr.second, kr2 = r2/2;
            int spin_q2 = q2%2, spin_r2 = r2%2;
            int ts = (spin_q2!=spin_r2)? 2 : 0;
            auto op2 = symbolic_oper(block2,'B',iqr);
            // sum_s <pq2||s1r2> as[1]
            symbolic_sum<Tm> top1;
            for(const auto& s1 : cindex1){
               auto op1 = symbolic_oper(block1,'C',s1).H();
               double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
               top1.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq2,s1/2,kr2), op1);
            }
            auto op12 = symbolic_prod(top1,op2);
            op12.ispins.push_back(std::make_tuple(1,ts,1));
            formulae.append(op12);
            // Hermitian part: q2<->r2
            if(kq2 == kr2) continue;
            // We use [Brq]^k = (-1)^k*[Bqr]^k
            auto op2H = op2.H();
            symbolic_sum<Tm> top1H;
            for(const auto& s1 : cindex1){
               auto op1 = symbolic_oper(block1,'C',s1).H();
               double fac = (ts==0)? 1.0/std::sqrt(2.0) : +std::sqrt(3.0/2.0);
               top1H.sum(fac*get_vint2e_su2(int2e,ts,p/2,kr2,s1/2,kq2), op1);
            }
            auto op12H = symbolic_prod(top1H,op2H);
            op12H.ispins.push_back(std::make_tuple(1,ts,1));
            formulae.append(op12H);
         }
      }

   // sum_s as[1] sum_qr <pq2||s1r2> aq[2]^+ar[2]
   template <typename Tm>
      void symbolic_compxwf_opS4c_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& bindex2,
            symbolic_task<Tm>& formulae){
         // sum_s as[1] sum_qr <pq2||s1r2> aq[2]^+ar[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& sa : cindex1){
            int sb = sa+1, ks = sa/2;
            auto op1a = symbolic_oper(block1,'C',sa).H();
            // sum_qr <pq2||s1r2> aq[2]^+ar[2]
            symbolic_sum<Tm> top2s, top2t, top2sH, top2tH;
            for(const auto& iqr : bindex2){
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq = q2/2;
               int r2 = qr.second, kr = r2/2;
               int spin_q2 = q2%2, spin_r2 = r2%2;
               int ts = (spin_q2!=spin_r2)? 2 : 0;
               auto op2 = symbolic_oper(block2,'B',iqr);
               // sum_s <pq2||s1r2> as[1]
               if(ts == 0){
                  double fac = 1.0/std::sqrt(2.0);
                  top2s.sum(fac*get_vint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
               }else{
                  double fac = -std::sqrt(3.0/2.0);
                  top2t.sum(fac*get_vint2e_su2(int2e,ts,kp,kq,ks,kr), op2);
               }
               // Hermitian part: q2<->r2
               if(kq == kr) continue;
               // We use [Brq]^k = (-1)^k*[Bqr]^k
               auto op2H = op2.H();
               if(ts == 0){
                  double fac = 1.0/std::sqrt(2.0);
                  top2sH.sum(fac*get_vint2e_su2(int2e,ts,kp,kr,ks,kq), op2H);
               }else{
                  double fac = +std::sqrt(3.0/2.0);
                  top2tH.sum(fac*get_vint2e_su2(int2e,ts,kp,kr,ks,kq), op2H);
               }
            }
            if(top2s.size()>0){ 
               auto op12 = symbolic_prod(op1a,top2s);
               op12.ispins.push_back(std::make_tuple(1,0,1));
               formulae.append(op12);
            }
            if(top2t.size()>0){
               auto op12 = symbolic_prod(op1a,top2t);
               op12.ispins.push_back(std::make_tuple(1,2,1));
               formulae.append(op12);
            }
            if(top2sH.size()>0){ 
               auto op12 = symbolic_prod(op1a,top2sH);
               op12.ispins.push_back(std::make_tuple(1,0,1));
               formulae.append(op12);
            }
            if(top2tH.size()>0){
               auto op12 = symbolic_prod(op1a,top2tH);
               op12.ispins.push_back(std::make_tuple(1,2,1));
               formulae.append(op12);
            }
         } // qa
      }

   //------
   // opS5
   //------
   // sum_q Ppq[1]*aq^+[2]
   template <typename Tm>
      void symbolic_compxwf_opS5a_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         // sum_q Ppq[1]*aq^+[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex2){
            int qb = qa+1, kq = qa/2;
            auto op2c = symbolic_oper(block2,'C',qa);
            // triplet Ppq: 
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
            int iproc_aa = distribute2('P',ifkr,size,ipq_aa,sorb);
            if(iproc_aa == rank){
               auto op1P_AA = symbolic_oper(block1,'P',ipq_aa);
               double fac = (kp<kq)? -std::sqrt(3.0/2.0) : std::sqrt(3.0/2.0); // Ppq1 = -Pqp1
               auto P1c2_AA = symbolic_prod<Tm>(op1P_AA,op2c,fac);
               P1c2_AA.ispins.push_back(std::make_tuple(2,1,1));
               formulae.append(P1c2_AA);
            }
            // singlet Ppq:
            int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
            int iproc_ab = distribute2('P',ifkr,size,ipq_ab,sorb);
            if(iproc_ab == rank){
               auto op1P_AB = symbolic_oper(block1,'P',ipq_ab);
               double fac = -1.0/std::sqrt(2.0); // Ppq0 = Pqp0
               auto P1c2_AB = symbolic_prod<Tm>(op1P_AB,op2c,fac);
               P1c2_AB.ispins.push_back(std::make_tuple(0,1,1));
               formulae.append(P1c2_AB);
            }
         } // qa
      }

   // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
   template <typename Tm>
      void symbolic_compxwf_opS5b_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& aindex1,
            symbolic_task<Tm>& formulae){
         // loop over Asr
         for(const auto& isr : aindex1){
            auto sr = oper_unpack(isr);
            int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2;
            int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2;
            int ts = (spin_s1!=spin_r1)? 0 : 2;
            auto op1 = symbolic_oper(block1,'A',isr).H();
            // sum_q <pq2||s1r1> aq[2]^+
            symbolic_sum<Tm> top2;
            for(const auto& q2a : cindex2){
               auto op2c = symbolic_oper(block2,'C',q2a);
               double fac = (ts==0)? -1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
               top2.sum(fac*get_xint2e_su2(int2e,ts,p/2,q2a/2,ks1,kr1), op2c);
            }
            auto op12 = symbolic_prod(op1,top2);
            op12.ispins.push_back(std::make_tuple(ts,1,1));
            formulae.append(op12);
         }
      }

   // sum_q (sum_sr <pq2||s1r1> Asr[1]^+) aq[2]^+
   template <typename Tm>
      void symbolic_compxwf_opS5c_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& aindex1,
            symbolic_task<Tm>& formulae){
         // sum_q aq[2]^+
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex2){
            int qb = qa+1, kq = qa/2;
            auto op2c = symbolic_oper(block2,'C',qa);
            // sum_sr <pq2||s1r1> Asr[1]^+
            symbolic_sum<Tm> top1s, top1t;   
            for(const auto& isr : aindex1){
               auto sr = oper_unpack(isr);
               int s1 = sr.first , ks = s1/2, spin_s1 = s1%2;
               int r1 = sr.second, kr = r1/2, spin_r1 = r1%2;
               int ts = (spin_s1!=spin_r1)? 0 : 2;
               auto op1 = symbolic_oper(block1,'A',isr).H();
               // sum_q <pq2||s1r1> aq[2]^+
               if(ts == 0){
                  double fac = -1.0/std::sqrt(2.0);
                  top1s.sum(fac*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op1);
               }else{
                  double fac = -std::sqrt(3.0/2.0);
                  top1t.sum(fac*get_xint2e_su2(int2e,ts,kp,kq,ks,kr), op1);
               }
            }
            if(top1s.size()>0){
               auto op12 = symbolic_prod(top1s,op2c);
               op12.ispins.push_back(std::make_tuple(0,1,1));
               formulae.append(op12);
            }
            if(top1t.size()>0){
               auto op12 = symbolic_prod(top1t,op2c);
               op12.ispins.push_back(std::make_tuple(2,1,1));
               formulae.append(op12);
            }
         }
      }

   //------
   // opS6
   //------
   // sum_q Qpq^[1]*aq[2]
   template <typename Tm>
      void symbolic_compxwf_opS6a_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const int p,
            const bool ifkr,
            const int sorb,
            const int size,
            const int rank,
            symbolic_task<Tm>& formulae){
         // sum_q Qpq^[1]*aq[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& qa : cindex2){
            int qb = qa+1, kq = qa/2;
            auto op2a = symbolic_oper(block2,'C',qa).H();
            // singlet Qpq
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
            int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,sorb);
            if(iproc_aa == rank){
               auto op1Q_AA = symbolic_oper(block1,'Q',ipq_aa);
               double fac = 1.0/std::sqrt(2.0); // singlet
               auto Q1a2_AA = (kp<kq)? symbolic_prod<Tm>(op1Q_AA,op2a,fac) : 
                  symbolic_prod<Tm>(op1Q_AA.H(),op2a,fac);
               Q1a2_AA.ispins.push_back(std::make_tuple(0,1,1));
               formulae.append(Q1a2_AA);
            }
            // triplet Qpq
            int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
            int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,sorb);
            if(iproc_ab == rank){
               auto op1Q_AB = symbolic_oper(block1,'Q',ipq_ab);
               double fac = std::sqrt(3.0/2.0); // triplet case: additional sign exist - different from nonSU2 case !
               auto Q1a2_AB = (kp<kq)? symbolic_prod<Tm>(op1Q_AB,op2a,fac) :
                  symbolic_prod<Tm>(op1Q_AB.H(),op2a,-fac);
               Q1a2_AB.ispins.push_back(std::make_tuple(2,1,1));
               formulae.append(Q1a2_AB);
            }
         } // qa
      }

   // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
   template <typename Tm>
      void symbolic_compxwf_opS6b_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& bindex1,
            symbolic_task<Tm>& formulae){
         // loop over Bqs
         for(const auto& iqs : bindex1){
            auto qs = oper_unpack(iqs);
            int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2;
            int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2;
            int ts = (spin_q1!=spin_s1)? 2 : 0;
            auto op1 = symbolic_oper(block1,'B',iqs);
            // sum_r -<pq1||s1r2> ar[2]
            symbolic_sum<Tm> top2;
            for(const auto& r2a : cindex2){
               auto op2 = symbolic_oper(block2,'C',r2a).H();
               double fac = (ts==0)? 1.0/std::sqrt(2.0) : std::sqrt(3.0/2.0);
               top2.sum(fac*get_vint2e_su2(int2e,ts,p/2,kq1,r2a/2,ks1), op2);
            }
            auto op12 = symbolic_prod(op1,top2);
            op12.ispins.push_back(std::make_tuple(ts,1,1));
            formulae.append(op12);
            // Hermitian part: q1<->s1
            if(kq1 == ks1) continue;
            // We use [Bsq]^k = (-1)^k*[Bqs]^k
            auto op1H = op1.H();
            symbolic_sum<Tm> top2H;
            for(const auto& r2a : cindex2){
               auto op2 = symbolic_oper(block2,'C',r2a).H();
               double fac = (ts==0)? 1.0/std::sqrt(2.0) : -std::sqrt(3.0/2.0);
               top2H.sum(fac*get_vint2e_su2(int2e,ts,p/2,ks1,r2a/2,kq1), op2); // s<->q
            }
            auto op12H = symbolic_prod(op1H,top2H);
            op12H.ispins.push_back(std::make_tuple(ts,1,1));
            formulae.append(op12H);
         }
      }

   // sum_r (sum_qs -<pq1||s1r2> aq[1]^+as[1]) ar[2]
   template <typename Tm>
      void symbolic_compxwf_opS6c_su2(const std::string block1,
            const std::string block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const integral::two_body<Tm>& int2e,
            const int p,
            const std::vector<int>& bindex1,
            symbolic_task<Tm>& formulae){
         // sum_r ar[2]
         int kp = p/2, pa = p, pb = pa+1;
         for(const auto& ra : cindex2){
            int rb = ra+1, kr = ra/2;
            auto op2a = symbolic_oper(block2,'C',ra).H();
            // sum_r (sum_qs -<pq1||s1r2> aq[1]^+as[1]) ar[2]
            symbolic_sum<Tm> top1s, top1t, top1sH, top1tH;
            // loop over Bqs
            for(const auto& iqs : bindex1){
               auto qs = oper_unpack(iqs);
               int q1 = qs.first , kq = q1/2, spin_q1 = q1%2;
               int s1 = qs.second, ks = s1/2, spin_s1 = s1%2;
               int ts = (spin_q1!=spin_s1)? 2 : 0;
               auto op1 = symbolic_oper(block1,'B',iqs);
               // sum_r -<pq1||s1r2> ar[2]
               if(ts == 0){
                  double fac = 1.0/std::sqrt(2.0);
                  top1s.sum(fac*get_vint2e_su2(int2e,ts,kp,kq,kr,ks), op1);
               }else{
                  double fac = std::sqrt(3.0/2.0);
                  top1t.sum(fac*get_vint2e_su2(int2e,ts,kp,kq,kr,ks), op1);
               }
               // Hermitian part: q1<->s1
               if(kq == ks) continue;
               // We use [Bsq]^k = (-1)^k*[Bqs]^k
               auto op1H = op1.H();
               if(ts == 0){
                  double fac = 1.0/std::sqrt(2.0);
                  top1sH.sum(fac*get_vint2e_su2(int2e,ts,kp,ks,kr,kq), op1H); // s<->q
               }else{
                  double fac = -std::sqrt(3.0/2.0);
                  top1tH.sum(fac*get_vint2e_su2(int2e,ts,kp,ks,kr,kq), op1H); // s<->q
               }
            }
            if(top1s.size()>0){
               auto op12 = symbolic_prod(top1s,op2a);
               op12.ispins.push_back(std::make_tuple(0,1,1));
               formulae.append(op12);
            }
            if(top1t.size()>0){
               auto op12 = symbolic_prod(top1t,op2a);
               op12.ispins.push_back(std::make_tuple(2,1,1));
               formulae.append(op12);
            }
            if(top1sH.size()>0){
               auto op12 = symbolic_prod(top1sH,op2a);
               op12.ispins.push_back(std::make_tuple(0,1,1));
               formulae.append(op12);
            }
            if(top1tH.size()>0){
               auto op12 = symbolic_prod(top1tH,op2a);
               op12.ispins.push_back(std::make_tuple(2,1,1));
               formulae.append(op12);
            }
         }
      }

} // ctns

#endif
