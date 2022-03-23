#ifndef SYMBOLIC_COMPXWF_H
#define SYMBOLIC_COMPXWF_H

#include "symbolic_task.h"
#include "symbolic_op1op2xwf.h"

namespace ctns{

// kernel for computing renormalized P|ket> or P^+|ket> 
template <typename Tm>
symbolic_task<Tm> symbolic_compxwf_opP(const std::string block1,
				       const std::string block2,
				       const std::vector<int>& cindex1,
				       const std::vector<int>& cindex2,
	                               const integral::two_body<Tm>& int2e,
				       const int index,
				       const int isym,
				       const bool ifkr,
				       const bool ifdagger=false){
   symbolic_task<Tm> formulae; 
   auto pq = oper_unpack(index);
   int p = pq.first, q = pq.second;
   auto sym_op = get_qsym_opP(isym, p, q);
   // 
   // Ppq = 1/2<pq||sr> aras  (p<q)
   //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
   //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
   //     + <pq||s1r2> ar2*as1	   => -<pq||s1r2> as1*ar2
   //
   // 1. P1*I2
   auto P1pq = symbolic_prod<Tm>(symbolic_oper(block1,'P',index,ifdagger));
   formulae.append(P1pq);
   // 2. I1*P2
   auto P2pq = symbolic_prod<Tm>(symbolic_oper(block2,'P',index,ifdagger));
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
		          isym,sym_op,oij,0,0,ifdagger); // as1*ar2
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
				       const bool ifkr,
			               const bool ifdagger=false){
   symbolic_task<Tm> formulae; 
   auto ps = oper_unpack(index);
   int p = ps.first, s = ps.second;
   auto sym_op = get_qsym_opQ(isym, p, s);
   //
   // Qps = <pq||sr> aq^+ar
   //     = <pq1||sr1> Bq1r1 	=> Qps^1
   // 	  + <pq2||sr2> Bq2r2 	=> Qps^2
   //     + <pq1||sr2> aq1^+ar2 => <pq1||sr2> aq1^+*ar2 
   //     + <pq2||sr1> aq2^+ar1 => -<pq2||sr1> ar1*aq2^+
   //
   // 1. Q1*I2
   auto Q1ps = symbolic_prod<Tm>(symbolic_oper(block1,'Q',index,ifdagger));
   formulae.append(Q1ps);
   // 2. I1*Q2
   auto Q2ps = symbolic_prod<Tm>(symbolic_oper(block2,'Q',index,ifdagger));
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
 		          isym,sym_op,o1ij,1,0,ifdagger); // aq1^+*ar2
   symbolic_op1op2xwf<Tm>(ifkr,formulae,block1,block2,cindex1,cindex2,
 		          isym,sym_op,o2ij,0,1,ifdagger); // aq1*ar2^+
   return formulae;
}

// kernel for computing renormalized Sp|ket> [6 terms]
template <typename Tm>
symbolic_task<Tm> symbolic_compxwf_opS(const std::string block1,
				       const std::string block2,
				       const std::vector<int>& cindex1,
				       const std::vector<int>& cindex2,
	                               const integral::two_body<Tm>& int2e,
		                       const int index,
				       const int isym,
				       const bool ifkr,
			               const int size,
			               const int rank,
			               const bool ifdagger=false){
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
   // 1. S1*I2
   auto S1p = symbolic_prod<Tm>(symbolic_oper(block1,'S',index,ifdagger));
   formulae.append(S1p);
   // 2. I1*S2
   auto S2p = symbolic_prod<Tm>(symbolic_oper(block2,'S',index,ifdagger));
   formulae.append(S2p);
   // cross terms
   int kc1 = ifkr? 2*cindex1.size() : cindex2.size();
   int kA1 = kc1*(kc1-1)/2;
   int kB1 = kc1*kc1;
   int kc2 = ifkr? 2*cindex2.size() : cindex2.size();
   int kA2 = kc2*(kc2-1)/2;
   int kB2 = kc2*kc2;
   bool combine_two_index3 = 0; //(kc1 <= kA2);
   bool combine_two_index4 = 0; //(kc1 <= kB2);
   bool combine_two_index5 = 1; //(kc2 <= kA1);
   bool combine_two_index6 = 1; //(kc2 <= kB1);
   auto aindex1 = oper_index_opA(cindex1, ifkr);
   auto bindex1 = oper_index_opB(cindex1, ifkr);
   auto aindex2 = oper_index_opA(cindex2, ifkr);
   auto bindex2 = oper_index_opB(cindex2, ifkr); 
   if(!ifkr){
      
      // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]	   
      if(combine_two_index3){
         // sum_q aq^+[1]*Ppq[2]
         for(const auto& q : cindex1){
            int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
            int iproc = distribute2(ipq,size);
            if(iproc == rank){
               auto op1c = symbolic_oper(block1,'C',q);
               auto op2P = symbolic_oper(block2,'P',ipq);
               auto c1P2 = (p<q)? symbolic_prod<Tm>(op1c,op2P) : 
           	     	          symbolic_prod<Tm>(op1c,op2P,-1.0);
               formulae.append(c1P2);
            }
         } // q
      }else{
         // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
         for(const auto& isr : aindex2){
	    int iproc = distribute2(isr,size);
	    if(iproc == rank){
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
		  top1.sum(int2e.get(p,q1,s2,r2),op1);
	       }
	       if(top1.size() > 0){
	          auto term = symbolic_prod<Tm>(top1,op2);
		  if(ifdagger) term = term.H();
		  formulae.append(term);
	       }
	    }
	 }
      }
 
      // 4. <pq2||s1r2> aq[2]^+ar[2]as[1]     
      if(combine_two_index4){
	 // sum_q aq[1]*Qpq[2]
         for(const auto& q : cindex1){
            int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
            int iproc = distribute2(ipq,size);
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
         // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
	 for(const auto& iqr : bindex2){
	    int iproc = distribute2(iqr,size);
	    if(iproc == rank){
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
		  top1.sum(int2e.get(p,q2,s1,r2),op1);
	       }
	       if(top1.size() > 0){
		  auto term = symbolic_prod<Tm>(top1,op2);
		  if(ifdagger) term = term.H();
		  formulae.append(term);
	       }
	       // Hermitian part: q2<->r2
	       if(q2 == r2) continue;	    
	       auto op2H = op2.H();
	       auto sym_op2H = op2H.get_qsym(isym);
	       symbolic_sum<Tm> top1H;
	       for(const auto& s1 : cindex1){
		  auto op1 = symbolic_oper(block1,'C',s1).H();
		  auto sym_op1 = op1.get_qsym(isym);
		  if(sym_op != sym_op1 + sym_op2H) continue;
		  top1H.sum(int2e.get(p,r2,s1,q2),op1);
	       }
	       if(top1H.size() > 0){
		  auto term = symbolic_prod<Tm>(top1H,op2H);
		  if(ifdagger) term = term.H();
		  formulae.append(term);
	       }
	    }
         }
      }
     
      // 5. <pq2||s1r1> aq[2]^+ar[1]as[1] 
      if(combine_two_index5){
         // sum_q Ppq[1]*aq^+[2]
         for(const auto& q : cindex2){
            int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
            int iproc = distribute2(ipq,size);
            if(iproc == rank){
               auto op2c = symbolic_oper(block2,'C',q);
               auto op1P = symbolic_oper(block1,'P',ipq);
               auto P1c2 = (p<q)? symbolic_prod<Tm>(op1P,op2c) : 
           	    	          symbolic_prod<Tm>(op1P,op2c,-1.0);
               formulae.append(P1c2);
            }
         } // q
      }else{
/*
	 // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
	 for(const auto& op1A : qops1('A')){
	    auto sr = oper_unpack(op1A.first);
	    int s1 = sr.first;
	    int r1 = sr.second;
	    const auto& op1 = op1A.second.H();
	    // sum_q <pq2||s1r1> aq[2]^+
	    stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
	    for(const auto& op2C : qops2('C')){
	       if(op2C.second.info.sym != tmp_op2.info.sym) continue;
	       int q2 = op2C.first;
	       const auto& op2 = op2C.second;
	       tmp_op2 += int2e.get(p,q2,s1,r1)*op2;
	    } 
	    opxwf += oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
	 }
      }
*/
      }

      // 6. <pq1||s1r2> aq[1]^+ar[2]as[1]   
      if(combine_two_index6){
	 // sum_q Qpq^[1]*aq[2]
         for(const auto& q : cindex2){
            int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
            int iproc = distribute2(ipq,size);
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
      }

   }else{
      
      int pa = p, pb = pa+1;

      // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]
      if(combine_two_index3){
	 // sum_q aq^+[1]*Ppq[2]    
         for(const auto& qa : cindex1){
            int qb = qa+1, kq = qa/2;
   	    auto op1c_A = symbolic_oper(block1,'C',qa);
            auto op1a_A = op1c_A.H();
            auto op1c_B = op1c_A.K(1);
   	    auto op1a_B = op1a_A.K(1);
   	    int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
   	    int iproc_aa = distribute2(ipq_aa,size);
   	    if(iproc_aa == rank){
   	       auto op2P_AA = symbolic_oper(block2,'P',ipq_aa);
   	       auto c1P2_AA = (kp<kq)? symbolic_prod<Tm>(op1c_A,op2P_AA) : 
   		   		       symbolic_prod<Tm>(op1c_A,op2P_AA,-1.0); 
   	       formulae.append(c1P2_AA);
   	    }
   	    int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
   	    int iproc_ab = distribute2(ipq_ab,size);
   	    if(iproc_ab == rank){
   	       auto op2P_AB = symbolic_oper(block2,'P',ipq_ab);
   	       auto c1P2_AB = (kp<kq)? symbolic_prod<Tm>(op1c_B,op2P_AB) :
   	               		       symbolic_prod<Tm>(op1c_B,op2P_AB.K(1),-1.0);
               formulae.append(c1P2_AB);
   	    }
         } // qa
      }else{
/*
         // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
         for(const auto& op2A : qops2('A')){
	    double wt =  wfacAP(op2A.first);
            auto sr = oper_unpack(op2A.first);
            int s2 = sr.first , ks2 = s2/2, spin_s2 = s2%2, s2K = s2+1-2*spin_s2;
            int r2 = sr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;
            const auto& op2 = op2A.second.H();
            const auto& op2K = op2.K(2-spin_s2-spin_r2);
	    // sum_q <pq1||s2r2> aq[1]^+
            stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
            stensor2<Tm> tmp_op1K(sym_op-op2K.info.sym, qrow1, qcol1);
            for(const auto& op1C : qops1('C')){
               int q1a = op1C.first, q1b = q1a+1;
               const auto& op1c_A = op1C.second;
	       const auto& op1c_B = op1c_A.K(1);
               tmp_op1 += int2e.get(p,q1a,s2,r2)*op1c_A + int2e.get(p,q1b,s2,r2)*op1c_B;
	       tmp_op1K += int2e.get(p,q1a,s2K,r2K)*op1c_A + int2e.get(p,q1b,s2K,r2K)*op1c_B;
            }
            opxwf += wt*oper_kernel_OOwf(superblock,site,tmp_op1,op2,0,ifdagger);
            opxwf += wt*oper_kernel_OOwf(superblock,site,tmp_op1K,op2K,0,ifdagger);
         }
*/
      }

      // 4. <pq2||s1r2> aq[2]^+ar[2]as[1]      
      if(combine_two_index4){
	 // sum_q aq[1]*Qpq[2]    
         for(const auto& qa : cindex1){
            int qb = qa+1, kq = qa/2;
   	    auto op1c_A = symbolic_oper(block1,'C',qa);
            auto op1a_A = op1c_A.H();
            auto op1c_B = op1c_A.K(1);
   	    auto op1a_B = op1a_A.K(1);
   	    int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
   	    int iproc_aa = distribute2(ipq_aa,size);
   	    if(iproc_aa == rank){
   	       auto op2Q_AA = symbolic_oper(block2,'Q',ipq_aa);
   	       auto a1Q2_AA = (kp<kq)? symbolic_prod<Tm>(op1a_A,op2Q_AA) : 
   	               		       symbolic_prod<Tm>(op1a_A,op2Q_AA.H());
   	       formulae.append(a1Q2_AA);
   	    }
   	    int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
   	    int iproc_ab = distribute2(ipq_ab,size);
   	    if(iproc_ab == rank){
   	       auto op2Q_AB = symbolic_oper(block2,'Q',ipq_ab);
   	       auto a1Q2_AB = (kp<kq)? symbolic_prod<Tm>(op1a_B,op2Q_AB) :
   	               		       symbolic_prod<Tm>(op1a_B,op2Q_AB.K(1).H());
               formulae.append(a1Q2_AB);
   	    }
         } // qa
      }else{
/*
         // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
         for(const auto& op2B : qops2('B')){
            auto qr = oper_unpack(op2B.first);
            int q2 = qr.first , kq2 = q2/2, spin_q2 = q2%2, q2K = q2+1-2*spin_q2;
            int r2 = qr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;;
            const auto& op2 = op2B.second;
	    const auto& op2K =  op2.K(2-spin_q2-spin_r2);
            // sum_s <pq2||s1r2> as[1]
            stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
            stensor2<Tm> tmp_op1K(sym_op-op2K.info.sym, qrow1, qcol1);
            for(const auto& op1C : qops1('C')){
               int s1a = op1C.first, s1b = s1a+1;
               const auto& op1a_A = op1C.second.H();
	       const auto& op1a_B = op1a_A.K(1);
               tmp_op1 += int2e.get(p,q2,s1a,r2)*op1a_A + int2e.get(p,q2,s1b,r2)*op1a_B;
	       tmp_op1K += int2e.get(p,q2K,s1a,r2K)*op1a_A + int2e.get(p,q2K,s1b,r2K)*op1a_B;
            }
            opxwf += oper_kernel_OOwf(superblock,site,tmp_op1,op2,0,ifdagger);
            opxwf += oper_kernel_OOwf(superblock,site,tmp_op1K,op2K,0,ifdagger);
	    // Hermitian part: q2<->r2
            if(kq2 == kr2) continue;
            stensor2<Tm> tmp_op1H(sym_op+op2.info.sym, qrow1, qcol1);
            stensor2<Tm> tmp_op1KH(sym_op+op2K.info.sym, qrow1, qcol1);
            for(const auto& op1C : qops1('C')){
               int s1a = op1C.first, s1b = s1a+1;
               const auto& op1a_A = op1C.second.H();
	       const auto& op1a_B = op1a_A.K(1);
               tmp_op1H += int2e.get(p,r2,s1a,q2)*op1a_A + int2e.get(p,r2,s1b,q2)*op1a_B;
	       tmp_op1KH += int2e.get(p,r2K,s1a,q2K)*op1a_A + int2e.get(p,r2K,s1b,q2K)*op1a_B;
            }
            opxwf += oper_kernel_OOwf(superblock,site,tmp_op1H,op2.H(),0,ifdagger);
            opxwf += oper_kernel_OOwf(superblock,site,tmp_op1KH,op2K.H(),0,ifdagger);
         }
*/
      }

      // 5. <pq2||s1r1> aq[2]^+ar[1]as[1]      
      if(combine_two_index5){
         // sum_q Ppq[1]*aq^+[2]
         for(const auto& qa : cindex2){
            int qb = qa+1, kq = qa/2;
   	    auto op2c_A = symbolic_oper(block2,'C',qa);
            auto op2a_A = op2c_A.H();
            auto op2c_B = op2c_A.K(1);
   	    auto op2a_B = op2a_A.K(1);
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
   	    int iproc_aa = distribute2(ipq_aa,size);
   	    if(iproc_aa == rank){
   	       auto op1P_AA = symbolic_oper(block1,'P',ipq_aa);
   	       auto P1c2_AA = (kp<kq)? symbolic_prod<Tm>(op1P_AA,op2c_A) : 
   	              		       symbolic_prod<Tm>(op1P_AA,op2c_A,-1.0); 
   	       formulae.append(P1c2_AA);
            } 
   	    int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
   	    int iproc_ab = distribute2(ipq_ab,size);
   	    if(iproc_ab == rank){
   	       auto op1P_AB = symbolic_oper(block1,'P',ipq_ab);
   	       auto P1c2_AB = (kp<kq)? symbolic_prod<Tm>(op1P_AB,op2c_B) :
   	               		       symbolic_prod<Tm>(op1P_AB.K(1),op2c_B,-1.0);
               formulae.append(P1c2_AB);
   	    }
         } // qa
      }else{
         // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
/*
         for(const auto& op1A : qops1('A')){
            double wt =  wfacAP(op1A.first);
	    auto sr = oper_unpack(op1A.first);
            int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
            int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2, r1K = r1+1-2*spin_r1;
            const auto& op1 = op1A.second.H();
	    const auto& op1K = op1.K(2-spin_s1-spin_r1);
            // sum_q <pq2||s1r1> aq[2]^+
            stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
            stensor2<Tm> tmp_op2K(sym_op-op1K.info.sym, qrow2, qcol2);
            for(const auto& op2C : qops2('C')){
               int q2a = op2C.first, q2b = q2a+1;
               const auto& op2c_A = op2C.second;
               const auto& op2c_B = op2c_A.K(1);
               tmp_op2 += int2e.get(p,q2a,s1,r1)*op2c_A + int2e.get(p,q2b,s1,r1)*op2c_B;
               tmp_op2K += int2e.get(p,q2a,s1K,r1K)*op2c_A + int2e.get(p,q2b,s1K,r1K)*op2c_B;
            }
            opxwf += wt*oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
            opxwf += wt*oper_kernel_OOwf(superblock,site,op1K,tmp_op2K,1,ifdagger);
         }
*/
      }

      // 6. <pq1||s1r2> aq[1]^+ar[2]as[1]      
      if(combine_two_index6){
         // sum_q Qpq^[1]*aq[2]
         for(const auto& qa : cindex2){
            int qb = qa+1, kq = qa/2;
   	    auto op2c_A = symbolic_oper(block2,'C',qa);
            auto op2a_A = op2c_A.H();
            auto op2c_B = op2c_A.K(1);
   	    auto op2a_B = op2a_A.K(1);
            int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
   	    int iproc_aa = distribute2(ipq_aa,size);
   	    if(iproc_aa == rank){
   	       auto op1Q_AA = symbolic_oper(block1,'Q',ipq_aa);
   	       auto Q1a2_AA = (kp<kq)? symbolic_prod<Tm>(op1Q_AA,op2a_A) : 
   	               		       symbolic_prod<Tm>(op1Q_AA.H(),op2a_A);
   	       formulae.append(Q1a2_AA);
            }
   	    int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
   	    int iproc_ab = distribute2(ipq_ab,size);
   	    if(iproc_ab == rank){
   	       auto op1Q_AB = symbolic_oper(block1,'Q',ipq_ab);
   	       auto Q1a2_AB = (kp<kq)? symbolic_prod<Tm>(op1Q_AB,op2a_B) :
   	               		       symbolic_prod<Tm>(op1Q_AB.K(1).H(),op2a_B);
               formulae.append(Q1a2_AB);
   	    }
         } // qa
      }else{
         // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
/*
         for(const auto& op1B : qops1('B')){
	    auto qs = oper_unpack(op1B.first);
            int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2, q1K = q1+1-2*spin_q1;
            int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
            const auto& op1 = op1B.second;
	    const auto& op1K = op1.K(2-spin_q1-spin_s1);
            // sum_r -<pq1||s1r2> ar[2]
            stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
            stensor2<Tm> tmp_op2K(sym_op-op1K.info.sym, qrow2, qcol2);
            for(const auto& op2C : qops2('C')){
               int r2a = op2C.first, r2b = r2a+1;
               const auto& op2a_A = op2C.second.H();
	       const auto& op2a_B = op2a_A.K(1);
               tmp_op2 -= int2e.get(p,q1,s1,r2a)*op2a_A + int2e.get(p,q1,s1,r2b)*op2a_B;
	       tmp_op2K -= int2e.get(p,q1K,s1K,r2a)*op2a_A + int2e.get(p,q1K,s1K,r2b)*op2a_B;
            }
            opxwf += oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
            opxwf += oper_kernel_OOwf(superblock,site,op1K,tmp_op2K,1,ifdagger);
            if(kq1 == ks1) continue;
            stensor2<Tm> tmp_op2H(sym_op+op1.info.sym, qrow2, qcol2);
            stensor2<Tm> tmp_op2KH(sym_op+op1K.info.sym, qrow2, qcol2);
            for(const auto& op2C : qops2('C')){
               int r2a = op2C.first, r2b = r2a+1;
               const auto& op2a_A = op2C.second.H();
               const auto& op2a_B = op2a_A.K(1);
               tmp_op2H -= int2e.get(p,s1,q1,r2a)*op2a_A + int2e.get(p,s1,q1,r2b)*op2a_B;
               tmp_op2KH -= int2e.get(p,s1K,q1K,r2a)*op2a_A + int2e.get(p,s1K,q1K,r2b)*op2a_B;
            }
            opxwf += oper_kernel_OOwf(superblock,site,op1.H(),tmp_op2H,1,ifdagger);
            opxwf += oper_kernel_OOwf(superblock,site,op1K.H(),tmp_op2KH,1,ifdagger);
         }
*/
      }
    
   } // ifkr
   return formulae;
}

// kernel for computing renormalized H|ket>: 
template <typename Tm>
symbolic_task<Tm> symbolic_compxwf_opH(const std::string block1,
			               const std::string block2,
			               const std::vector<int>& cindex1,
			               const std::vector<int>& cindex2,
			               const bool ifkr,
			               const int size,
			               const int rank){
   symbolic_task<Tm> formulae;
   // for AP,BQ terms: to ensure the optimal scaling!
   const bool ifNC = cindex1.size() <= cindex2.size(); 
   auto AP1 = ifNC? 'A' : 'P';
   auto AP2 = ifNC? 'P' : 'A';
   auto BQ1 = ifNC? 'B' : 'Q';
   auto BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? cindex1 : cindex2;
   auto aindex = oper_index_opA(cindex, ifkr);
   auto bindex = oper_index_opB(cindex, ifkr);
   //
   // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   //   = H1 + H2
   //   + p1^+*Sp1^2 + h.c.
   //   + q2^+*Sq2^1 + h.c.
   //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
   //   + <p1q2||s1r2> p1^+q2^+r2s1 
   //
   const double scale = ifkr? 0.25 : 0.5; 
   // 1. H1*I2
   auto H1 = symbolic_prod<Tm>(symbolic_oper(block1,'H',0), scale);
   formulae.append(H1);
   // 2. I1*H2
   auto H2 = symbolic_prod<Tm>(symbolic_oper(block2,'H',0), scale);
   formulae.append(H2);
   // One-index operators
   // 3. sum_p1 p1^+ Sp1^2 + h.c. 
   for(const auto& p1 : cindex1){
      auto op1C = symbolic_oper(block1,'C',p1);
      auto op2S = symbolic_oper(block2,'S',p1);
      auto C1S2 = symbolic_prod<Tm>(op1C, op2S);
      formulae.append(C1S2);
   }
   // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
   for(const auto& q2 : cindex2){
      auto op1S = symbolic_oper(block1,'S',q2);
      auto op2C = symbolic_oper(block2,'C',q2);
      auto S1C2 = symbolic_prod<Tm>(op1S, op2C, -1.0);
      formulae.append(S1C2);
   }
   // Two-index operators
   // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
   for(const auto& index : aindex){
      int iproc = distribute2(index,size);
      if(iproc == rank){
         const double wt = ifkr? wfacAP(index) : 1.0;
         auto op1 = symbolic_oper(block1,AP1,index);
         auto op2 = symbolic_oper(block2,AP2,index);
         auto AP = symbolic_prod<Tm>(op1, op2, wt);
         formulae.append(AP);
      } // iproc
   }
   // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
   for(const auto& index : bindex){
      int iproc = distribute2(index,size);
      if(iproc == rank){
         const double wt = ifkr? wfacBQ(index) : wfac(index);
         auto op1 = symbolic_oper(block1,BQ1,index);
         auto op2 = symbolic_oper(block2,BQ2,index);
         auto BQ = symbolic_prod<Tm>(op1, op2, wt);
         formulae.append(BQ);
      } // iproc
   }
   return formulae;
}

} // ctns

#endif
