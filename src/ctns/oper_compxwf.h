#ifndef OPER_COMPXWF_H
#define OPER_COMPXWF_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_op1op2xwf.h"
#include "oper_timer.h"

namespace ctns{

// 
// Compute complementary operators (PQSH) in the superblock acting on a wavefunction
//

// kernel for computing renormalized P|ket> or P^+|ket> 
template <typename Tm>
stensor3<Tm> oper_compxwf_opP(const std::string superblock,
		              const stensor3<Tm>& site,
		              const oper_dict<Tm>& qops1,
		              const oper_dict<Tm>& qops2,
	                      const integral::two_body<Tm>& int2e,
		              const int index,
		              const bool ifdagger=false){
   auto t0 = tools::get_time();
   
   const int  isym = qops1.isym;
   const bool ifkr = qops1.ifkr;
   auto pq = oper_unpack(index);
   int p = pq.first, q = pq.second;
   auto sym_op = get_qsym_opP(isym, p, q);
   auto sym_opwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;
   stensor3<Tm> opwf(sym_opwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
   // 
   // Ppq = 1/2<pq||sr> aras  (p<q)
   //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
   //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
   //     + <pq||s1r2> ar2*as1	   => -<pq||s1r2> as1*ar2
   //
   // 1. P1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1('P').at(index),ifdagger);
   // 2. I1*P2
   opwf += oper_kernel_IOwf(superblock,site,qops2('P').at(index),0,ifdagger);
   // 3. -<pq||s1r2> as1*ar2
   std::map<std::pair<int,int>,Tm> oij;
   if(!ifkr){
      for(const auto& op1C : qops1('C')){
         int s1 = op1C.first;
         for(const auto& op2C : qops2('C')){
	    int r2 = op2C.first;
	    oij[std::make_pair(s1,r2)] = -int2e.get(p,q,s1,r2);
	 }
      }
   }else{
      for(const auto& op1C : qops1('C')){
         int s1a = op1C.first, s1b = s1a+1;
         for(const auto& op2C : qops2('C')){
	    int r2a = op2C.first, r2b = r2a+1;
	    oij[std::make_pair(s1a,r2a)] = -int2e.get(p,q,s1a,r2a);
	    oij[std::make_pair(s1a,r2b)] = -int2e.get(p,q,s1a,r2b);
	    oij[std::make_pair(s1b,r2a)] = -int2e.get(p,q,s1b,r2a);
	    oij[std::make_pair(s1b,r2b)] = -int2e.get(p,q,s1b,r2b);
	 }
      }
   }
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1('C'),qops2('C'),
		  sym_op,oij,0,0,ifdagger); // as1*ar2

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nP += 1;
      oper_timer.tP += tools::get_duration(t1-t0);
   }
   return opwf;
}

// kernel for computing renormalized Q|ket> or Q^+|ket>
template <typename Tm>
stensor3<Tm> oper_compxwf_opQ(const std::string superblock,
		              const stensor3<Tm>& site,
			      const oper_dict<Tm>& qops1,
			      const oper_dict<Tm>& qops2,
	                      const integral::two_body<Tm>& int2e,
		              const int index,
			      const bool ifdagger=false){
   auto t0 = tools::get_time();

   const int  isym = qops1.isym;
   const bool ifkr = qops1.ifkr;
   auto ps = oper_unpack(index);
   int p = ps.first, s = ps.second;
   auto sym_op = get_qsym_opQ(isym, p, s);
   auto sym_opwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;
   stensor3<Tm> opwf(sym_opwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
   //
   // Qps = <pq||sr> aq^+ar
   //     = <pq1||sr1> Bq1r1 	=> Qps^1
   // 	  + <pq2||sr2> Bq2r2 	=> Qps^2
   //     + <pq1||sr2> aq1^+ar2 => <pq1||sr2> aq1^+*ar2 
   //     + <pq2||sr1> aq2^+ar1 => -<pq2||sr1> ar1*aq2^+
   //
   // 1. Q1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1('Q').at(index),ifdagger);
   // 2. I1*Q2
   opwf += oper_kernel_IOwf(superblock,site,qops2('Q').at(index),0,ifdagger);
   // 3. <pq1||sr2> aq1^+*ar2 &  4. -<pr2||sq1> aq1*ar2^+
   std::map<std::pair<int,int>,Tm> o1ij, o2ij;
   if(!ifkr){
      for(const auto& op1C : qops1('C')){
         int q1 = op1C.first;
         for(const auto& op2C : qops2('C')){
            int r2 = op2C.first;
            o1ij[std::make_pair(q1,r2)] =  int2e.get(p,q1,s,r2);
	    o2ij[std::make_pair(q1,r2)] = -int2e.get(p,r2,s,q1);
         }
      }	
   }else{
      for(const auto& op1C : qops1('C')){
         int q1a = op1C.first, q1b = q1a+1;
         for(const auto& op2C : qops2('C')){
            int r2a = op2C.first, r2b = r2a+1;
            o1ij[std::make_pair(q1a,r2a)] =  int2e.get(p,q1a,s,r2a);
            o1ij[std::make_pair(q1a,r2b)] =  int2e.get(p,q1a,s,r2b);
            o1ij[std::make_pair(q1b,r2a)] =  int2e.get(p,q1b,s,r2a);
            o1ij[std::make_pair(q1b,r2b)] =  int2e.get(p,q1b,s,r2b);
	    o2ij[std::make_pair(q1a,r2a)] = -int2e.get(p,r2a,s,q1a);
	    o2ij[std::make_pair(q1a,r2b)] = -int2e.get(p,r2b,s,q1a);
	    o2ij[std::make_pair(q1b,r2a)] = -int2e.get(p,r2a,s,q1b);
	    o2ij[std::make_pair(q1b,r2b)] = -int2e.get(p,r2b,s,q1b);
         }
      }	
   }
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1('C'),qops2('C'),
		  sym_op,o1ij,1,0,ifdagger); // aq1^+*ar2
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1('C'),qops2('C'),
		  sym_op,o2ij,0,1,ifdagger); // aq1*ar2^+

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nQ += 1;
      oper_timer.tQ += tools::get_duration(t1-t0);
   }
   return opwf;
}

// kernel for computing renormalized Sp|ket> [6 terms]
template <typename Tm>
stensor3<Tm> oper_compxwf_opS(const std::string superblock,
		              const stensor3<Tm>& site,
			      const oper_dict<Tm>& qops1,
			      const oper_dict<Tm>& qops2,
	                      const integral::two_body<Tm>& int2e,
		              const int index,
			      const int size,
			      const int rank,
			      const bool ifdagger=false){
   auto t0 = tools::get_time();

   const int  isym = qops1.isym;
   const bool ifkr = qops1.ifkr;
   int p = index, kp = p/2;
   auto sym_op = get_qsym_opS(isym, index);
   auto sym_opwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;  
   stensor3<Tm> opwf(sym_opwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
   //
   // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   //    = Sp^1 + Sp^2 (S exists in both blocks)
   //    + <pq1||s2r2> aq[1]^+ar[2]as[2] => aq^+[1]*Ppq1[2]  = sum_q aq^+[1]*Ppq[2]
   //    + <pq2||s1r2> aq[2]^+ar[2]as[1] => Qps1[2]*as[1]    = sum_q aq[1]*Qpq[2]
   //    + <pq2||s1r1> aq[2]^+ar[1]as[1] => aq2^+[2]*Ppq2[1] = sum_q Ppq[1]*aq^+[2]
   //    + <pq1||s1r2> aq[1]^+ar[2]as[1] => Qpr2[1]*ar[2]    = sum_q Qpq[1]*aq[2]
   //
   // 1. S1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1('S').at(index),ifdagger);
   // 2. I1*S2
   opwf += oper_kernel_IOwf(superblock,site,qops2('S').at(index),1,ifdagger);
   // cross terms
   if(!ifkr){
      // 3. sum_q aq^+[1]*Ppq[2] + aq[1]*Qpq[2]
      for(const auto& op1C : qops1('C')){
         int q = op1C.first;
	 int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
	 int iproc = distribute2(ipq,size);
	 if(iproc == rank){
            const auto& op1c = op1C.second;
	    const auto& op2P = (p<q)? qops2('P').at(ipq) : -qops2('P').at(ipq);
            opwf += oper_kernel_OOwf(superblock,site,op1c,op2P,0,ifdagger);
            const auto& op1a = op1C.second.H();
            const auto& op2Q = (p<q)? qops2('Q').at(ipq) : qops2('Q').at(ipq).H();
            opwf += oper_kernel_OOwf(superblock,site,op1a,op2Q,0,ifdagger);
	 }
      }
      // 4. sum_q Ppq[1]*aq^+[2] + Qpq^[1]*aq[2]
      for(const auto& op2C : qops2('C')){
         int q = op2C.first;
	 int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
	 int iproc = distribute2(ipq,size);
	 if(iproc == rank){
            const auto& op2c = op2C.second;
	    const auto& op1P = (p<q)? qops1('P').at(ipq) : -qops1('P').at(ipq);
            opwf += oper_kernel_OOwf(superblock,site,op1P,op2c,1,ifdagger);
            const auto& op2a = op2C.second.H();
            const auto& op1Q = (p<q)? qops1('Q').at(ipq) : qops1('Q').at(ipq).H();
            opwf += oper_kernel_OOwf(superblock,site,op1Q,op2a,1,ifdagger);
	 }
      }
   }else{
      int pa = p, pb = pa+1; 
      // 3. sum_q aq^+[1]*Ppq[2] + aq[1]*Qpq[2]
      for(const auto& op1C : qops1('C')){
         int qa = op1C.first, qb = qa+1, kq = qa/2;
	 const auto& op1c_A = op1C.second;
         const auto& op1a_A = op1C.second.H();
	 int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
	 int iproc_aa = distribute2(ipq_aa,size);
	 if(iproc_aa == rank){
	    const auto& op2P_AA = (kp<kq)? qops2('P').at(ipq_aa) : -qops2('P').at(ipq_aa);
            const auto& op2Q_AA = (kp<kq)? qops2('Q').at(ipq_aa) :  qops2('Q').at(ipq_aa).H();
            opwf += oper_kernel_OOwf(superblock,site,op1c_A,op2P_AA,0,ifdagger);
            opwf += oper_kernel_OOwf(superblock,site,op1a_A,op2Q_AA,0,ifdagger);
	 } 
         const auto& op1c_B = op1c_A.K(1);
	 const auto& op1a_B = op1a_A.K(1);
	 int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
	 int iproc_ab = distribute2(ipq_ab,size);
	 if(iproc_ab == rank){
	    const auto& op2P_AB = (kp<kq)? qops2('P').at(ipq_ab) : -qops2('P').at(ipq_ab).K(1);
            const auto& op2Q_AB = (kp<kq)? qops2('Q').at(ipq_ab) :  qops2('Q').at(ipq_ab).K(1).H();
            opwf += oper_kernel_OOwf(superblock,site,op1c_B,op2P_AB,0,ifdagger);
            opwf += oper_kernel_OOwf(superblock,site,op1a_B,op2Q_AB,0,ifdagger);
	 }
      }
      // 4. sum_q Ppq[1]*aq^+[2] + Qpq^[1]*aq[2]
      for(const auto& op2C : qops2('C')){
         int qa = op2C.first, qb = qa+1, kq = qa/2;
         const auto& op2c_A = op2C.second;
         const auto& op2a_A = op2C.second.H();
         int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
	 int iproc_aa = distribute2(ipq_aa,size);
	 if(iproc_aa == rank){
	    const auto& op1P_AA = (kp<kq)? qops1('P').at(ipq_aa) : -qops1('P').at(ipq_aa);
            const auto& op1Q_AA = (kp<kq)? qops1('Q').at(ipq_aa) :  qops1('Q').at(ipq_aa).H();
            opwf += oper_kernel_OOwf(superblock,site,op1P_AA,op2c_A,1,ifdagger);
            opwf += oper_kernel_OOwf(superblock,site,op1Q_AA,op2a_A,1,ifdagger);
         } 
	 const auto& op2c_B = op2c_A.K(1);
         const auto& op2a_B = op2a_A.K(1);
	 int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
	 int iproc_ab = distribute2(ipq_ab,size);
	 if(iproc_ab == rank){
            const auto& op1P_AB = (kp<kq)? qops1('P').at(ipq_ab) : -qops1('P').at(ipq_ab).K(1);
            const auto& op1Q_AB = (kp<kq)? qops1('Q').at(ipq_ab) :  qops1('Q').at(ipq_ab).K(1).H();
            opwf += oper_kernel_OOwf(superblock,site,op1P_AB,op2c_B,1,ifdagger);
            opwf += oper_kernel_OOwf(superblock,site,op1Q_AB,op2a_B,1,ifdagger);
	 }
      }
   } // ifkr

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nS += 1;
      oper_timer.tS += tools::get_duration(t1-t0);
   }
   return opwf;
}

// kernel for computing renormalized H|ket>
template <typename Tm>
stensor3<Tm> oper_compxwf_opH(const std::string superblock,
		              const stensor3<Tm>& site,
		              const oper_dict<Tm>& qops1,
		              const oper_dict<Tm>& qops2,
	                      const integral::two_body<Tm>& int2e,
			      const int size,
			      const int rank){
   auto t0 = tools::get_time();

   const int  isym = qops1.isym;
   const bool ifkr = qops1.ifkr;
   const bool dagger = true;
   // for AP,BQ terms
   const auto& cindex1 = qops1.cindex;
   const auto& cindex2 = qops2.cindex;
   const bool ifNC = cindex1.size() <= cindex2.size(); 
   char AP1 = ifNC? 'A' : 'P';
   char AP2 = ifNC? 'P' : 'A';
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
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
   stensor3<Tm> opwf(site.info.sym, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
   // 1. H1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1('H').at(0));
   // 2. I1*H2
   opwf += oper_kernel_IOwf(superblock,site,qops2('H').at(0),0);
   if(!ifkr){
      // One-index operators
      // 3. sum_p1 p1^+ Sp1^2 + h.c. 
      for(const auto& op1C : qops1('C')){
         int p1 = op1C.first;
         const auto& op1c = op1C.second;
         const auto& op2S = qops2('S').at(p1);
         opwf += oper_kernel_OOwf(superblock,site,op1c,op2S,1);
         opwf -= oper_kernel_OOwf(superblock,site,op1c,op2S,1,dagger);
      }
      // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
      for(const auto& op2C : qops2('C')){
         int q2 = op2C.first;
         const auto& op2c = op2C.second;
         const auto& op1S = qops1('S').at(q2);
         opwf -= oper_kernel_OOwf(superblock,site,op1S,op2c,1);
         opwf += oper_kernel_OOwf(superblock,site,op1S,op2c,1,dagger);
      }
      // Two-index operators
      // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            const auto& op1 = qops1(AP1).at(index);
            const auto& op2 = qops2(AP2).at(index);
            opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
            opwf += oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
         } // iproc
      }
      // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            const auto& op1 = qops1(BQ1).at(index);
            const auto& op2 = qops2(BQ2).at(index);
            const Tm wt = wfac(index);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	    auto tmp1 = oper_kernel_OOwf(superblock,site,op1,op2,0);
            auto tmp2 = oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	    linalg::xaxpy(opwf.size(), wt, tmp1.data(), opwf.data());
	    linalg::xaxpy(opwf.size(), wt, tmp2.data(), opwf.data());
         } // iproc
      }
   }else{
      // One-index operators
      // 3. sum_p1 p1^+ Sp1^2 + h.c. 
      for(const auto& op1C : qops1('C')){
         int p1 = op1C.first;
         const auto& op1c_A = op1C.second;
         const auto& op2S_A = qops2('S').at(p1);
         opwf += oper_kernel_OOwf(superblock,site,op1c_A,op2S_A,1);
         opwf -= oper_kernel_OOwf(superblock,site,op1c_A,op2S_A,1,dagger);
         // KR part
	 const auto& op1c_B = op1c_A.K(1);
	 const auto& op2S_B = op2S_A.K(1);
         opwf += oper_kernel_OOwf(superblock,site,op1c_B,op2S_B,1);
         opwf -= oper_kernel_OOwf(superblock,site,op1c_B,op2S_B,1,dagger);
      }
      // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
      for(const auto& op2C : qops2('C')){
         int q2 = op2C.first;
         const auto& op2c_A = op2C.second;
         const auto& op1S_A = qops1('S').at(q2);
         opwf -= oper_kernel_OOwf(superblock,site,op1S_A,op2c_A,1);
         opwf += oper_kernel_OOwf(superblock,site,op1S_A,op2c_A,1,dagger);
         // KR part
         const auto& op2c_B = op2c_A.K(1);
         const auto& op1S_B = op1S_A.K(1);
         opwf -= oper_kernel_OOwf(superblock,site,op1S_B,op2c_B,1);
         opwf += oper_kernel_OOwf(superblock,site,op1S_B,op2c_B,1,dagger);
      }
      // Two-index operators
      // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            const Tm wt = wfacAP(index);
            const auto& op1_A = qops1(AP1).at(index);
            const auto& op2_A = qops2(AP2).at(index);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
            auto tmp1a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
            auto tmp2a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
	    linalg::xaxpy(opwf.size(), wt, tmp1a.data(), opwf.data());
	    linalg::xaxpy(opwf.size(), wt, tmp2a.data(), opwf.data());
	    // NOTE: the following lines work for A_{pq} & A_{p\bqr{q}}, 
	    // because the global sign in K() does not matter as the pair AP 
	    // has even no. of barred indices! That is, the phases will get 
	    // cancelled in op1 and op2 after the time-reversal operation!
	    const auto& op1_B = op1_A.K(0);  
	    const auto& op2_B = op2_A.K(0); 
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
            auto tmp1b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
            auto tmp2b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
	    linalg::xaxpy(opwf.size(), wt, tmp1b.data(), opwf.data());
	    linalg::xaxpy(opwf.size(), wt, tmp2b.data(), opwf.data());
         } // iproc
      }
      // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            const Tm wt = wfacBQ(index);
            const auto& op1_A = qops1(BQ1).at(index);
            const auto& op2_A = qops2(BQ2).at(index);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
            auto tmp1a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
            auto tmp2a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
	    linalg::xaxpy(opwf.size(), wt, tmp1a.data(), opwf.data());
	    linalg::xaxpy(opwf.size(), wt, tmp2a.data(), opwf.data());
            // KR part
            const auto& op1_B = op1_A.K(0);
            const auto& op2_B = op2_A.K(0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
            //opwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
            auto tmp1b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
            auto tmp2b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
	    linalg::xaxpy(opwf.size(), wt, tmp1b.data(), opwf.data());
	    linalg::xaxpy(opwf.size(), wt, tmp2b.data(), opwf.data());
         } // iproc
      }
   } // ifkr

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nH += 1;
      oper_timer.tH += tools::get_duration(t1-t0);
   }
   return opwf;
}

} // ctns

#endif
