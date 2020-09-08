#ifndef CTNS_OPER_OPWF_H
#define CTNS_OPER_OPWF_H

#include "ctns_oper_util.h"

namespace ctns{

// kernel for computing renormalized P|ket> or P^+|ket> 
template <typename Tm>
qtensor3<Tm> oper_opwf_opP(const std::string& superblock,
		           const qtensor3<Tm>& site,
		           oper_dict<Tm>& qops1,
		           oper_dict<Tm>& qops2,
	                   const integral::two_body<Tm>& int2e,
	                   const integral::one_body<Tm>& int1e,
		           const int index,
		           const bool ifdagger=false){
   const bool Htype = tools::is_complex<Tm>();
   auto spq = oper_unpack(index);
   int spincase = std::get<0>(spq);
   int p = 2*std::get<1>(spq);
   int q = 2*std::get<2>(spq)+spincase;
   // determine symmetry of Paa/Pab
   qsym sym_op = (!Htype & spincase==0)? qsym(-2,-2) : qsym(-2,0);
   qsym sym_opwf = ifdagger? -sym_op+site.sym : sym_op+site.sym;
   qtensor3<Tm> opwf(sym_opwf, site.qmid, site.qrow, site.qcol, site.dir);
   // 
   // Ppq = 1/2<pq||sr> aras  (p<q)
   //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
   //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
   //     + <pq||s1r2> ar2*as1	   => -<pq||s1r2> as1*ar2
   //
   // 1. P1*I2
   assert(qops1.find('P') != qops1.end());
   opwf += oper_kernel_OIwf(superblock,site,qops1['P'].at(index),ifdagger);
   // 2. I1*P2
   assert(qops2.find('P') != qops2.end());
   opwf += oper_kernel_IOwf(superblock,site,qops2['P'].at(index),0,ifdagger);
   // 3. -<pq||s1r2> as1*ar2
   const auto& qrow1 = qops1['P'].at(index).qrow;
   const auto& qcol1 = qops1['P'].at(index).qcol;
   const auto& qrow2 = qops2['P'].at(index).qrow;
   const auto& qcol2 = qops2['P'].at(index).qcol;
   double sgn = ifdagger? -1.0 : 1.0; // (aras)^+ = as^+ar^+
   if(Htype){

      if(qops1['C'].size() < qops2['C'].size()){
         // s1 * (-<pq||s1r2> r2) 
         for(const auto& op1C : qops1['C']){
            int s1 = op1C.first, s1A = 2*s1, s1B = s1A+1;
            const auto& op1a = op1C.second.H();
            // op2 = -<pq||s1r2> r2
            qtensor2<Tm> op2A(qsym(-1,0), qrow2, qcol2);
            qtensor2<Tm> op2B(qsym(-1,0), qrow2, qcol2);
            for(const auto& op2C : qops2['C']){
               int r2 = op2C.first, r2A = 2*r2, r2B = r2A+1;
               const auto& op2a = op2C.second.H(); 
               op2A -= int2e.get(p,q,s1A,r2A)*op2a + int2e.get(p,q,s1A,r2B)*op2a.K(1);
               op2B -= int2e.get(p,q,s1B,r2A)*op2a + int2e.get(p,q,s1B,r2B)*op2a.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1a,op2A,1,ifdagger)
                        +oper_kernel_OOwf(superblock,site,op1a.K(1),op2B,1,ifdagger));
	 }
      }else{
         // (-<pq||s1r2> s1) * r2
	 for(const auto& op2C : qops2['C']){
            int r2 = op2C.first, r2A = 2*r2, r2B = r2A+1;
	    const auto& op2 = op2C.second.H();
            // op1 = -<pq||s1r2> s1
            qtensor2<Tm> op1A(qsym(-1,0), qrow1, qcol1);
            qtensor2<Tm> op1B(qsym(-1,0), qrow1, qcol1);
            for(const auto& op1C : qops1['C']){
               int s1 = op1C.first, s1A = 2*s1, s1B = s1A+1;
	       const auto& op1a = op1C.second.H();
	       op1A -= int2e.get(p,q,s1A,r2A)*op1a + int2e.get(p,q,s1B,r2A)*op1a.K(1);
	       op1B -= int2e.get(p,q,s1A,r2B)*op1a + int2e.get(p,q,s1B,r2B)*op1a.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1A,op2,1,ifdagger)
                        +oper_kernel_OOwf(superblock,site,op1B,op2.K(1),1,ifdagger));
         }

      }

   }else{
	  
/*
      if(spincase == 0){

      // NR: Paa
      for(const auto& op1C : qops1['C']){
         int s1 = op1C.first;
         const auto& op1a = op1C.second.H();
         qtensor2 op2A(qsym(-1,-1), qrow, qcol);
         // op2 = -<pq||s1r2> r2
         for(const auto& op2C : qops2['C']){
            int r2 = op2C.first;
            const auto& op2a = op2C.second.H();
            op2A -= int2e.get(p,q,2*s1  ,2*r2  )*op2a;
         }
         if(ifdagger){
            opwf -= oper_kernel_OOwf(superblock,site,op1.H(),op2A.H(),1);
         }else{
            opwf += oper_kernel_OOwf(superblock,site,op1,op2A,1);
         }
      }

      }else if(spincase == 1){

      // NR: Pab
      for(const auto& op1C : qops1['C']){
         int s1 = op1C.first;
         const auto& op1a = op1C.second.H();
         qtensor2 op2A(qsym(-1,+1), qrow, qcol);
         qtensor2 op2B(qsym(-1,-1), qrow, qcol);
         // op2 = -<pq||s1r2> r2
         for(const auto& op2C : qops2['C']){
            int r2 = op2C.first;
            const auto& op2a = op2C.second.H(); 
            op2A -= int2e.get(p,q,2*s1  ,2*r2+1)*op2a.K(1);
            op2B -= int2e.get(p,q,2*s1+1,2*r2  )*op2a;
         }
         if(ifdagger){
            opwf -= oper_kernel_OOwf(superblock,site,op1.H(),op2A.H(),1);
            opwf -= oper_kernel_OOwf(superblock,site,op1.K(1).H(),op2B.H(),1);
         }else{
            opwf += oper_kernel_OOwf(superblock,site,op1,op2A,1);
            opwf += oper_kernel_OOwf(superblock,site,op1.K(1),op2B,1);
         }
      }
*/
   } // Htype
   return opwf;
}

// kernel for computing renormalized Q|ket> or Q^+|ket>
template <typename Tm>
qtensor3<Tm> oper_opwf_opQ(const std::string& superblock,
		           const qtensor3<Tm>& site,
			   oper_dict<Tm>& qops1,
			   oper_dict<Tm>& qops2,
	                   const integral::two_body<Tm>& int2e,
	                   const integral::one_body<Tm>& int1e,
		           const int index,
			   const bool ifdagger=false){
   const bool Htype = tools::is_complex<Tm>();
   auto sps = oper_unpack(index);
   int spincase = std::get<0>(sps);
   int p = 2*std::get<1>(sps);
   int s = 2*std::get<2>(sps)+spincase;
   // determine symmetry of Qaa/Qab
   qsym sym_op = (!Htype & spincase==1)? qsym(0,-1) : qsym(0,0);
   qsym sym_opwf = ifdagger? -sym_op+site.sym : sym_op+site.sym;
   qtensor3<Tm> opwf(sym_opwf, site.qmid, site.qrow, site.qcol, site.dir);
   //
   // Qps = <pq||sr> aq^+ar
   //     = <pq1||sr1> Bq1r1 	=> Qps^1
   // 	  + <pq2||sr2> Bq2r2 	=> Qps^2
   //     + <pq1||sr2> aq1^+ar2 => <pq1||sr2> aq1^+*ar2 
   //     + <pq2||sr1> aq2^+ar1 => -<pq2||sr1> ar1*aq2^+
   //
   // 1. Q1*I2
   assert(qops1.find('Q') != qops1.end());
   opwf += oper_kernel_OIwf(superblock,site,qops1['Q'].at(index),ifdagger);
   // 2. I1*Q2
   assert(qops2.find('Q') != qops2.end());
   opwf += oper_kernel_IOwf(superblock,site,qops2['Q'].at(index),0,ifdagger);
   // 3. <pq1||sr2> aq1^+*ar2 
   // 4. -<pq2||sr1> ar1*aq2^+
   const auto& qrow1 = qops1['Q'].at(index).qrow;
   const auto& qcol1 = qops1['Q'].at(index).qcol;
   const auto& qrow2 = qops2['Q'].at(index).qrow;
   const auto& qcol2 = qops2['Q'].at(index).qcol;
   double sgn = ifdagger? -1.0 : 1.0; // (aras)^+ = as^+ar^+
   if(Htype){ 

      if(qops1['C'].size() < qops2['C'].size()){
         // 3. q1^+ * (<pq1||sr2> r2)
         for(const auto& op1C : qops1['C']){
            int q1 = op1C.first, q1A = 2*q1, q1B = q1A+1;
            const auto& op1c = op1C.second;
            // op2 = <pq1||sr2> r2
	    qtensor2<Tm> op2A(qsym(-1,0), qrow2, qcol2);
	    qtensor2<Tm> op2B(qsym(-1,0), qrow2, qcol2);
            for(const auto& op2C : qops2['C']){
               int r2 = op2C.first, r2A = 2*r2, r2B = r2A+1;
	       const auto& op2a = op2C.second.H();
               op2A += int2e.get(p,q1A,s,r2A)*op2a + int2e.get(p,q1A,s,r2B)*op2a.K(1);
	       op2B += int2e.get(p,q1B,s,r2A)*op2a + int2e.get(p,q1B,s,r2B)*op2a.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1c,op2A,1,ifdagger)
			+oper_kernel_OOwf(superblock,site,op1c.K(1),op2B,1,ifdagger));
         }
         // 4. r1 * (-<pq2||sr1> q2^+)
         for(const auto& op1C : qops1['C']){
            int r1 = op1C.first, r1A = 2*r1, r1B = r1A+1;
            const auto& op1a = op1C.second.H();
            // op2 = -<pq2||sr1> q2^+
	    qtensor2<Tm> op2A(qsym(1,0), qrow2, qcol2);
	    qtensor2<Tm> op2B(qsym(1,0), qrow2, qcol2);
            for(const auto& op2C : qops2['C']){
               int q2 = op2C.first, q2A = 2*q2, q2B = q2A+1;
	       const auto& op2c = op2C.second;
	       op2A -= int2e.get(p,q2A,s,r1A)*op2c + int2e.get(p,q2B,s,r1A)*op2c.K(1);
	       op2B -= int2e.get(p,q2A,s,r1B)*op2c + int2e.get(p,q2B,s,r1B)*op2c.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1a,op2A,1,ifdagger)
			+oper_kernel_OOwf(superblock,site,op1a.K(1),op2B,1,ifdagger));
         }
      }else{
         // 3. (<pq1||sr2> q1^+) * r2
         for(const auto& op2C : qops2['C']){
            int r2 = op2C.first, r2A = 2*r2, r2B = r2A+1;
            const auto& op2a = op2C.second.H();
            // op1 = <pq1||sr2> q1^+
            qtensor2<Tm> op1A(qsym(1,0), qrow1, qcol1);
            qtensor2<Tm> op1B(qsym(1,0), qrow1, qcol1);
            for(const auto& op1C : qops1['C']){
               int q1 = op1C.first, q1A = 2*q1, q1B = q1A+1;
	       const auto& op1c = op1C.second;
	       op1A += int2e.get(p,q1A,s,r2A)*op1c + int2e.get(p,q1B,s,r2A)*op1c.K(1);
	       op1B += int2e.get(p,q1A,s,r2B)*op1c + int2e.get(p,q1B,s,r2B)*op1c.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1A,op2a,1,ifdagger)
			+oper_kernel_OOwf(superblock,site,op1B,op2a.K(1),1,ifdagger));    
         }
         // 4. (-<pq2||sr1> r1) * q2^+
         for(const auto& op2C : qops2['C']){
            int q2 = op2C.first, q2A = 2*q2, q2B = q2A+1;
            const auto& op2c = op2C.second;
            // op1 = -<pq2||sr1> r1
            qtensor2<Tm> op1A(qsym(-1,0), qrow1, qcol1);
            qtensor2<Tm> op1B(qsym(-1,0), qrow1, qcol1);
            for(const auto& op1C : qops1['C']){
               int r1 = op1C.first, r1A = 2*r1, r1B = r1A+1;
	       const auto& op1a = op1C.second.H();
	       op1A -= int2e.get(p,q2A,s,r1A)*op1a + int2e.get(p,q2A,s,r1B)*op1a.K(1);
	       op1B -= int2e.get(p,q2B,s,r1A)*op1a + int2e.get(p,q2B,s,r1B)*op1a.K(1);
            }
            opwf += sgn*(oper_kernel_OOwf(superblock,site,op1A,op2c,1,ifdagger)
			+oper_kernel_OOwf(superblock,site,op1B,op2c.K(1),1,ifdagger));
         }
      }

   }else{

   } // Htype
   return opwf;
}

// kernel for computing renormalized Sp|ket> [6 terms]
template <typename Tm>
qtensor3<Tm> oper_opwf_opS(const std::string& superblock,
		           const qtensor3<Tm>& site,
			   oper_dict<Tm>& qops1,
			   oper_dict<Tm>& qops2,
	                   const integral::two_body<Tm>& int2e,
	                   const integral::one_body<Tm>& int1e,
		           const int kp,
			   const bool ifdagger=false){
   const bool Htype = tools::is_complex<Tm>();
   // determine symmetry
   qsym sym_op = Htype? qsym(-1,0) : qsym(-1,-1); // Sa
   qsym sym_opwf = ifdagger? -sym_op+site.sym : sym_op+site.sym;  
   qtensor3<Tm> opwf(sym_opwf, site.qmid, site.qrow, site.qcol, site.dir);
   //
   // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   //    = Sp^1 + Sp^2 (S exists in both blocks)
   //    + <pq2||s1r1> aq2^+ar1as1 => aq2^+*Ppq2^1 = sum_q2 Ppq2^1*aq2^+
   //    + <pq1||s2r2> aq1^+ar2as2 => aq1^+*Ppq1^2 = sum_q1 aq1^+*Ppq1^2
   //    + <pq1||s1r2> aq1^+ar2as1 => Qpr2^1*ar2   = sum_r2 Qpr2^1*ar2
   //    + <pq2||s1r2> aq2^+ar2as1 => Qps1^2*as1   = sum_s1 as1*Qps1^2
   //
   // 1. S1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1['S'].at(kp),ifdagger);
   // 2. I1*S2
   opwf += oper_kernel_IOwf(superblock,site,qops2['S'].at(kp),1,ifdagger);
   // 3. sum_q2{A,B} Ppq2^1*aq2^+
   assert(qops1.find('P') != qops1.end());
   for(const auto& op2C : qops2['C']){
      int q2 = op2C.first;
      const auto& op2c = op2C.second;
      // P[pA,qA] = -P[qA,pA]
      const auto& op1PAA = (kp < q2)? qops1['P'].at(oper_pack(0,kp,q2)):
				     -qops1['P'].at(oper_pack(0,q2,kp));
      // P[pA,qB] = -P[qB,pA] = -P[qA,pB].K(1) 
      const auto& op1PAB = (kp < q2)? qops1['P'].at(oper_pack(1,kp,q2)):
         			     -qops1['P'].at(oper_pack(1,q2,kp)).K(1);
      // no sign change for dagger case: e*o=e
      opwf += oper_kernel_OOwf(superblock,site,op1PAA,op2c,1,ifdagger)
            + oper_kernel_OOwf(superblock,site,op1PAB,op2c.K(1),1,ifdagger);
   }
   // 4. sum_q1{A,B} aq1^+*Ppq1^2
   assert(qops2.find('P') != qops2.end());
   for(const auto& op1C : qops1['C']){
      int q1 = op1C.first;
      const auto& op1c = op1C.second;
      const auto& op2PAA = (kp < q1)? qops2['P'].at(oper_pack(0,kp,q1)):
         			     -qops2['P'].at(oper_pack(0,q1,kp));
      const auto& op2PAB = (kp < q1)? qops2['P'].at(oper_pack(1,kp,q1)):
         			     -qops2['P'].at(oper_pack(1,q1,kp)).K(1);
      opwf += oper_kernel_OOwf(superblock,site,op1c,op2PAA,0,ifdagger)
	    + oper_kernel_OOwf(superblock,site,op1c.K(1),op2PAB,0,ifdagger);
   }
   // 5. sum_r2{A,B} Qpr2^1*ar2
   assert(qops1.find('Q') != qops1.end());
   for(const auto& op2C : qops2['C']){
      int r2 = op2C.first;
      const auto& op2a = op2C.second.H();
      // Q[pA,rA] = Q[rA,pA]^+ (p > r)
      const auto& op1QAA = (kp < r2)? qops1['Q'].at(oper_pack(0,kp,r2)):
         			      qops1['Q'].at(oper_pack(0,r2,kp)).H();
      // Q[pA,rB] = Q[rB,pA]^+ = Q[rA,pB].K(1)^+
      const auto& op1QAB = (kp < r2)? qops1['Q'].at(oper_pack(1,kp,r2)):
      				      qops1['Q'].at(oper_pack(1,r2,kp)).K(1).H();
      opwf += oper_kernel_OOwf(superblock,site,op1QAA,op2a,1,ifdagger)
            + oper_kernel_OOwf(superblock,site,op1QAB,op2a.K(1),1,ifdagger);
   }
   // 6. sum_s1{A,B} as1*Qps1^2
   assert(qops2.find('Q') != qops2.end());
   for(const auto& op1C : qops1['C']){
      int s1 = op1C.first;
      const auto& op1a = op1C.second.H();
      const auto& op2QAA = (kp < s1)? qops2['Q'].at(oper_pack(0,kp,s1)):
         			      qops2['Q'].at(oper_pack(0,s1,kp)).H();
      const auto& op2QAB = (kp < s1)? qops2['Q'].at(oper_pack(1,kp,s1)):
         			      qops2['Q'].at(oper_pack(1,s1,kp)).K(1).H();
      opwf += oper_kernel_OOwf(superblock,site,op1a,op2QAA,0,ifdagger)
	    + oper_kernel_OOwf(superblock,site,op1a.K(1),op2QAB,0,ifdagger);
   }
   return opwf;
}

// kernel for computing renormalized H|ket>
template <typename Tm>
qtensor3<Tm> oper_opwf_opH(const std::string& superblock,
		           const qtensor3<Tm>& site,
		           oper_dict<Tm>& qops1,
		           oper_dict<Tm>& qops2,
	                   const integral::two_body<Tm>& int2e,
	                   const integral::one_body<Tm>& int1e){
   const bool dagger = true;
   qtensor3<Tm> opwf(site.sym, site.qmid, site.qrow, site.qcol, site.dir);
   //
   // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   //   = H1 + H2
   //   + p1^+*Sp1^2 + h.c.
   //   + q2^+*Sq2^1 + h.c.
   //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
   //   + <p1q2||s1r2> p1^+q2^+r2s1 
   //
   // 1. H1*I2
   opwf += oper_kernel_OIwf(superblock,site,qops1['H'].at(0));
   // 2. I1*H2
   opwf += oper_kernel_IOwf(superblock,site,qops2['H'].at(0),0);
   //
   // One-index operators
   //
   // 3. p1^+ Sp1^2 + h.c. 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1c = op1C.second;
      const auto& op2S = qops2['S'].at(p1);
      opwf += oper_kernel_OOwf(superblock,site,op1c,op2S,1);
      opwf -= oper_kernel_OOwf(superblock,site,op1c,op2S,1,dagger);
      // KR part
      opwf += oper_kernel_OOwf(superblock,site,op1c.K(1),op2S.K(1),1);
      opwf -= oper_kernel_OOwf(superblock,site,op1c.K(1),op2S.K(1),1,dagger);
   }
   // 4. q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
   for(const auto& op2C : qops2['C']){
      int q2 = op2C.first;
      const auto& op2c = op2C.second;
      const auto& op1S = qops1['S'].at(q2);
      opwf -= oper_kernel_OOwf(superblock,site,op1S,op2c,1);
      opwf += oper_kernel_OOwf(superblock,site,op1S,op2c,1,dagger);
      // KR part
      opwf -= oper_kernel_OOwf(superblock,site,op1S.K(1),op2c.K(1),1);
      opwf += oper_kernel_OOwf(superblock,site,op1S.K(1),op2c.K(1),1,dagger);
   }
   //
   // Two-index operators
   //
   // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
   assert(qops1.find('A') != qops1.end() && qops2.find('P') != qops2.end()); 
   assert(qops1.find('P') != qops1.end() && qops2.find('A') != qops2.end());
   if(qops1['A'].size() < qops2['A'].size()){ 
      for(const auto& op1A : qops1['A']){
	 const auto& index = op1A.first;
         const auto& op1 = op1A.second;
	 const auto& op2 = qops2['P'].at(index);
         double wt = wfacAP(index);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	 // KR part: phases will get cancelled in op1 and op2.
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0,dagger);
      }
   }else{
      // Prs^1+Ars^2+ + h.c.
      for(const auto& op2A : qops2['A']){
	 const auto& index = op2A.first;
	 const auto& op2 = op2A.second;
         const auto& op1 = qops1['P'].at(index);
         double wt = wfacAP(index);
	 opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	 // KR part
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0,dagger);
      }
   }
   // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
   assert(qops1.find('B') != qops1.end() && qops2.find('Q') != qops2.end()); 
   assert(qops1.find('Q') != qops1.end() && qops2.find('B') != qops2.end());
   if(qops1['B'].size() < qops2['B'].size()){
      // Bps^1*Qps^2
      for(const auto& op1B : qops1['B']){
	 const auto& index = op1B.first;
         const auto& op1 = op1B.second;
	 const auto& op2 = qops2['Q'].at(index);
	 double wt = wfacBQ(index);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	 // KR part
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0,dagger);
      }
   }else{
      // Qqr^1*Bqr^2
      for(const auto& op2B : qops2['B']){
	 const auto& index = op2B.first;
         const auto& op2 = op2B.second;
         const auto& op1 = qops1['Q'].at(index);
	 double wt = wfacBQ(index);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
	 // KR part
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0);
         opwf += wt*oper_kernel_OOwf(superblock,site,op1.K(0),op2.K(0),0,dagger);
      }
   }
   return opwf;
}

} // ctns

#endif
