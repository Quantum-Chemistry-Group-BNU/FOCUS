#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// kernel for computing renormalized P|ket> [3 terms]
qtensor3 tns::oper_kernel_Pwf(const string& superblock,
		              const qtensor3& ksite,
			      oper_dict& qops1,
			      oper_dict& qops2,
	                      const integral::two_body& int2e,
	                      const integral::one_body& int1e,
		              const int index){
   // determine symmetry
   auto pq = oper_unpack(index);
   int p = pq.first,  spin_p = p%2;
   int q = pq.second, spin_q = q%2;
   qsym symP;
   if(spin_p == 0 && spin_q == 0){
      symP = qsym(-2,-2); // Paa
   }else if(spin_p == 1 && spin_q == 1){
      symP = qsym(-2, 0); // Pbb
   }else{
      symP = qsym(-2,-1); // Pos
   }
   qtensor3 opwf(symP+ksite.sym, ksite.qmid, ksite.qrow, ksite.qcol, ksite.dir);
   // 
   // Ppq = <pq||sr> aras [r>s] (p<q)
   //     = <pq||s1r1> As1r1 => Ppq^1
   //     + <pq||s2r2> As2r2 => Ppq^2
   //     + <pq||s1r2> ar2*as1 
   //
   // 1. CC: Pc*Ir with Pc = <pLqL||sCrC> rCsC [=(sCrC)^+, s<r] 
   bool ifP1 = qops1.find('P') != qops1.end();
   if(ifP1){
      opwf += oper_kernel_OIwf(superblock,ksite,qops1['P'].at(index));
   }else{
      auto& it = qops1['A'].begin()->second;
      qtensor2 op1Pdag(-symP, it.qrow, it.qcol);
      for(const auto& op1A : qops1['A']){
         if(op1Pdag.sym != op1A.second.sym) continue;
         auto sr = oper_unpack(op1A.first);
         int s1 = sr.first;
         int r1 = sr.second;
         op1Pdag += int2e.getAnti(p,q,s1,r1)*op1A.second;
      }
      opwf += oper_kernel_OIwf(superblock,ksite,op1Pdag.T());
   }
   // 2. RR: Ic*Pr
   bool ifP2 = qops2.find('P') != qops2.end();
   if(ifP2){
      opwf += oper_kernel_IOwf(superblock,ksite,qops2['P'].at(index),0);
   }else{
      // P_pLqL^R = sum_rRsR <pLqL||sRrR> rRsR
      auto& it = qops2['A'].begin()->second;
      qtensor2 op2Pdag(-symP, it.qrow, it.qcol);
      for(const auto& op2A : qops2['A']){
         if(op2Pdag.sym != op2A.second.sym) continue;
         auto sr = oper_unpack(op2A.first);
         int s2 = sr.first; 
         int r2 = sr.second;
         op2Pdag += int2e.getAnti(p,q,s2,r2)*op2A.second;
      }
      opwf += oper_kernel_IOwf(superblock,ksite,op2Pdag.T(),0);
   }
   // 3. <pq||s1r2> ar2*as1 : use one with smaller support in the outer sum
   if(qops1['C'].size() < qops2['C'].size()){
      // 3. s1 * (-<pq||s1r2> r2)
      for(const auto& op1C : qops1['C']){
         int s1 = op1C.first;
	 const auto& op1 = op1C.second;
         // op2 = -<pq||s1r2> r2
         qsym rsym = symP + op1.sym;
	 auto& it = qops2['C'].begin()->second;
         qtensor2 op2(-rsym, it.qrow, it.qcol);
         for(const auto& op2C : qops2['C']){
            if(op2.sym != op2C.second.sym) continue;
            int r2 = op2C.first;
            op2 -= int2e.getAnti(p,q,s1,r2)*op2C.second;
         }
         opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),1);
      }
   }else{
      // 3. (-<pq||s1r2> s1) * r2
      for(const auto& op2C : qops2['C']){
         int r2 = op2C.first;
	 const auto& op2 = op2C.second;
         // op1 = -<pq||s1r2> s1
	 qsym rsym = symP + op2.sym;
	 auto& it = qops1['C'].begin()->second;
         qtensor2 op1(-rsym, it.qrow, it.qcol);
         for(const auto& op1C : qops1['C']){
            if(op1.sym != op1C.second.sym) continue;
            int s1 = op1C.first;
            op1 -= int2e.getAnti(p,q,s1,r2)*op1C.second;
         }
         opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),1);
      }
   }	   
   return opwf;
}

// kernel for computing renormalized Q|ket> [4 terms]
qtensor3 tns::oper_kernel_Qwf(const string& superblock,
		              const qtensor3& ksite,
			      oper_dict& qops1,
			      oper_dict& qops2,
	                      const integral::two_body& int2e,
	                      const integral::one_body& int1e,
		              const int index){
   // determine symmetry
   auto ps = oper_unpack(index);
   int p = ps.first,  spin_p = p%2;
   int s = ps.second, spin_s = s%2;
   qsym symQ;
   if(spin_p == 0 && spin_s == 1){
      symQ = qsym(0,-1); // Qab ~ b^+a
   }else if(spin_p == 1 && spin_s == 0){
      symQ = qsym(0, 1); // Qba ~ a^+b
   }else{
      symQ = qsym(0, 0); // Qss ~ a^+a,b^+b
   }
   qtensor3 opwf(symQ+ksite.sym, ksite.qmid, ksite.qrow, ksite.qcol, ksite.dir);
   //
   // Qps = <pq||sr> aq^+ar
   //     = <pq1||sr1> Bq1r1 => Qps^1
   // 	  + <pq2||sr2> Bq2r2 => Qss^2
   //     + <pq1||sr2> aq1^+ar2
   //     + <pq2||sr1> aq2^+ar1
   //
   // 1. CC: Qc*Ir with Qc = <pLqC||sLrC> qC^+rC
   bool ifQ1 = qops1.find('Q') != qops1.end();
   if(ifQ1){
      opwf += oper_kernel_OIwf(superblock,ksite,qops1['Q'].at(index));
   }else{
      auto& it = qops1['B'].begin()->second;
      qtensor2 op1Q(symQ, it.qrow, it.qcol);
      for(const auto& op1B : qops1['B']){
         if(op1Q.sym != op1B.second.sym) continue;
         auto qr = oper_unpack(op1B.first);
         int q1 = qr.first;
         int r1 = qr.second;
         op1Q += int2e.getAnti(p,q1,s,r1)*op1B.second;
      }
      opwf += oper_kernel_OIwf(superblock,ksite,op1Q);
   }
   // 2. RR: Ic*Qr
   bool ifQ2 = qops2.find('Q') != qops2.end();
   if(ifQ2){
      opwf += oper_kernel_IOwf(superblock,ksite,qops2['Q'].at(index),0);
   }else{
      // Q_pLsL^R = sum_qRrR <pLqR||sLrR> qR^+rR
      auto& it = qops2['B'].begin()->second;
      qtensor2 op2Q(symQ, it.qrow, it.qcol);
      for(const auto& op2B : qops2['B']){
         if(op2Q.sym != op2B.second.sym) continue;
         auto qr = oper_unpack(op2B.first);
         int q2 = qr.first;
         int r2 = qr.second;
         op2Q += int2e.getAnti(p,q2,s,r2)*op2B.second;
      }
      opwf += oper_kernel_IOwf(superblock,ksite,op2Q,0);
   }
   // 3. <pq1||sr2> aq1^+ar2
   // 4. <pq2||sr1> aq2^+ar1
   if(qops1['C'].size() < qops2['C'].size()){
      // 3. q1^+ * (<pq1||sr2> r2)
      for(const auto& op1C : qops1['C']){
	 int q1 = op1C.first;
	 const auto& op1 = op1C.second;
	 // op2 = <pq1||sr2> r2 
	 qsym rsym = symQ - op1.sym;
	 auto& it = qops2['C'].begin()->second;
	 qtensor2 op2(-rsym, it.qrow, it.qcol);
	 for(const auto& op2C : qops2['C']){
	    if(op2.sym != op2C.second.sym) continue;
	    int r2 = op2C.first;
	    op2 += int2e.getAnti(p,q1,s,r2)*op2C.second;
	 }
	 opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),1);
      }
      // 4. r1 * (-<pq2||sr1> q2^+)
      for(const auto& op1C : qops1['C']){
	 int r1 = op1C.first;
	 const auto& op1 = op1C.second;
	 // op2 = -<pq2||sr1> q2^+
	 qsym rsym = symQ + op1.sym;
	 auto& it = qops2['C'].begin()->second;
	 qtensor2 op2(rsym, it.qrow, it.qcol);
	 for(const auto& op2C : qops2['C']){
            if(op2.sym != op2C.second.sym) continue;
	    int q2 = op2C.first;
	    op2 -= int2e.getAnti(p,q2,s,r1)*op2C.second;
	 }
	 opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,1);
      }
   }else{
      // 3. (<pq1||sr2> q1^+) * r2
      for(const auto& op2C : qops2['C']){
	 int r2 = op2C.first;
	 const auto& op2 = op2C.second;
	 // op1 = <pq1||sr2> q1^+
	 qsym rsym = symQ + op2.sym;
	 auto& it = qops1['C'].begin()->second;
	 qtensor2 op1(rsym, it.qrow, it.qcol);
	 for(const auto& op1C : qops1['C']){
	    if(op1.sym != op1C.second.sym) continue;
	    int q1 = op1C.first;
	    op1 += int2e.getAnti(p,q1,s,r2)*op1C.second;
	 }
	 opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),1);
      }
      // 4. (-<pq2||sr1> r1) * q2^+
      for(const auto& op2C : qops2['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 // op1 = -<pq2||sr1> r1
	 qsym rsym = symQ - op2.sym;
	 auto& it = qops1['C'].begin()->second;
	 qtensor2 op1(-rsym, it.qrow, it.qcol);
	 for(const auto& op1C : qops1['C']){
            if(op1.sym != op1C.second.sym) continue;
	    int r1 = op1C.first;
	    op1 -= int2e.getAnti(p,q2,s,r1)*op1C.second;
	 }
	 opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,1);
      }
   }
   return opwf;
}

// kernel for computing renormalized Sp|ket> [6 terms]
qtensor3 tns::oper_kernel_Swf(const string& superblock,
		              const qtensor3& ksite,
			      oper_dict& qops1,
			      oper_dict& qops2,
	                      const integral::two_body& int2e,
	                      const integral::one_body& int1e,
		              const int index){
   // determine symmetry
   int p = index, spin_p = p%2;
   qsym symS = (spin_p == 0)? qsym(-1,-1) : qsym(-1,0); // Sa or Sb
   qtensor3 opwf(symS+ksite.sym, ksite.qmid, ksite.qrow, ksite.qcol, ksite.dir);
   //
   // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   //    = Sp^1 + Sp^2 (S exists in both blocks)
   //    + <pq2||s1r1> aq2^+ar1as1 => aq2^+*Ppq2^1
   //    + <pq1||s1r2> aq1^+ar2as1 => Qpr2^1*ar2 
   //    + <pq1||s2r2> aq1^+ar2as2 => aq1^+*Ppq1^2
   //    + <pq2||s1r2> aq2^+ar2as1 => Qps1^2*as1
   //
   // 1. qCrCsC: Sc*Ir
   opwf += oper_kernel_OIwf(superblock,ksite,qops1['S'].at(p));
   // 2. qRrRsR: Ic*Sr
   opwf += oper_kernel_IOwf(superblock,ksite,qops2['S'].at(p),1);
   // 3. <pq2||s1r1> aq2^+ ar1 as1 
   bool ifP1 = qops1.find('P') != qops1.end();
   if(ifP1){
      // 3. <pq2||s1r1> aq2^+ar1as1 => Ppq2^1*aq2^+
      for(const auto& op2C : qops2['C']){
         int q2 = op2C.first;
         const auto& op2 = op2C.second;
	 // special treatment of index for P 
         if(p < q2){	
	    const auto& op1P = qops1['P'].at(oper_pack(p,q2));
            opwf += oper_kernel_OOwf(superblock,ksite,op1P,op2,1);
	 }else{
	    const auto& op1P = qops1['P'].at(oper_pack(q2,p));
            opwf -= oper_kernel_OOwf(superblock,ksite,op1P,op2,1);
	 }
      }
   }else{
      if(qops1['A'].size() < qops2['C'].size()){
         // 3. <pq2||s1r1> aq2^+ ar1 as1 = ar1as1 (<pq2||s1r1>aq2^+)
         for(const auto& op1A : qops1['A']){
            auto sr = oper_unpack(op1A.first);
            int s1 = sr.first;
            int r1 = sr.second;
	    const auto& op1 = op1A.second;
            // op2 = <pq2||s1r1>aq2^+
            qsym rsym = symS + op1.sym;
	    auto& it = qops2['C'].begin()->second;
            qtensor2 op2(rsym, it.qrow, it.qcol);
            for(const auto& op2C : qops2['C']){
               if(op2.sym != op2C.second.sym) continue;
               int q2 = op2C.first;
               op2 += int2e.getAnti(p,q2,s1,r1)*op2C.second;
            }
            opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,1);
         }
      }else{
         // 3. <pq2||s1r1> aq2^+ ar1 as1 = (<pq2||s1r1>ar1as1) aq2^+
	 for(const auto& op2C : qops2['C']){
	    int q2 = op2C.first;
	    const auto& op2 = op2C.second;
	    // op1 = <pq2||s1r1>ar1as1
	    qsym rsym = symS - op2.sym;
	    auto& it = qops1['A'].begin()->second;
	    qtensor2 op1(-rsym, it.qrow, it.qcol);
	    for(const auto& op1A : qops1['A']){
	       if(op1.sym != op1A.second.sym) continue;
	       auto sr = oper_unpack(op1A.first);
	       int s1 = sr.first;
	       int r1 = sr.second;
	       op1 += int2e.getAnti(p,q2,s1,r1)*op1A.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,1);
	 }
      }
   }
   // 4. <pq1||s1r2> aq1^+ ar2 as1 
   bool ifQ1 = qops1.find('Q') != qops1.end();
   if(ifQ1){
      // 4. <pq1||s1r2> aq1^+ar2as1 => Qpr2^1*ar2
      for(const auto& op2C : qops2['C']){
         int r2 = op2C.first;
         const auto& op2 = op2C.second;
         const auto& op1Q = qops1['Q'].at(oper_pack(p,r2));
         opwf += oper_kernel_OOwf(superblock,ksite,op1Q,op2.T(),1);
      }
   }else{
      if(qops1['B'].size() < qops2['C'].size()){
         // 4. <pq1||s1r2> aq1^+ ar2 as1 = aq1^+as1 (-<pq1||s1r2>ar2)
         for(const auto& op1B : qops1['B']){
            auto qs = oper_unpack(op1B.first);
            int q1 = qs.first;
            int s1 = qs.second;
	    const auto& op1 = op1B.second;
            // op2 = -<pq1||s1r2>ar2
            qsym rsym = symS - op1.sym;
            auto& it = qops2['C'].begin()->second;
	    qtensor2 op2(-rsym, it.qrow, it.qcol);
            for(const auto& op2C : qops2['C']){
               if(op2.sym != op2C.second.sym) continue;
               int r2 = op2C.first;
               op2 -= int2e.getAnti(p,q1,s1,r2)*op2C.second;
            }
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),1);
         }
      }else{
         // 4. <pq1||s1r2> aq1^+ ar2 as1 = (-<pq1||s1r2> aq1^+as1) ar2
	 for(const auto& op2C : qops2['C']){
	    int r2 = op2C.first;
	    const auto& op2 = op2C.second;
	    // op1 = -<pq1||s1r2> aq1^+as1
	    qsym rsym = symS + op2.sym;
	    auto& it = qops1['B'].begin()->second;
	    qtensor2 op1(rsym, it.qrow, it.qcol);
	    for(const auto& op1B : qops1['B']){
	       if(op1.sym != op1B.second.sym) continue;
	       auto qs = oper_unpack(op1B.first);
	       int q1 = qs.first;
	       int s1 = qs.second;
	       op1 -= int2e.getAnti(p,q1,s1,r2)*op1B.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),1);
	 }
      }
   }
   // 5. <pq1||s2r2> aq1^+ ar2 as2 
   bool ifP2 = qops2.find('P') != qops2.end();
   if(ifP2){
      // 5. <pq1||s2r2> aq1^+ar2as2 => aq1^+*Ppq1^2
      for(const auto& op1C : qops1['C']){
         int q1 = op1C.first;
	 const auto& op1 = op1C.second;
	 // special treatment of index for P
	 if(p < q1){ 
            const auto& op2P = qops2['P'].at(oper_pack(p,q1));
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2P,0);
	 }else{
            const auto& op2P = qops2['P'].at(oper_pack(q1,p));
            opwf -= oper_kernel_OOwf(superblock,ksite,op1,op2P,0);
	 }
      }
   }else{
      if(qops1['C'].size() < qops2['A'].size()){
         // 5. <pq1||s2r2> aq1^+ar2as2 = aq1^+ (<pq1||s2r2>ar2as2)
         for(const auto& op1C : qops1['C']){
            int q1 = op1C.first;
	    const auto& op1 = op1C.second;
	    // op2 = <pq1||s2r2>ar2as2
            qsym rsym = symS - op1.sym;
	    auto& it = qops2['A'].begin()->second;
            qtensor2 op2(-rsym, it.qrow, it.qcol);
            for(const auto& op2A : qops2['A']){
               if(op2.sym != op2A.second.sym) continue;
               auto sr = oper_unpack(op2A.first);
               int s2 = sr.first;
               int r2 = sr.second;
               op2 += int2e.getAnti(p,q1,s2,r2)*op2A.second;
            }
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),0);
         }
      }else{
         // 5. <pq1||s2r2> aq1^+ar2as2 = (<pq1||s2r2> aq1^+) ar2as2
         for(const auto& op2A : qops2['A']){
	    auto sr = oper_unpack(op2A.first);
	    int s2 = sr.first;
	    int r2 = sr.second;
	    const auto& op2 = op2A.second;
	    // op1 = <pq1||s2r2> aq1^+
	    qsym rsym = symS + op2.sym;
	    auto& it = qops1['C'].begin()->second;
	    qtensor2 op1(rsym, it.qrow, it.qcol);
	    for(const auto& op1C : qops1['C']){
 	       if(op1.sym != op1C.second.sym) continue;
	       int q1 = op1C.first;
	       op1 += int2e.getAnti(p,q1,s2,r2)*op1C.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),0);
	 }
      }
   }
   // 6. <pq2||s1r2> aq2^+ ar2 as1 
   bool ifQ2 = qops2.find('Q') != qops2.end();
   if(ifQ2){
      // 6. <pq2||s1r2> aq2^+ar2as1 => Qps1^2*as1
      for(const auto& op1C : qops1['C']){
         int s1 = op1C.first;
	 const auto& op1 = op1C.second;
         const auto& op2Q = qops2['Q'].at(oper_pack(p,s1));
         opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2Q,0);
      }
   }else{
      if(qops1['C'].size() < qops2['B'].size()){
         // 6. <pq2||s1r2> aq2^+ar2as1 = as1 (<pq2||s1r2>aq2^+ar2)
         for(const auto& op1C : qops1['C']){
            int s1 = op1C.first;
	    const auto& op1 = op1C.second;
	    // op2 = <pq2||s1r2>aq2^+ar2
            qsym rsym = symS + op1.sym;
	    auto& it = qops2['B'].begin()->second;
            qtensor2 op2(rsym, it.qrow, it.qcol);
            for(const auto& op2B : qops2['B']){
               if(op2.sym != op2B.second.sym) continue;
               auto qr = oper_unpack(op2B.first);
               int q2 = qr.first;
               int r2 = qr.second;
               op2 += int2e.getAnti(p,q2,s1,r2)*op2B.second;
            }
            opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,0);
	 }
      }else{
         // 6. <pq2||s1r2> aq2^+ar2as1 = (<pq2||s1r2> as1) aq2^+ar2
      	 for(const auto& op2B : qops2['B']){
	    auto qr = oper_unpack(op2B.first);
	    int q2 = qr.first;
	    int r2 = qr.second;
	    const auto& op2 = op2B.second;
	    // op1 = <pq2||s1r2> as1
	    qsym rsym = symS - op2.sym;
	    auto& it = qops1['C'].begin()->second;
	    qtensor2 op1(-rsym, it.qrow, it.qcol);
	    for(const auto& op1C : qops1['C']){
	       if(op1.sym != op1C.second.sym) continue;
	       int s1 = op1C.first;
	       op1 += int2e.getAnti(p,q2,s1,r2)*op1C.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,0);
	 }
      }
   }
   return opwf;
}

// kernel for computing renormalized H|ket>
qtensor3 tns::oper_kernel_Hwf(const std::string& superblock,
		              const qtensor3& ksite,
		              oper_dict& qops1,
		              oper_dict& qops2,
	                      const integral::two_body& int2e,
	                      const integral::one_body& int1e){
   qsym symH = qsym(0,0);
   qtensor3 opwf(symH+ksite.sym, ksite.qmid, ksite.qrow, ksite.qcol, ksite.dir);
   //
   // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   //   = H1 + H2
   //   + p1^+*Sp1^2 + h.c.
   //   + q2^+*Sq2^1 + h.c.
   //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
   //   + <p1q2||s1r2> p1^+q2^+r2s1 
   //
   // 1. Hc*Ir
   opwf += oper_kernel_OIwf(superblock,ksite,qops1['H'].at(0));
   // 2. Ic*Hr
   opwf += oper_kernel_IOwf(superblock,ksite,qops2['H'].at(0),0);
   // 3. p1^+ Sp1^2 + h.c. 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      const auto& op2S = qops2['S'].at(p1);
      opwf += oper_kernel_OOwf(superblock,ksite,op1,op2S,1);
      opwf -= oper_kernel_OOwf(superblock,ksite,op1.T(),op2S.T(),1);
   }
   // 4. q2^+ Sq2^1 + h.c. 
   for(const auto& op2C : qops2['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      const auto& op1S = qops1['S'].at(q2);
      opwf -= oper_kernel_OOwf(superblock,ksite,op1S,op2,1);
      opwf += oper_kernel_OOwf(superblock,ksite,op1S.T(),op2.T(),1);
   }
   // 5. <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
   int ifA1P2 = (qops1.find('A') != qops1.end() && qops2.find('P') != qops2.end()); 
   int ifP1A2 = (qops1.find('P') != qops1.end() && qops2.find('A') != qops2.end());
   int ifA1A2 = (qops1.find('A') != qops1.end() && qops2.find('A') != qops2.end());
   assert(ifA1P2 + ifP1A2 + ifA1A2 == 1);
   if(ifA1P2){ 
      // Apq^1*Ppq^2 + h.c.
      for(const auto& op1A : qops1['A']){
         const auto& op1 = op1A.second;
         auto pq = oper_unpack(op1A.first);
	 int p = pq.first;
	 int q = pq.second;
	 // special treatment of index for P 
	 if(p < q){
	    const auto& op2 = qops2['P'].at(oper_pack(p,q));
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
            opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),0);
	 }else{
	    const auto& op2 = qops2['P'].at(oper_pack(q,p));
            opwf -= oper_kernel_OOwf(superblock,ksite,op1,op2,0);
            opwf -= oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),0);
	 }
      }
   }else if(ifP1A2){
      // Prs^1+Ars^2+ + h.c.
      for(const auto& op2A : qops2['A']){
         const auto& op2 = op2A.second;
         auto rs = oper_unpack(op2A.first);
	 int r = rs.first;
	 int s = rs.second;
	 // special treatment of index for P 
	 if(r < s){
            const auto& op1 = qops1['P'].at(oper_pack(r,s));
	    opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
            opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),0);
	 }else{
            const auto& op1 = qops1['P'].at(oper_pack(s,r));
	    opwf -= oper_kernel_OOwf(superblock,ksite,op1,op2,0);
            opwf -= oper_kernel_OOwf(superblock,ksite,op1.T(),op2.T(),0);
	 }
      }
   }else if(ifA1A2){
      if(qops1['A'].size() < qops2['A'].size()){
         // 5. p1^+q1^+ (<p1q1||s2r2> r2s2) + h.c.
         for(const auto& op1A : qops1['A']){
            auto pq = oper_unpack(op1A.first);
            int p1 = pq.first;
            int q1 = pq.second;
            const auto& op1 = op1A.second;
            // op2 = <p1q1||s2r2> r2s2
            qsym rsym = symH - op1.sym;
	    auto& it = qops2['A'].begin()->second;
            qtensor2 op2(-rsym, it.qrow, it.qcol);
            for(const auto& op2A : qops2['A']){
               if(op2.sym != op2A.second.sym) continue;
               auto sr = oper_unpack(op2A.first);
               int s2 = sr.first;
               int r2 = sr.second;
               op2 += int2e.getAnti(p1,q1,s2,r2)*op2A.second;
            } // rs
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),0);
            opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,0);
         } // pq
      }else{
         // 5. (<p1q1||s2r2> p1^+q1^+) r2s2 + h.c.
	 for(const auto& op2A : qops2['A']){
	    auto sr = oper_unpack(op2A.first);
	    int s2 = sr.first;
	    int r2 = sr.second;
	    const auto& op2 = op2A.second;
	    // op1 = <p1q1||s2r2> p1^+q1^+
	    qsym rsym = symH + op2.sym;
	    auto& it = qops1['A'].begin()->second;
	    qtensor2 op1(rsym, it.qrow, it.qcol);
	    for(const auto& op1A : qops1['A']){
	       if(op1.sym != op1A.second.sym) continue;
	       auto pq = oper_unpack(op1A.first);
	       int p1 = pq.first;
	       int q1 = pq.second;
	       op1 += int2e.getAnti(p1,q1,s2,r2)*op1A.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1,op2.T(),0);
	    opwf += oper_kernel_OOwf(superblock,ksite,op1.T(),op2,0);
	 }
      }
   }
   // 6. <p1q2||s1r2> p1^+q2^+r2s1 
   int ifB1Q2 = (qops1.find('B') != qops1.end() && qops2.find('Q') != qops2.end()); 
   int ifQ1B2 = (qops1.find('Q') != qops1.end() && qops2.find('B') != qops2.end());
   int ifB1B2 = (qops1.find('B') != qops1.end() && qops2.find('B') != qops2.end());
   assert(ifB1Q2 + ifQ1B2 + ifB1B2 == 1);
   if(ifB1Q2){
      // Bps^1*Qps^2
      for(const auto& op1B : qops1['B']){
         const auto& op1 = op1B.second;
         const auto& op2 = qops2['Q'].at(op1B.first);
         opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
      }
   }else if(ifQ1B2){
      // Qqr^1*Bqr^2
      for(const auto& op2B : qops2['B']){
         const auto& op2 = op2B.second;
         const auto& op1 = qops1['Q'].at(op2B.first);
         opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
      }
   }else if(ifB1B2){
      if(qops1['B'].size() < qops2['B'].size()){
         // 6. p1^+s1 (<p1q2||s1r2> q2^+r2)
         for(const auto& op1B : qops1['B']){
            auto ps = oper_unpack(op1B.first);
            int p1 = ps.first;
            int s1 = ps.second;
            const auto& op1 = op1B.second;
            // op2 = <p1q2||s1r2> q2^+r2
            qsym rsym = symH - op1.sym;
            auto& it = qops2['B'].begin()->second;
            qtensor2 op2(rsym, it.qrow, it.qcol);
            for(const auto& op2B : qops2['B']){
               if(op2.sym != op2B.second.sym) continue;
               auto qr = oper_unpack(op2B.first);
               int q2 = qr.first;
               int r2 = qr.second;
               op2 += int2e.getAnti(p1,q2,s1,r2)*op2B.second;
            }
            opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
	 }
      }else{
         // 6. (<p1q2||s1r2> p1^+s1) q2^r2 
	 for(const auto& op2B : qops2['B']){
	    auto qr = oper_unpack(op2B.first);
	    int q2 = qr.first;
	    int r2 = qr.second;
	    const auto& op2 = op2B.second;
	    // <p1q2||s1r2> p1^+s1
	    qsym rsym = symH - op2.sym;
	    auto& it = qops1['B'].begin()->second;
	    qtensor2 op1(rsym, it.qrow, it.qcol);
	    for(const auto& op1B : qops1['B']){
	       if(op1.sym != op1B.second.sym) continue;
	       auto ps = oper_unpack(op1B.first);
	       int p1 = ps.first;
	       int s1 = ps.second;
	       op1 += int2e.getAnti(p1,q2,s1,r2)*op1B.second;
	    }
	    opwf += oper_kernel_OOwf(superblock,ksite,op1,op2,0);
	 }
      }
   } 
   return opwf;
}
