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

/*
// kernel for computing renormalized Sp|ket> [6 terms]
template <typename Tm>
qtensor3<Tm> oper_opwf_opS(const std::string& superblock,
		           const qtensor3<Tm>& site,
			   oper_dict<Tm>& qops1,
			   oper_dict<Tm>& qops2,
	                   const integral::two_body<Tm>& int2e,
	                   const integral::one_body<Tm>& int1e,
		           const int index,
			   const bool ifdagger=false){
   // determine symmetry
   int p = index, spin_p = p%2;
   qsym symS = (spin_p == 0)? qsym(-1,-1) : qsym(-1,0); // Sa or Sb
   qsym sym_opwf = ifdagger? -symS+site.sym : symS+site.sym;  
   qtensor3 opwf(sym_opwf, site.qmid, site.qrow, site.qcol, site.dir);
   //
   // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   //    = Sp^1 + Sp^2 (S exists in both blocks)
   //    + <pq2||s1r1> aq2^+ar1as1 => aq2^+*Ppq2^1
   //    + <pq1||s1r2> aq1^+ar2as1 => Qpr2^1*ar2 
   //    + <pq1||s2r2> aq1^+ar2as2 => aq1^+*Ppq1^2
   //    + <pq2||s1r2> aq2^+ar2as1 => Qps1^2*as1
   //

   // 1. qCrCsC: Sc*Ir
   opwf += oper_kernel_OIwf(superblock,site,qops1['S'].at(p),ifdagger);
   
   // 2. qRrRsR: Ic*Sr
   opwf += oper_kernel_IOwf(superblock,site,qops2['S'].at(p),1,ifdagger);
   
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
	    if(ifdagger){
	       // no sign change: e*o=e
               opwf += oper_kernel_OOwf(superblock,site,op1P.T(),op2.T(),1); 
	    }else{
               opwf += oper_kernel_OOwf(superblock,site,op1P,op2,1);
	    }
	 }else{
	    const auto& op1P = qops1['P'].at(oper_pack(q2,p));
	    if(ifdagger){
               opwf -= oper_kernel_OOwf(superblock,site,op1P.T(),op2.T(),1);
	    }else{
               opwf -= oper_kernel_OOwf(superblock,site,op1P,op2,1);
	    }
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
	    if(ifdagger){
               opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),1);
	    }else{
               opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,1);
	    }
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
	    if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),1);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,1);
	    }
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
	 if(ifdagger){
            opwf += oper_kernel_OOwf(superblock,site,op1Q.T(),op2,1);
	 }else{
            opwf += oper_kernel_OOwf(superblock,site,op1Q,op2.T(),1);
	 }
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
            if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,1);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),1);
	    }
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
	    if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,1);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),1);
	    }
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
	    if(ifdagger){
               opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2P.T(),0);
	    }else{
               opwf += oper_kernel_OOwf(superblock,site,op1,op2P,0);
	    }
	 }else{
            const auto& op2P = qops2['P'].at(oper_pack(q1,p));
	    if(ifdagger){
               opwf -= oper_kernel_OOwf(superblock,site,op1.T(),op2P.T(),0);
	    }else{
               opwf -= oper_kernel_OOwf(superblock,site,op1,op2P,0);
	    }
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
            if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
	    }
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
	    if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
	    }
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
	 if(ifdagger){
            opwf += oper_kernel_OOwf(superblock,site,op1,op2Q.T(),0);
	 }else{
            opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2Q,0);
	 }
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
	    if(ifdagger){
               opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
	    }else{
               opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
	    }
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
	    if(ifdagger){
	       opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
	    }else{
	       opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
	    }
	 }
      }
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
   qsym symH = qsym(0,0);
   qtensor3 opwf(symH+site.sym, site.qmid, site.qrow, site.qcol, site.dir);
   //
   // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   //   = H1 + H2
   //   + p1^+*Sp1^2 + h.c.
   //   + q2^+*Sq2^1 + h.c.
   //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
   //   + <p1q2||s1r2> p1^+q2^+r2s1 
   //
   
   // 1. Hc*Ir
   opwf += oper_kernel_OIwf(superblock,site,qops1['H'].at(0));
   
   // 2. Ic*Hr
   opwf += oper_kernel_IOwf(superblock,site,qops2['H'].at(0),0);
   
   // 3. p1^+ Sp1^2 + h.c. 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      const auto& op2S = qops2['S'].at(p1);
      opwf += oper_kernel_OOwf(superblock,site,op1,op2S,1);
      opwf -= oper_kernel_OOwf(superblock,site,op1.T(),op2S.T(),1);
   }
   
   // 4. q2^+ Sq2^1 + h.c. 
   for(const auto& op2C : qops2['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      const auto& op1S = qops1['S'].at(q2);
      opwf -= oper_kernel_OOwf(superblock,site,op1S,op2,1);
      opwf += oper_kernel_OOwf(superblock,site,op1S.T(),op2.T(),1);
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
            opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
            opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2.T(),0);
	 }else{
	    const auto& op2 = qops2['P'].at(oper_pack(q,p));
            opwf -= oper_kernel_OOwf(superblock,site,op1,op2,0);
            opwf -= oper_kernel_OOwf(superblock,site,op1.T(),op2.T(),0);
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
	    opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
            opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2.T(),0);
	 }else{
            const auto& op1 = qops1['P'].at(oper_pack(s,r));
	    opwf -= oper_kernel_OOwf(superblock,site,op1,op2,0);
            opwf -= oper_kernel_OOwf(superblock,site,op1.T(),op2.T(),0);
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
            opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
            opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
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
	    opwf += oper_kernel_OOwf(superblock,site,op1,op2.T(),0);
	    opwf += oper_kernel_OOwf(superblock,site,op1.T(),op2,0);
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
         opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
      }
   }else if(ifQ1B2){
      // Qqr^1*Bqr^2
      for(const auto& op2B : qops2['B']){
         const auto& op2 = op2B.second;
         const auto& op1 = qops1['Q'].at(op2B.first);
         opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
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
            opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
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
	    opwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
	 }
      }
   }
   return opwf;
}
*/

} // ctns

#endif
