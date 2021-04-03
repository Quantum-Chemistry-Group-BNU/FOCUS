#ifndef OPER_COMPXWF_H
#define OPER_COMPXWF_H

namespace ctns{

/*
 compute complementary operators in the superblock acting on a wavefunction
*/
 
// opwf = (sum_{ij} oij*a1^(d)[i]*a2^(d)[i])*wf
template <typename Tm>
void oper_op1op2xwf_nkr(qtensor3<Tm>& opwf,
		        const std::string& superblock,
		        const qtensor3<Tm>& site,
		        oper_map<Tm>& qops1C,
		        oper_map<Tm>& qops2C,
		        std::map<std::pair<int,int>,Tm>& oij,
			const bool ifdagger1,
			const bool ifdagger2,
			const bool ifdagger){
   Tm sgn = ifdagger? -1.0 : 1.0;
   const auto& sym_op = opwf.sym;
   const auto& qrow1 = (qops1C.begin()->second).qrow;
   const auto& qcol1 = (qops1C.begin()->second).qcol;
   const auto& qrow2 = (qops2C.begin()->second).qrow;
   const auto& qcol2 = (qops2C.begin()->second).qcol;
   // sum_i a1[i] * (sum_j oij a2[j])
   if(qops1C.size() <= qops2C.size()){
      for(const auto& op1C : qops1C){
         int i = op1C.first;
	 const auto& op1 = ifdagger1? op1C.second : op1C.second.H();
	 // tmp_op2 = sum_j oij a2[j]
	 qtensor2<Tm> tmp_op2(sym_op-op1.sym, qrow2, qcol2);
	 for(const auto& op2C : qops2C){
	    bool symAllowed = tmp_op2.sym == (ifdagger2? op2C.second.sym : -op2C.second.sym); 
	    if(not symAllowed) continue;
	    int j = op2C.first;
	    const auto& op2 = ifdagger2? op2C.second : op2C.second.H();	
	    tmp_op2 += oij[std::make_pair(i,j)]*op2;
	 }
	 opwf += sgn*oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger); 
      }
   // sum_j (sum_i oij a1[i]) * a2[j]
   }else{
      // this part appears when the branch is larger 
      for(const auto& op2C : qops2C){
	 int j = op2C.first;
	 const auto& op2 = ifdagger2? op2C.second : op2C.second.H();
	 // tmp_op1 = sum_i oij a1[i]
	 qtensor2<Tm> tmp_op1(sym_op-op2.sym, qrow1, qcol1);
         for(const auto& op1C : qops1C){
	    bool symAllowed = tmp_op1.sym == (ifdagger1? op1C.second.sym : -op1C.second.sym);
	    if(not symAllowed) continue;
	    int i = op1C.first;
	    const auto& op1 = ifdagger1? op1C.second : op1C.second.H();
	    tmp_op1 += oij[std::make_pair(i,j)]*op1;
	 }
	 opwf += sgn*oper_kernel_OOwf(superblock,site,tmp_op1,op2,1,ifdagger);
      }
   }
}

// TRS version: only unbar part of creation ops is stored
// opwf = (sum_{ij}oij*a1^(d)[i]*a2^(d)[i])*wf
template <typename Tm>
void oper_op1op2xwf_kr(qtensor3<Tm>& opwf,
		       const std::string& superblock,
		       const qtensor3<Tm>& site,
		       oper_map<Tm>& qops1C,
		       oper_map<Tm>& qops2C,
		       std::map<std::pair<int,int>,Tm>& oij,
		       const bool ifdagger1,
		       const bool ifdagger2,
		       const bool ifdagger){
   Tm sgn = ifdagger? -1.0 : 1.0;
   const auto& sym_op = opwf.sym;
   const auto& qrow1 = (qops1C.begin()->second).qrow;
   const auto& qcol1 = (qops1C.begin()->second).qcol;
   const auto& qrow2 = (qops2C.begin()->second).qrow;
   const auto& qcol2 = (qops2C.begin()->second).qcol;
   // sum_i a1[i] * (sum_j oij a2[j])
   if(qops1C.size() <= qops2C.size()){
      for(const auto& op1C : qops1C){
         int ia = op1C.first, ib = ia+1;
	 const auto& op1a = ifdagger1? op1C.second : op1C.second.H();
	 const auto& op1b = op1a.K(1);
	 // top2 = sum_j oij a2[j]
	 qtensor2<Tm> top2a(sym_op-op1a.sym, qrow2, qcol2);
	 qtensor2<Tm> top2b(sym_op-op1b.sym, qrow2, qcol2);
	 for(const auto& op2C : qops2C){
	    int ja = op2C.first, jb = ja+1;
	    const auto& op2a = ifdagger2? op2C.second : op2C.second.H();
            const auto& op2b = op2a.K(1);	    
	    top2a += oij[std::make_pair(ia,ja)]*op2a + oij[std::make_pair(ia,jb)]*op2b;
	    top2b += oij[std::make_pair(ib,ja)]*op2a + oij[std::make_pair(ib,jb)]*op2b;
	 }
	 opwf += sgn*(oper_kernel_OOwf(superblock,site,op1a,top2a,1,ifdagger)
		     +oper_kernel_OOwf(superblock,site,op1b,top2b,1,ifdagger));
      }
   // sum_j (sum_i oij a1[i]) * a2[j]
   }else{
      // this part appears when the branch is larger 
      for(const auto& op2C : qops2C){
	 int ja = op2C.first, jb = ja+1;
	 const auto& op2a = ifdagger2? op2C.second : op2C.second.H();
	 const auto& op2b = op2a.K(1);
	 // top1 = sum_i oij a1[i]
	 qtensor2<Tm> top1a(sym_op-op2a.sym, qrow1, qcol1);
	 qtensor2<Tm> top1b(sym_op-op2b.sym, qrow1, qcol1);
         for(const auto& op1C : qops1C){
	    int ia = op1C.first, ib = ia+1;
	    const auto& op1a = ifdagger1? op1C.second : op1C.second.H();
	    const auto& op1b = op1a.K(1);
	    top1a += oij[std::make_pair(ia,ja)]*op1a + oij[std::make_pair(ib,ja)]*op1b;
	    top1b += oij[std::make_pair(ia,jb)]*op1a + oij[std::make_pair(ib,jb)]*op1b;
	 }
	 opwf += sgn*(oper_kernel_OOwf(superblock,site,top1a,op2a,1,ifdagger)
	             +oper_kernel_OOwf(superblock,site,top1b,op2b,1,ifdagger));
      }
   }
}

template <typename Tm>
void oper_op1op2xwf(const bool& ifkr,
		    qtensor3<Tm>& opwf,
		    const std::string& superblock,
		    const qtensor3<Tm>& site,
		    oper_map<Tm>& qops1C,
		    oper_map<Tm>& qops2C,
		    std::map<std::pair<int,int>,Tm>& oij,
		    const bool ifdagger1,
		    const bool ifdagger2,
		    const bool ifdagger){
   if(not ifkr){
      oper_op1op2xwf_nkr(opwf, superblock, site, qops1C, qops2C, 
		         oij, ifdagger1, ifdagger2, ifdagger);
   }else{
      oper_op1op2xwf_kr(opwf, superblock, site, qops1C, qops2C, 
		        oij, ifdagger1, ifdagger2, ifdagger);
   }
}

// kernel for computing renormalized P|ket> or P^+|ket> 
template <typename Tm>
qtensor3<Tm> oper_compxwf_opP(const std::string& superblock,
		              const qtensor3<Tm>& site,
		              oper_dict<Tm>& qops1,
		              oper_dict<Tm>& qops2,
			      const int& isym,
			      const bool& ifkr,
	                      const integral::two_body<Tm>& int2e,
	                      const integral::one_body<Tm>& int1e,
		              const int index,
		              const bool ifdagger=false){
   auto pq = oper_unpack(index);
   int p = pq.first,  kp = p/2, spin_p = p%2;
   int q = pq.second, kq = q/2, spin_q = q%2;
   // determine symmetry of Ppq
   qsym sym_op;
   if(isym == 1){
      sym_op = qsym(-2,0);
   }else if(isym == 2){
      if(spin_p == 0 && spin_q == 0){
         sym_op = qsym(-2,-2); // Paa
      }else if(spin_p == 1 && spin_q == 1){
         sym_op = qsym(-2, 2); // Pbb
      }else{
         sym_op = qsym(-2, 0); // Pos
      }
   }
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
   std::map<std::pair<int,int>,Tm> oij;
   if(not ifkr){
      for(const auto& op1C : qops1['C']){
         int s1 = op1C.first;
         for(const auto& op2C : qops2['C']){
	    int r2 = op2C.first;
	    oij[std::make_pair(s1,r2)] = -int2e.get(p,q,s1,r2);
	 }
      }
   }else{
      for(const auto& op1C : qops1['C']){
         int s1a = op1C.first, s1b = s1a+1;
         for(const auto& op2C : qops2['C']){
	    int r2a = op2C.first, r2b = r2a+1;
	    oij[std::make_pair(s1a,r2a)] = -int2e.get(p,q,s1a,r2a);
	    oij[std::make_pair(s1a,r2b)] = -int2e.get(p,q,s1a,r2b);
	    oij[std::make_pair(s1b,r2a)] = -int2e.get(p,q,s1b,r2a);
	    oij[std::make_pair(s1b,r2b)] = -int2e.get(p,q,s1b,r2b);
	 }
      }
   }
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1['C'],qops2['C'],
		  oij,false,false,ifdagger);
   return opwf;
}

// kernel for computing renormalized Q|ket> or Q^+|ket>
template <typename Tm>
qtensor3<Tm> oper_compxwf_opQ(const std::string& superblock,
		              const qtensor3<Tm>& site,
			      oper_dict<Tm>& qops1,
			      oper_dict<Tm>& qops2,
			      const int& isym,
			      const bool& ifkr,
	                      const integral::two_body<Tm>& int2e,
	                      const integral::one_body<Tm>& int1e,
		              const int index,
			      const bool ifdagger=false){
   auto ps = oper_unpack(index);
   int p = ps.first,  kp = p/2, spin_p = p%2;
   int s = ps.second, ks = s/2, spin_s = s%2;
   // determine symmetry of Qps
   qsym sym_op;
   if(isym == 1){
      sym_op = qsym(0,0);
   }else if(isym == 2){
      if(spin_p == 0 && spin_s == 0){
         sym_op = qsym(0, 0); // Qaa
      }else if(spin_p == 1 && spin_s == 1){
         sym_op = qsym(0, 0); // Qbb
      }else if(spin_p == 0 && spin_s == 1){
         sym_op = qsym(0,-2); // Qab 
      }else if(spin_p == 1 && spin_s == 0){
         sym_op = qsym(0, 2); // Qba 
      }
   }
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
   // 3. <pq1||sr2> aq1^+*ar2 &  4. -<pr2||sq1> aq1*ar2^+
   std::map<std::pair<int,int>,Tm> o1ij, o2ij;
   if(not ifkr){
      for(const auto& op1C : qops1['C']){
         int q1 = op1C.first;
         for(const auto& op2C : qops2['C']){
            int r2 = op2C.first;
            o1ij[std::make_pair(q1,r2)] =  int2e.get(p,q1,s,r2);
	    o2ij[std::make_pair(q1,r2)] = -int2e.get(p,r2,s,q1);
         }
      }	
   }else{
      for(const auto& op1C : qops1['C']){
         int q1a = op1C.first, q1b = q1a+1;
         for(const auto& op2C : qops2['C']){
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
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1['C'],qops2['C'],
		  o1ij,true,false,ifdagger);
   oper_op1op2xwf(ifkr,opwf,superblock,site,qops1['C'],qops2['C'],
		  o2ij,false,true,ifdagger);
   return opwf;
}

// kernel for computing renormalized Sp|ket> [6 terms]
template <typename Tm>
qtensor3<Tm> oper_compxwf_opS(const std::string& superblock,
		              const qtensor3<Tm>& site,
			      oper_dict<Tm>& qops1,
			      oper_dict<Tm>& qops2,
			      const int& isym,
			      const bool& ifkr,
	                      const integral::two_body<Tm>& int2e,
	                      const integral::one_body<Tm>& int1e,
		              const int index,
			      const bool ifdagger=false){
   int p = index, kp = p/2, spin_p = p%2;
   // determine symmetry
   qsym sym_op;
   if(isym == 1){
      sym_op = qsym(-1,0);
   }else if(isym == 2){
      sym_op = (spin_p==0)? qsym(-1,-1) : qsym(-1,1);
   }
   qsym sym_opwf = ifdagger? -sym_op+site.sym : sym_op+site.sym;  
   qtensor3<Tm> opwf(sym_opwf, site.qmid, site.qrow, site.qcol, site.dir);
/*
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
*/
   return opwf;
}

// kernel for computing renormalized H|ket>
template <typename Tm>
qtensor3<Tm> oper_compxwf_opH(const std::string& superblock,
		              const qtensor3<Tm>& site,
		              oper_dict<Tm>& qops1,
		              oper_dict<Tm>& qops2,
			      const int& isym,
			      const bool& ifkr,
	                      const integral::two_body<Tm>& int2e,
	                      const integral::one_body<Tm>& int1e){
   const bool dagger = true;
   qtensor3<Tm> opwf(site.sym, site.qmid, site.qrow, site.qcol, site.dir);
/*
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
*/
   return opwf;
}

} // ctns

#endif
