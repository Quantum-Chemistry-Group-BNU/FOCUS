#ifndef OPER_COMPXWF_H
#define OPER_COMPXWF_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_op1op2xwf.h"
#include "oper_timer.h"
#include "oper_partition.h"

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
         auto sym_opxwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;
         stensor3<Tm> opxwf(sym_opxwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
         // 
         // Ppq = 1/2<pq||sr> aras  (p<q)
         //     = <pq||s1r1> As1r1 [r>s] => Ppq^1
         //     + <pq||s2r2> As2r2 [r>s] => Ppq^2
         //     + <pq||s1r2> ar2*as1	   => -<pq||s1r2> as1*ar2
         //
         // 1. P1*I2
         opxwf += oper_kernel_OIwf(superblock,site,qops1('P').at(index),ifdagger);
         // 2. I1*P2
         opxwf += oper_kernel_IOwf(superblock,site,qops2('P').at(index),0,ifdagger);
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
         oper_op1op2xwf(ifkr,opxwf,superblock,site,qops1('C'),qops2('C'),
               sym_op,oij,0,0,ifdagger); // as1*ar2

         auto t1 = tools::get_time();
#ifdef _OPENMP
#pragma omp critical
#endif
         {
            oper_timer.nP += 1;
            oper_timer.tP += tools::get_duration(t1-t0);
         }
         return opxwf;
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
         auto sym_opxwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;
         stensor3<Tm> opxwf(sym_opxwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
         //
         // Qps = <pq||sr> aq^+ar
         //     = <pq1||sr1> Bq1r1 	=> Qps^1
         // 	  + <pq2||sr2> Bq2r2 	=> Qps^2
         //     + <pq1||sr2> aq1^+ar2 => <pq1||sr2> aq1^+*ar2 
         //     + <pq2||sr1> aq2^+ar1 => -<pq2||sr1> ar1*aq2^+
         //
         // 1. Q1*I2
         opxwf += oper_kernel_OIwf(superblock,site,qops1('Q').at(index),ifdagger);
         // 2. I1*Q2
         opxwf += oper_kernel_IOwf(superblock,site,qops2('Q').at(index),0,ifdagger);
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
         oper_op1op2xwf(ifkr,opxwf,superblock,site,qops1('C'),qops2('C'),
               sym_op,o1ij,1,0,ifdagger); // aq1^+*ar2
         oper_op1op2xwf(ifkr,opxwf,superblock,site,qops1('C'),qops2('C'),
               sym_op,o2ij,0,1,ifdagger); // aq1*ar2^+

         auto t1 = tools::get_time();
#ifdef _OPENMP
#pragma omp critical
#endif
         {
            oper_timer.nQ += 1;
            oper_timer.tQ += tools::get_duration(t1-t0);
         }
         return opxwf;
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
            const bool ifdist1,
            const bool ifdagger=false){
         auto t0 = tools::get_time();

         const int isym = qops1.isym;
         const bool ifkr = qops1.ifkr;
         int p = index, kp = p/2;
         auto sym_op = get_qsym_opS(isym, index);
         auto sym_opxwf = ifdagger? -sym_op+site.info.sym : sym_op+site.info.sym;  
         stensor3<Tm> opxwf(sym_opxwf, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
         //
         // Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
         //    = Sp^1 + Sp^2 (S exists in both blocks)
         //    + <pq1||s2r2> aq[1]^+ar[2]as[2] 
         //    + <pq2||s1r2> aq[2]^+ar[2]as[1] 
         //    + <pq2||s1r1> aq[2]^+ar[1]as[1] 
         //    + <pq1||s1r2> aq[1]^+ar[2]as[1] 
         //
         int iproc = distribute1(ifkr,size,p);
         if(!ifdist1 or iproc==rank){
            // 1. S1*I2
            opxwf += oper_kernel_OIwf(superblock,site,qops1('S').at(index),ifdagger);
            // 2. I1*S2
            opxwf += oper_kernel_IOwf(superblock,site,qops2('S').at(index),1,ifdagger);
         }
         // cross terms
         const auto& cindex1 = qops1.cindex;
         const auto& cindex2 = qops2.cindex;
         int kc1 = ifkr? 2*cindex1.size() : cindex1.size();
         int kA1 = kc1*(kc1-1)/2;
         int kB1 = kc1*kc1;
         int kc2 = ifkr? 2*cindex2.size() : cindex2.size();
         int kA2 = kc2*(kc2-1)/2;
         int kB2 = kc2*kc2;
         // determine NC or CN partition
         assert(qops2.ifexist('A') or qops2.ifexist('Q'));
         assert(qops2.ifexist('B') or qops2.ifexist('Q'));
         assert(qops1.ifexist('A') or qops1.ifexist('P'));
         assert(qops1.ifexist('B') or qops1.ifexist('Q'));
         bool combine_two_index3 = qops2.ifexist('P') and ((qops2.ifexist('A') and kc1<=kA2) or !qops2.ifexist('A')); 
         bool combine_two_index4 = qops2.ifexist('Q') and ((qops2.ifexist('B') and kc1<=kB2) or !qops2.ifexist('B'));
         bool combine_two_index5 = qops1.ifexist('P') and ((qops1.ifexist('A') and kc2<=kA1) or !qops1.ifexist('A'));
         bool combine_two_index6 = qops1.ifexist('Q') and ((qops1.ifexist('B') and kc2<=kB1) or !qops1.ifexist('B'));
         auto aindex1_dist = oper_index_opA_dist(cindex1, ifkr, size, rank, int2e.sorb);
         auto bindex1_dist = oper_index_opB_dist(cindex1, ifkr, size, rank, int2e.sorb);
         auto aindex2_dist = oper_index_opA_dist(cindex2, ifkr, size, rank, int2e.sorb);
         auto bindex2_dist = oper_index_opB_dist(cindex2, ifkr, size, rank, int2e.sorb); 
         const auto& qrow1 = qops1.qbra;
         const auto& qcol1 = qops1.qket;
         const auto& qrow2 = qops2.qbra;
         const auto& qcol2 = qops2.qket;
         if(!ifkr){

            // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]
            if(combine_two_index3){
               // sum_q aq^+[1]*Ppq[2]
               for(const auto& q : cindex1){
                  int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
                  int iproc = distribute2('P',ifkr,size,ipq,int2e.sorb);
                  if(iproc == rank){
                     const auto& op1c = qops1('C').at(q);
                     const auto& op2P = (p<q)? qops2('P').at(ipq) : -qops2('P').at(ipq);
                     opxwf += oper_kernel_OOwf(superblock,site,op1c,op2P,0,ifdagger);
                  }
               }
            }else{
               // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
               for(const auto& isr : aindex2_dist){
                  auto sr = oper_unpack(isr);
                  int s2 = sr.first;
                  int r2 = sr.second;
                  const auto& op2 = qops2('A').at(isr).H();
                  // sum_q <pq1||s2r2> aq[1]^+
                  stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
                  for(const auto& q1 : cindex1){
                     const auto& op1 = qops1('C').at(q1);     
                     if(op1.info.sym != tmp_op1.info.sym) continue;
                     tmp_op1 += int2e.get(p,q1,s2,r2)*op1;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,tmp_op1,op2,0,ifdagger);
               }
            }

            // 4. <pq2||s1r2> aq[2]^+ar[2]as[1] 
            if(combine_two_index4){
               // sum_q aq[1]*Qpq[2]
               for(const auto& q : cindex1){
                  int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
                  int iproc = distribute2('Q',ifkr,size,ipq,int2e.sorb);
                  if(iproc == rank){
                     const auto& op1a = qops1('C').at(q).H();
                     const auto& op2Q = (p<q)? qops2('Q').at(ipq) : qops2('Q').at(ipq).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1a,op2Q,0,ifdagger);
                  }
               }
            }else{
               // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
               for(const auto& iqr : bindex2_dist){
                  auto qr = oper_unpack(iqr);
                  int q2 = qr.first;
                  int r2 = qr.second;
                  const auto& op2 = qops2('B').at(iqr);
                  // sum_s <pq2||s1r2> as[1]
                  stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
                  for(const auto& s1 : cindex1){
                     const auto& op1c = qops1('C').at(s1); 
                     if(-op1c.info.sym != tmp_op1.info.sym) continue;
                     const auto& op1 = op1c.H();
                     tmp_op1 += int2e.get(p,q2,s1,r2)*op1;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,tmp_op1,op2,0,ifdagger);
                  // Hermitian part: q2<->r2
                  if(q2 == r2) continue;	    
                  stensor2<Tm> tmp_op1H(sym_op+op2.info.sym, qrow1, qcol1);
                  for(const auto& s1 : cindex1){
                     const auto& op1c = qops1('C').at(s1);
                     if(-op1c.info.sym != tmp_op1H.info.sym) continue;
                     const auto& op1 = op1c.H();
                     tmp_op1H += int2e.get(p,r2,s1,q2)*op1;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,tmp_op1H,op2.H(),0,ifdagger);
               }
            }

            // 5. <pq2||s1r1> aq[2]^+ar[1]as[1]
            if(combine_two_index5){
               // sum_q Ppq[1]*aq^+[2]
               for(const auto& q : cindex2){
                  int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
                  int iproc = distribute2('P',ifkr,size,ipq,int2e.sorb);
                  if(iproc == rank){
                     const auto& op2c = qops2('C').at(q);
                     const auto& op1P = (p<q)? qops1('P').at(ipq) : -qops1('P').at(ipq);
                     opxwf += oper_kernel_OOwf(superblock,site,op1P,op2c,1,ifdagger);
                  }
               }
            }else{
               // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
               for(const auto& isr : aindex1_dist){ 
                  auto sr = oper_unpack(isr);
                  int s1 = sr.first;
                  int r1 = sr.second;
                  const auto& op1 = qops1('A').at(isr).H();
                  // sum_q <pq2||s1r1> aq[2]^+
                  stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
                  for(const auto& q2 : cindex2){
                     const auto& op2 = qops2('C').at(q2);
                     if(op2.info.sym != tmp_op2.info.sym) continue;
                     tmp_op2 += int2e.get(p,q2,s1,r1)*op2;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
               }
            }

            // 6. <pq1||s1r2> aq[1]^+ar[2]as[1] 
            if(combine_two_index6){
               // sum_q Qpq^[1]*aq[2]
               for(const auto& q : cindex2){
                  int ipq = (p<q)? oper_pack(p,q) : oper_pack(q,p);
                  int iproc = distribute2('Q',ifkr,size,ipq,int2e.sorb);
                  if(iproc == rank){
                     const auto& op2a = qops2('C').at(q).H();
                     const auto& op1Q = (p<q)? qops1('Q').at(ipq) : qops1('Q').at(ipq).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1Q,op2a,1,ifdagger);
                  }
               }
            }else{
               // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
               for(const auto& iqs : bindex1_dist){
                  auto qs = oper_unpack(iqs);
                  int q1 = qs.first;
                  int s1 = qs.second;
                  const auto& op1 = qops1('B').at(iqs);
                  // sum_r -<pq1||s1r2> ar[2]
                  stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
                  for(const auto& r2 : cindex2){
                     const auto& op2c = qops2('C').at(r2);
                     if(-op2c.info.sym != tmp_op2.info.sym) continue;
                     const auto& op2 = op2c.H();
                     tmp_op2 -= int2e.get(p,q1,s1,r2)*op2;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
                  // Hermitian part: q1<->s1
                  if(q1 == s1) continue;
                  stensor2<Tm> tmp_op2H(sym_op+op1.info.sym, qrow2, qcol2);
                  for(const auto& r2 : cindex2){
                     const auto& op2c = qops2('C').at(r2);
                     if(-op2c.info.sym != tmp_op2H.info.sym) continue;		   
                     const auto& op2 = op2c.H();
                     tmp_op2H -= int2e.get(p,s1,q1,r2)*op2;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,op1.H(),tmp_op2H,1,ifdagger);
               }
            }

         }else{

            // Kramers symmetry-adapted version 
            int pa = p, pb = pa+1;

            // 3. <pq1||s2r2> aq[1]^+ar[2]as[2]
            if(combine_two_index3){
               // sum_q aq^+[1]*Ppq[2]
               for(const auto& qa : cindex1){
                  int qb = qa+1, kq = qa/2;
                  const auto& op1c_A = qops1('C').at(qa);
                  int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
                  int iproc_aa = distribute2('P',ifkr,size,ipq_aa,int2e.sorb);
                  if(iproc_aa == rank){
                     const auto& op2P_AA = (kp<kq)? qops2('P').at(ipq_aa) : -qops2('P').at(ipq_aa);
                     opxwf += oper_kernel_OOwf(superblock,site,op1c_A,op2P_AA,0,ifdagger);
                  } 
                  const auto& op1c_B = op1c_A.K(1);
                  int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
                  int iproc_ab = distribute2('P',ifkr,size,ipq_ab,int2e.sorb);
                  if(iproc_ab == rank){
                     const auto& op2P_AB = (kp<kq)? qops2('P').at(ipq_ab) : -qops2('P').at(ipq_ab).K(1);
                     opxwf += oper_kernel_OOwf(superblock,site,op1c_B,op2P_AB,0,ifdagger);
                  }
               }
            }else{
               // sum_sr (sum_q <pq1||s2r2> aq[1]^+) Asr[2]^+
               for(const auto& isr : aindex2_dist){
                  double wt =  wfacAP(isr);
                  auto sr = oper_unpack(isr);
                  int s2 = sr.first , ks2 = s2/2, spin_s2 = s2%2, s2K = s2+1-2*spin_s2;
                  int r2 = sr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;
                  const auto& op2 = qops2('A').at(isr).H();
                  const auto& op2K = op2.K(2-spin_s2-spin_r2);
                  // sum_q <pq1||s2r2> aq[1]^+
                  stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
                  stensor2<Tm> tmp_op1K(sym_op-op2K.info.sym, qrow1, qcol1);
                  for(const auto& q1a : cindex1){
                     int q1b = q1a+1;
                     const auto& op1c_A = qops1('C').at(q1a);
                     const auto& op1c_B = op1c_A.K(1);
                     tmp_op1 += int2e.get(p,q1a,s2,r2)*op1c_A + int2e.get(p,q1b,s2,r2)*op1c_B;
                     tmp_op1K += int2e.get(p,q1a,s2K,r2K)*op1c_A + int2e.get(p,q1b,s2K,r2K)*op1c_B;
                  }
                  opxwf += wt*oper_kernel_OOwf(superblock,site,tmp_op1,op2,0,ifdagger);
                  opxwf += wt*oper_kernel_OOwf(superblock,site,tmp_op1K,op2K,0,ifdagger);
               }
            }

            // 4. <pq2||s1r2> aq[2]^+ar[2]as[1]      
            if(combine_two_index4){
               // sum_q aq[1]*Qpq[2]
               for(const auto& qa : cindex1){
                  int qb = qa+1, kq = qa/2;
                  const auto& op1a_A = qops1('C').at(qa).H();
                  int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
                  int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,int2e.sorb);
                  if(iproc_aa == rank){
                     const auto& op2Q_AA = (kp<kq)? qops2('Q').at(ipq_aa) :  qops2('Q').at(ipq_aa).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1a_A,op2Q_AA,0,ifdagger);
                  } 
                  const auto& op1a_B = op1a_A.K(1);
                  int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
                  int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,int2e.sorb);
                  if(iproc_ab == rank){
                     const auto& op2Q_AB = (kp<kq)? qops2('Q').at(ipq_ab) :  qops2('Q').at(ipq_ab).K(1).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1a_B,op2Q_AB,0,ifdagger);
                  }
               } 
            }else{
               // sum_qr (sum_s <pq2||s1r2> as[1]) aq[2]^+ar[2]
               for(const auto& iqr : bindex2_dist){
                  auto qr = oper_unpack(iqr);
                  int q2 = qr.first , kq2 = q2/2, spin_q2 = q2%2, q2K = q2+1-2*spin_q2;
                  int r2 = qr.second, kr2 = r2/2, spin_r2 = r2%2, r2K = r2+1-2*spin_r2;;
                  const auto& op2 = qops2('B').at(iqr);
                  const auto& op2K =  op2.K(2-spin_q2-spin_r2);
                  // sum_s <pq2||s1r2> as[1]
                  stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
                  stensor2<Tm> tmp_op1K(sym_op-op2K.info.sym, qrow1, qcol1);
                  for(const auto& s1a : cindex1){
                     int s1b = s1a+1;
                     const auto& op1a_A = qops1('C').at(s1a).H();
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
                  for(const auto& s1a : cindex1){
                     int s1b = s1a+1;
                     const auto& op1a_A = qops1('C').at(s1a).H();
                     const auto& op1a_B = op1a_A.K(1);
                     tmp_op1H += int2e.get(p,r2,s1a,q2)*op1a_A + int2e.get(p,r2,s1b,q2)*op1a_B;
                     tmp_op1KH += int2e.get(p,r2K,s1a,q2K)*op1a_A + int2e.get(p,r2K,s1b,q2K)*op1a_B;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,tmp_op1H,op2.H(),0,ifdagger);
                  opxwf += oper_kernel_OOwf(superblock,site,tmp_op1KH,op2K.H(),0,ifdagger);
               }
            }

            // 5. <pq2||s1r1> aq[2]^+ar[1]as[1]      
            if(combine_two_index5){
               // sum_q Ppq[1]*aq^+[2]
               for(const auto& qa : cindex2){
                  int qb = qa+1, kq = qa/2;
                  const auto& op2c_A = qops2('C').at(qa);
                  int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
                  int iproc_aa = distribute2('P',ifkr,size,ipq_aa,int2e.sorb);
                  if(iproc_aa == rank){
                     const auto& op1P_AA = (kp<kq)? qops1('P').at(ipq_aa) : -qops1('P').at(ipq_aa);
                     opxwf += oper_kernel_OOwf(superblock,site,op1P_AA,op2c_A,1,ifdagger);
                  } 
                  const auto& op2c_B = op2c_A.K(1);
                  int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
                  int iproc_ab = distribute2('P',ifkr,size,ipq_ab,int2e.sorb);
                  if(iproc_ab == rank){
                     const auto& op1P_AB = (kp<kq)? qops1('P').at(ipq_ab) : -qops1('P').at(ipq_ab).K(1);
                     opxwf += oper_kernel_OOwf(superblock,site,op1P_AB,op2c_B,1,ifdagger);
                  }
               }
            }else{
               // sum_sr Asr[1]^+ (sum_q <pq2||s1r1> aq[2]^+)
               for(const auto& isr : aindex1_dist){
                  double wt =  wfacAP(isr);
                  auto sr = oper_unpack(isr);
                  int s1 = sr.first , ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
                  int r1 = sr.second, kr1 = r1/2, spin_r1 = r1%2, r1K = r1+1-2*spin_r1;
                  const auto& op1 = qops1('A').at(isr).H();
                  const auto& op1K = op1.K(2-spin_s1-spin_r1);
                  // sum_q <pq2||s1r1> aq[2]^+
                  stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
                  stensor2<Tm> tmp_op2K(sym_op-op1K.info.sym, qrow2, qcol2);
                  for(const auto& q2a : cindex2){
                     int q2b = q2a+1;
                     const auto& op2c_A = qops2('C').at(q2a);
                     const auto& op2c_B = op2c_A.K(1);
                     tmp_op2 += int2e.get(p,q2a,s1,r1)*op2c_A + int2e.get(p,q2b,s1,r1)*op2c_B;
                     tmp_op2K += int2e.get(p,q2a,s1K,r1K)*op2c_A + int2e.get(p,q2b,s1K,r1K)*op2c_B;
                  }
                  opxwf += wt*oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
                  opxwf += wt*oper_kernel_OOwf(superblock,site,op1K,tmp_op2K,1,ifdagger);
               }
            }

            // 6. <pq1||s1r2> aq[1]^+ar[2]as[1]      
            if(combine_two_index6){
               // sum_q Qpq^[1]*aq[2]
               for(const auto& qa : cindex2){
                  int qb = qa+1, kq = qa/2;
                  const auto& op2a_A = qops2('C').at(qa).H();
                  int ipq_aa = (kp<kq)? oper_pack(pa,qa) : oper_pack(qa,pa);
                  int iproc_aa = distribute2('Q',ifkr,size,ipq_aa,int2e.sorb);
                  if(iproc_aa == rank){
                     const auto& op1Q_AA = (kp<kq)? qops1('Q').at(ipq_aa) :  qops1('Q').at(ipq_aa).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1Q_AA,op2a_A,1,ifdagger);
                  } 
                  const auto& op2a_B = op2a_A.K(1);
                  int ipq_ab = (kp<kq)? oper_pack(pa,qb) : oper_pack(qa,pb);
                  int iproc_ab = distribute2('Q',ifkr,size,ipq_ab,int2e.sorb);
                  if(iproc_ab == rank){
                     const auto& op1Q_AB = (kp<kq)? qops1('Q').at(ipq_ab) :  qops1('Q').at(ipq_ab).K(1).H();
                     opxwf += oper_kernel_OOwf(superblock,site,op1Q_AB,op2a_B,1,ifdagger);
                  }
               }
            }else{
               // sum_qs aq[1]^+as[1] (sum_r -<pq1||s1r2> ar[2])
               for(const auto& iqs : bindex1_dist){
                  auto qs = oper_unpack(iqs);
                  int q1 = qs.first , kq1 = q1/2, spin_q1 = q1%2, q1K = q1+1-2*spin_q1;
                  int s1 = qs.second, ks1 = s1/2, spin_s1 = s1%2, s1K = s1+1-2*spin_s1;
                  const auto& op1 = qops1('B').at(iqs);
                  const auto& op1K = op1.K(2-spin_q1-spin_s1);
                  // sum_r -<pq1||s1r2> ar[2]
                  stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
                  stensor2<Tm> tmp_op2K(sym_op-op1K.info.sym, qrow2, qcol2);
                  for(const auto& r2a : cindex2){
                     int r2b = r2a+1;
                     const auto& op2a_A = qops2('C').at(r2a).H();
                     const auto& op2a_B = op2a_A.K(1);
                     tmp_op2 -= int2e.get(p,q1,s1,r2a)*op2a_A + int2e.get(p,q1,s1,r2b)*op2a_B;
                     tmp_op2K -= int2e.get(p,q1K,s1K,r2a)*op2a_A + int2e.get(p,q1K,s1K,r2b)*op2a_B;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
                  opxwf += oper_kernel_OOwf(superblock,site,op1K,tmp_op2K,1,ifdagger);
                  // Hermitian part: q1<->s1
                  if(kq1 == ks1) continue;
                  stensor2<Tm> tmp_op2H(sym_op+op1.info.sym, qrow2, qcol2);
                  stensor2<Tm> tmp_op2KH(sym_op+op1K.info.sym, qrow2, qcol2);
                  for(const auto& r2a : cindex2){
                     int r2b = r2a+1;
                     const auto& op2a_A = qops2('C').at(r2a).H();
                     const auto& op2a_B = op2a_A.K(1);
                     tmp_op2H -= int2e.get(p,s1,q1,r2a)*op2a_A + int2e.get(p,s1,q1,r2b)*op2a_B;
                     tmp_op2KH -= int2e.get(p,s1K,q1K,r2a)*op2a_A + int2e.get(p,s1K,q1K,r2b)*op2a_B;
                  }
                  opxwf += oper_kernel_OOwf(superblock,site,op1.H(),tmp_op2H,1,ifdagger);
                  opxwf += oper_kernel_OOwf(superblock,site,op1K.H(),tmp_op2KH,1,ifdagger);
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
         return opxwf;
      }

   // kernel for computing renormalized H|ket>
   template <typename Tm>
      stensor3<Tm> oper_compxwf_opH(const std::string superblock,
            const stensor3<Tm>& site,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const int size,
            const int rank,
            const bool ifdist1){
         auto t0 = tools::get_time();

         const int  isym = qops1.isym;
         const bool ifkr = qops1.ifkr;
         const bool dagger = true;
         // for AP,BQ terms
         const auto& cindex1 = qops1.cindex;
         const auto& cindex2 = qops2.cindex;
         const bool ifNC = determine_NCorCN_opH(qops1.oplist, qops2.oplist, cindex1.size(), cindex2.size());
         char AP1 = ifNC? 'A' : 'P';
         char AP2 = ifNC? 'P' : 'A';
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? cindex1 : cindex2;
         auto aindex_dist = oper_index_opA_dist(cindex, ifkr, size, rank, qops1.sorb);
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         //
         // H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
         //   = H1 + H2
         //   + p1^+*Sp1^2 + h.c.
         //   + q2^+*Sq2^1 + h.c.
         //   + <p1q1||s2r2> p1^+q1^+r2s2 + h.c.
         //   + <p1q2||s1r2> p1^+q2^+r2s1 
         //
         stensor3<Tm> opxwf(site.info.sym, site.info.qrow, site.info.qcol, site.info.qmid, site.info.dir);
         if(!ifdist1 or rank==0){
            // 1. H1*I2
            opxwf += oper_kernel_OIwf(superblock,site,qops1('H').at(0));
            // 2. I1*H2
            opxwf += oper_kernel_IOwf(superblock,site,qops2('H').at(0),0);
         }
         if(!ifkr){
            // One-index operators
            // 3. sum_p1 p1^+ Sp1^2 + h.c. 
            for(const auto& p1 : cindex1){
               int iproc = distribute1(ifkr,size,p1);
               if(!ifdist1 or iproc==rank){
                  const auto& op1c = qops1('C').at(p1);
                  const auto& op2S = qops2('S').at(p1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1c,op2S,1);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1c,op2S,1,dagger);
               }
            }
            // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
            for(const auto& q2 : cindex2){
               int iproc = distribute1(ifkr,size,q2);
               if(!ifdist1 or iproc==rank){
                  const auto& op2c = qops2('C').at(q2);
                  const auto& op1S = qops1('S').at(q2);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1S,op2c,1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1S,op2c,1,dagger);
               }
            }
            // Two-index operators
            // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
            for(const auto& index : aindex_dist){
               const auto& op1 = qops1(AP1).at(index);
               const auto& op2 = qops2(AP2).at(index);
               opxwf += oper_kernel_OOwf(superblock,site,op1,op2,0);
               opxwf += oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
            }
            // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
            for(const auto& index : bindex_dist){
               const auto& op1 = qops1(BQ1).at(index);
               const auto& op2 = qops2(BQ2).at(index);
               const Tm wt = wfac(index);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
               auto tmp1 = oper_kernel_OOwf(superblock,site,op1,op2,0);
               auto tmp2 = oper_kernel_OOwf(superblock,site,op1,op2,0,dagger);
               linalg::xaxpy(opxwf.size(), wt, tmp1.data(), opxwf.data());
               linalg::xaxpy(opxwf.size(), wt, tmp2.data(), opxwf.data());
            }
         }else{
            // One-index operators
            // 3. sum_p1 p1^+ Sp1^2 + h.c.
            for(const auto& p1 : cindex1){
               int iproc = distribute1(ifkr,size,p1);
               if(!ifdist1 or iproc==rank){
                  const auto& op1c_A = qops1('C').at(p1);
                  const auto& op2S_A = qops2('S').at(p1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1c_A,op2S_A,1);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1c_A,op2S_A,1,dagger);
                  // KR part
                  const auto& op1c_B = op1c_A.K(1);
                  const auto& op2S_B = op2S_A.K(1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1c_B,op2S_B,1);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1c_B,op2S_B,1,dagger);
               }
            }
            // 4. sum_q2 q2^+ Sq2^1 + h.c. = -Sq2^1 q2^+ + h.c. 
            for(const auto& q2 : cindex2){
               int iproc = distribute1(ifkr,size,q2);
               if(!ifdist1 or iproc==rank){ 
                  const auto& op2c_A = qops2('C').at(q2);
                  const auto& op1S_A = qops1('S').at(q2);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1S_A,op2c_A,1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1S_A,op2c_A,1,dagger);
                  // KR part
                  const auto& op2c_B = op2c_A.K(1);
                  const auto& op1S_B = op1S_A.K(1);
                  opxwf -= oper_kernel_OOwf(superblock,site,op1S_B,op2c_B,1);
                  opxwf += oper_kernel_OOwf(superblock,site,op1S_B,op2c_B,1,dagger);
               }
            }
            // Two-index operators
            // 5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
            for(const auto& index : aindex_dist){
               const Tm wt = wfacAP(index);
               const auto& op1_A = qops1(AP1).at(index);
               const auto& op2_A = qops2(AP2).at(index);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
               auto tmp1a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
               auto tmp2a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
               linalg::xaxpy(opxwf.size(), wt, tmp1a.data(), opxwf.data());
               linalg::xaxpy(opxwf.size(), wt, tmp2a.data(), opxwf.data());
               // NOTE: the following lines work for A_{pq} & A_{p\bqr{q}}, 
               // because the global sign in K() does not matter as the pair AP 
               // has even no. of barred indices! That is, the phases will get 
               // cancelled in op1 and op2 after the time-reversal operation!
               const auto& op1_B = op1_A.K(0);  
               const auto& op2_B = op2_A.K(0); 
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
               auto tmp1b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
               auto tmp2b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
               linalg::xaxpy(opxwf.size(), wt, tmp1b.data(), opxwf.data());
               linalg::xaxpy(opxwf.size(), wt, tmp2b.data(), opxwf.data());
            }
            // 6. Bps^1*Qps^2 / Qqr^1*Bqr^2
            for(const auto& index : bindex_dist){
               const Tm wt = wfacBQ(index);
               const auto& op1_A = qops1(BQ1).at(index);
               const auto& op2_A = qops2(BQ2).at(index);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
               auto tmp1a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0);
               auto tmp2a = oper_kernel_OOwf(superblock,site,op1_A,op2_A,0,dagger);
               linalg::xaxpy(opxwf.size(), wt, tmp1a.data(), opxwf.data());
               linalg::xaxpy(opxwf.size(), wt, tmp2a.data(), opxwf.data());
               // KR part
               const auto& op1_B = op1_A.K(0);
               const auto& op2_B = op2_A.K(0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
               //opxwf += wt*oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
               auto tmp1b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0);
               auto tmp2b = oper_kernel_OOwf(superblock,site,op1_B,op2_B,0,dagger);
               linalg::xaxpy(opxwf.size(), wt, tmp1b.data(), opxwf.data());
               linalg::xaxpy(opxwf.size(), wt, tmp2b.data(), opxwf.data());
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
         return opxwf;
      }

} // ctns

#endif
