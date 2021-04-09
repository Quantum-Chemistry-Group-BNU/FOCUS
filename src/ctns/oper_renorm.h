#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#include "oper_kernel.h"
#include "oper_rbasis.h"
#include "oper_compxwf.h"

namespace ctns{

// kernel for computing renormalized ap^+
template <typename Tm>
void oper_renorm_opC(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opC" << std::endl;
   auto t0 = tools::get_time();
   // 1. p1^+*I2 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      auto Hwf = oper_kernel_OIwf(superblock,site,op1);
      qops['C'][p1] = oper_kernel_renorm(superblock,site,Hwf);
   }
   // 2. I1*p2^+ 
   for(const auto& op2C : qops2['C']){
      int p2 = op2C.first;
      const auto& op2 = op2C.second;
      auto Hwf = oper_kernel_IOwf(superblock,site,op2,1);
      qops['C'][p2] = oper_kernel_renorm(superblock,site,Hwf); 
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for ctns::oper_renorm_opC : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// kernel for computing renormalized Apq=ap^+aq^+
template <typename Tm>
void oper_renorm_opA(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opA" << std::endl;
   auto t0 = tools::get_time();
   // 1. p1^+q1^+ * I2
   for(const auto& op1A : qops1['A']){
      int pq1 = op1A.first;
      const auto& op1 = op1A.second;
      auto Hwf = oper_kernel_OIwf(superblock,site,op1);
      qops['A'][pq1] = oper_kernel_renorm(superblock,site,Hwf);
   }
   // 2. I1 * p2^+q2^+ (p<q) 
   for(const auto& op2A : qops2['A']){
      int pq2 = op2A.first;
      const auto& op2 = op2A.second;
      auto Hwf = oper_kernel_IOwf(superblock,site,op2,0);
      qops['A'][pq2] = oper_kernel_renorm(superblock,site,Hwf);
   }
   // 3. p1^+ * q2^+
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      for(const auto& op2C : qops2['C']){
	 int p2 = op2C.first;
	 const auto& op2 = op2C.second;
	 //
	 // tricky part: determine the storage pattern for Apq
	 //
	 if(not ifkr){
            auto Hwf = oper_kernel_OOwf(superblock,site,op1,op2,1);
	    auto qt2 = oper_kernel_renorm(superblock,site,Hwf);
            // only store Apq where p<q: total number (2K)*(2K-1)/2 ~ O(2K^2)
	    if(p1 < p2){
	       qops['A'][oper_pack(p1,p2)] = qt2; 
	    }else{
	       qops['A'][oper_pack(p2,p1)] = -qt2;
	    }
	 }else{
	    int kp1 = p1/2;
	    int kp2 = p2/2;
            assert(p1%2 == 0 && p2%2 == 0 && kp1 != kp2);
	    // If time-reversal symmetry adapted basis is used, Apq blocks:
	    // pA+qA+ and pA+qB+: K*(K-1)/2+K*(K+1)/2=K^2 (reduction by half)
            if(kp1 < kp2){
	       // <a1^+a2^+> = [a1^+]*[a2^+]
	       // storage: <a1A^+a2A^+>,<a1A^+a2B^+>
               auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2,1);
	       auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa); 
               auto Hwfab = oper_kernel_OOwf(superblock,site,op1,op2.K(1),1);
	       auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab); 
	       qops['A'][oper_pack(p1,p2)] = qt2aa; 
	       qops['A'][oper_pack(p1,p2+1)] = qt2ab; 
	    }else{
 	       // <a2^+a1^+> = -<a1^+a2^+> = -[a1^+]*[a2^+]
	       // storage <a2A^+a1A^+>,<a2A^+a1B^+> 
               auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2,1);
	       auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa); 
               auto Hwfab = oper_kernel_OOwf(superblock,site,op1.K(1),op2,1);
	       auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab); 
	       qops['A'][oper_pack(p2,p1)] = -qt2aa;
	       qops['A'][oper_pack(p2,p1+1)] = -qt2ab; 
	    }
	 }
      }
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opA : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// kernel for computing renormalized ap^+aq
template <typename Tm>
void oper_renorm_opB(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opB" << std::endl;
   auto t0 = tools::get_time();
   // 1. p1^+q1 * I2
   for(const auto& op1B : qops1['B']){
      int pq1 = op1B.first;
      const auto& op1 = op1B.second;
      auto Hwf = oper_kernel_OIwf(superblock,site,op1);
      qops['B'][pq1] = oper_kernel_renorm(superblock,site,Hwf);
   }
   // 2. I1 * p2^+q2 
   for(const auto& op2B : qops2['B']){
      int pq2 = op2B.first;
      const auto& op2 = op2B.second;
      auto Hwf = oper_kernel_IOwf(superblock,site,op2,0);
      qops['B'][pq2] = oper_kernel_renorm(superblock,site,Hwf);
   }
   // 3. p1^+*q2 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      for(const auto& op2C : qops2['C']){
	 int p2 = op2C.first;
	 const auto& op2 = op2C.second;
	 //
	 // tricky part: determine the storage pattern for Bps
	 //
	 if(not ifkr){
	    // only store Bps (p<=s): (2K)*(2K+1)/2 ~ O(2K^2)
	    if(p1 < p2){
	       auto Hwf = oper_kernel_OOwf(superblock,site,op1,op2.H(),1);
	       qops['B'][oper_pack(p1,p2)] =  oper_kernel_renorm(superblock,site,Hwf);
	    }else{
	       auto Hwf = oper_kernel_OOwf(superblock,site,op1.H(),op2,1);
	       qops['B'][oper_pack(p2,p1)] = -oper_kernel_renorm(superblock,site,Hwf);
	    }
	 }else{
	    int kp1 = p1/2;
	    int kp2 = p2/2;
            assert(p1%2 == 0 && p2%2 == 0 && kp1 != kp2);
	    // If time-reversal symmetry adapted basis is used, Apq blocks:
	    // pA+qA and pA+qB: K*(K+1)/2+K*(K+1)/2=K(K+1) (reduction by half)
	    if(kp1 < kp2){ 
	       // <a1^+a2> = [a1^+]*[a2]
	       // storage: <a1A^+a2A>,<a1A^+a2B>  
	       auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2.H(),1);
	       auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa);
	       auto Hwfab = oper_kernel_OOwf(superblock,site,op1,op2.K(1).H(),1);
	       auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab);
	       qops['B'][oper_pack(p1,p2)] = qt2aa; 
	       qops['B'][oper_pack(p1,p2+1)] = qt2ab;
	    }else{ 
	       // <a2^+a1> = -<a1*a2^+> = -[a1]*[a2^+]
	       // storage: <a2A^+a1A>,<a2A^+a1B> 
	       auto Hwfaa = oper_kernel_OOwf(superblock,site,op1.H(),op2,1);
	       auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa);
	       auto Hwfab = oper_kernel_OOwf(superblock,site,op1.K(1).H(),op2,1);
	       auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab);
	       qops['B'][oper_pack(p2,p1)] = -qt2aa; 
	       qops['B'][oper_pack(p2,p1+1)] = -qt2ab;
	    }
         }
      }
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for tns::oper_renorm_opB : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

template <typename Tm>
void oper_renorm_opP(const std::string& superblock,	
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opP" << std::endl;
   auto t0 = tools::get_time();
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   std::vector<int> index;
   for(int kp : ksupp){
      int pa = 2*kp, pb = pa+1;
      for(int kq : ksupp){
	 int qa = 2*kq, qb = qa+1;
	 //
	 // tricky part: determine the storage pattern for Ppq for p,q in ksupp
	 //
	 if(kp < kq){
            index.push_back(oper_pack(pa,qa)); // Paa 
	    index.push_back(oper_pack(pa,qb)); // Pab
	    if(not ifkr){
	       // since if kp<kq, pb<qa and pb<qb hold
	       index.push_back(oper_pack(pb,qa));
	       index.push_back(oper_pack(pb,qb));
	    }
	 }else if(kp == kq){
            index.push_back(oper_pack(pa,pb)); // Pab 
	 }
      }
   }
   for(const int pq : index){
      auto Hwf = oper_compxwf_opP(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,pq);
      qops['P'][pq] = oper_kernel_renorm(superblock,site,Hwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opP : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

template <typename Tm>
void oper_renorm_opQ(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opQ" << std::endl;
   auto t0 = tools::get_time();
   // initialization for Qps = <pq||sr> aq^+ar
   std::vector<int> index;
   for(int kp : ksupp){
      int pa = 2*kp, pb = pa+1;
      for(int ks : ksupp){
	 int sa = 2*ks, sb = sa+1;
	 if(kp <= ks){ 
	    index.push_back(oper_pack(pa,sa));
	    index.push_back(oper_pack(pa,sb));
	    if(not ifkr){
	       // if kp=ks, QpApB is stored while QpBpA is redundant,
	       // because it can be related with QpApB using Hermiticity if bra=ket.
	       if(kp != ks) index.push_back(oper_pack(pb,sa));
	       index.push_back(oper_pack(pb,sb));
	    }
	 }
      }
   }
   for(const int ps : index){
      auto Hwf = oper_compxwf_opQ(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,ps);
      qops['Q'][ps] = oper_kernel_renorm(superblock,site,Hwf);
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for ctns::oper_renorm_opQ : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

template <typename Tm>
void oper_renorm_opS(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opS" << std::endl;
   auto t0 = tools::get_time();
   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   std::vector<int> index;
   for(int kp: ksupp){
      int pa = 2*kp, pb = pa+1;
      index.push_back(pa);
      if(not ifkr) index.push_back(pb);
   }
   for(const int p : index){
      auto Hwf = oper_compxwf_opS(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,p);
      qops['S'][p] = oper_kernel_renorm(superblock,site,Hwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opS : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

template <typename Tm>
void oper_renorm_opH(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opH" << std::endl;
   auto t0 = tools::get_time();
   
   auto Hwf = oper_compxwf_opH(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e);
   qops['H'][0] = oper_kernel_renorm(superblock,site,Hwf);
   
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for tns::oper_renorm_opH : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// renormalize ops
template <typename Km>
void oper_renorm_opAll(const std::string& superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<typename Km::dtype>& int2e,
		       const integral::one_body<typename Km::dtype>& int1e,
		       oper_dict<typename Km::dtype>& qops1,
		       oper_dict<typename Km::dtype>& qops2,
		       oper_dict<typename Km::dtype>& qops,
		       const bool debug=true){
   const bool ifcheck = true;
   auto t0 = tools::get_time();
   const bool ifkr = kind::is_kramers<Km>();
   const int isym = Km::isym;
   std::cout << "ctns::oper_renorm_opAll coord=" << p 
             << " superblock=" << superblock 
	     << " ifkr=" << ifkr << std::endl;
  
   // settings for current site & ksupp 
   auto& node = icomb.topo.get_node(p);
   qtensor3<typename Km::dtype> site;
   std::vector<int> ksupp; // support for index of complementary ops 
   if(superblock == "cr"){
      site = icomb.rsites.at(p);
      ksupp = node.lsupport;
   }else if(superblock == "lc"){
      site = icomb.lsites.at(p);
      auto pr = node.right;
      ksupp = icomb.topo.get_node(pr).rsupport;
   }else if(superblock == "lr"){
      site = icomb.lsites.at(p);
      auto pc = node.center;
      ksupp = icomb.topo.get_node(pc).rsupport;
   }
  
   // C
   oper_renorm_opC(superblock, site, qops1, qops2, qops, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'C');
   // A
   oper_renorm_opA(superblock, site, qops1, qops2, qops, ifkr, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'A');
   // B
   oper_renorm_opB(superblock, site, qops1, qops2, qops, ifkr, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'B');
   // P
   oper_renorm_opP(superblock, site, qops1, qops2, qops, isym, ifkr, ksupp, int2e, int1e, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'P', int2e, int1e);
   // Q
   oper_renorm_opQ(superblock, site, qops1, qops2, qops, isym, ifkr, ksupp, int2e, int1e, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'Q', int2e, int1e);
   // S
   oper_renorm_opS(superblock, site, qops1, qops2, qops, isym, ifkr, ksupp, int2e, int1e, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'S', int2e, int1e);
   
   // H
   oper_renorm_opH(superblock, site, qops1, qops2, qops, isym, ifkr, int2e, int1e, debug);
   if(debug && ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'H', int2e, int1e);
   // consistency check for Hamiltonian
   const auto& H = qops['H'].at(0);
   auto diffH = (H-H.H()).normF();
   if(diffH > 1.e-10){
      H.print("H",2);
      std::cout << "error: H-H.H() is too large! diffH=" << diffH << std::endl;
      exit(1);
   }

   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opAll : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
      std::cout << std::endl;
   }
}

} // ctns

#endif
