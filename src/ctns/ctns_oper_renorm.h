#ifndef CTNS_OPER_RENORM_H
#define CTNS_OPER_RENORM_H

#include "ctns_io.h"
#include "ctns_comb.h"
#include "ctns_oper_util.h"
#include "ctns_oper_rbasis.h"
#include "ctns_oper_kernel.h"
#include "ctns_oper_opwf.h"

namespace ctns{

// kernel for computing renormalized ap^+
template <typename Tm>
void oper_renorm_opC(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opC" << std::endl;
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
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opA" << std::endl;
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
	 // pAqA and pAqB
	 if(p1 < p2){
	    // <a1^+a2^+> = [a1^+]*[a2^+]
	    // storage: <a1A^+a2A^+>,<a1A^+a2B^+>
            auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2,1);
	    auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa); 
            auto Hwfab = oper_kernel_OOwf(superblock,site,op1,op2.K(1),1);
	    auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab); 
	    qops['A'][oper_pack(0,p1,p2)] = qt2aa; 
	    qops['A'][oper_pack(1,p1,p2)] = qt2ab; 
	 }else{
 	    // <a2^+a1^+> = -<a1^+a2^+> = -[a1^+]*[a2^+]
	    // storage <a2A^+a1A^+>,<a2A^+a1B^+> 
            auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2,1);
	    auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa); 
            auto Hwfab = oper_kernel_OOwf(superblock,site,op1.K(1),op2,1);
	    auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab); 
	    qops['A'][oper_pack(0,p2,p1)] = -qt2aa;
	    qops['A'][oper_pack(1,p2,p1)] = -qt2ab; 
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
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opB" << std::endl;
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
	 // pAqA and pAqB
	 if(p1 < p2){ 
	    // <a1^+a2> = [a1^+]*[a2]
	    // storage: <a1A^+a2A>,<a1A^+a2B>  
	    auto Hwfaa = oper_kernel_OOwf(superblock,site,op1,op2.H(),1);
	    auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa);
	    auto Hwfab = oper_kernel_OOwf(superblock,site,op1,op2.K(1).H(),1);
	    auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab);
	    qops['B'][oper_pack(0,p1,p2)] = qt2aa; 
	    qops['B'][oper_pack(1,p1,p2)] = qt2ab;
	 }else{ 
	    // <a2^+a1> = -<a1*a2^+> = -[a1]*[a2^+]
	    // storage: <a2A^+a1A>,<a2A^+a1B> 
	    auto Hwfaa = oper_kernel_OOwf(superblock,site,op1.H(),op2,1);
	    auto qt2aa = oper_kernel_renorm(superblock,site,Hwfaa);
	    auto Hwfab = oper_kernel_OOwf(superblock,site,op1.K(1).H(),op2,1);
	    auto qt2ab = oper_kernel_renorm(superblock,site,Hwfab);
	    qops['B'][oper_pack(0,p2,p1)] = -qt2aa; 
	    qops['B'][oper_pack(1,p2,p1)] = -qt2ab;
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
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opP" << std::endl;
   auto t0 = tools::get_time();
   // initialization for Ppq = <pq||sr> aras [r>s] (p<q)
   std::vector<int> index;
   for(int kp : ksupp){
      for(int kq : ksupp){
	 if(kp < kq){
            index.push_back(oper_pack(0,kp,kq)); // Paa 
	    index.push_back(oper_pack(1,kp,kq)); // Pab
	 }else if(kp == kq){
            index.push_back(oper_pack(1,kp,kq)); // Pab 
	 }
      }
   }
   for(const int spq : index){
      auto Hwf = oper_opwf_opP(superblock,site,qops1,qops2,int2e,int1e,spq);
      qops['P'][spq] = oper_kernel_renorm(superblock,site,Hwf);
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
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opQ" << std::endl;
   auto t0 = tools::get_time();
   // initialization for Qps = <pq||sr> aq^+ar
   std::vector<int> index;
   for(int kp : ksupp){
      for(int ks : ksupp){
	 if(kp <= ks){
            index.push_back(oper_pack(0,kp,ks)); // Qaa 
	    index.push_back(oper_pack(1,kp,ks)); // Qab
	 }
      }
   }
   for(const int ps : index){
      //auto Hwf = oper_opwf__opQ(superblock,site,qops1,qops2,int2e,int1e,ps);
      //qops['Q'][ps] = oper_kernel_renorm(superblock,site,Hwf);
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
		     const std::vector<int>& ksupp,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opS" << std::endl;
   auto t0 = tools::get_time();
   // initialization for 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   for(const int kp : ksupp){
      //auto Hwf = oper_opwf_opS(superblock,site,qops1,qops2,int2e,int1e,kp);
      //qops['S'][kp] = oper_kernel_renorm(superblock,site,Hwf);
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
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "ctns::oper_renorm_opH" << std::endl;
   auto t0 = tools::get_time();
   // <b1b2|H|b1b2>
   {
      //auto Hwf = oper_opwf_opH(superblock,site,qops1,qops2,int2e,int1e);
      //qops['H'][0] = oper_kernel_renorm(superblock,site,Hwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for tns::oper_renorm_opH : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}
 
// renormalize ops
template <typename Tm>
oper_dict<Tm> oper_renorm_opAll(const std::string& superblock,
			      const comb<Tm>& icomb,
		              const comb_coord& p,
			      oper_dict<Tm>& qops1,
			      oper_dict<Tm>& qops2,
		              const integral::two_body<Tm>& int2e,
		              const integral::one_body<Tm>& int1e,
			      const bool debug=true){
   const bool ifcheck = true;
   auto t0 = tools::get_time();
   std::cout << "\nctns::oper_renorm_opAll coord=" << p 
             << " superblock=" << superblock << std::endl;
   auto& node = icomb.topo.get_node(p);
   qtensor3<Tm> site;
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
   oper_dict<Tm> qops;
   // C,S,H
   oper_renorm_opC(superblock,site,qops1,qops2,qops,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'C');
/*
   oper_renorm_opS(superblock,site,qops1,qops2,qops,
   		   ksupp,int2e,int1e,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'S',int2e,int1e);
   oper_renorm_opH(superblock,site,qops1,qops2,qops,
   		   int2e,int1e,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'H',int2e,int1e);
   // consistency check
   auto H = qops['H'].at(0);
   auto diffH = (H-H.T()).normF();
   if(diffH > 1.e-10){
      H.print("H",2);
      cout << "error: H-H.T is too large! diffH=" << diffH << endl;
      exit(1);
   }
*/
   // AB/PQ
   oper_renorm_opA(superblock,site,qops1,qops2,qops,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'A');
   
   oper_renorm_opB(superblock,site,qops1,qops2,qops,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'B');

   oper_renorm_opP(superblock,site,qops1,qops2,qops,ksupp,int2e,int1e,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'P',int2e,int1e);
   exit(1);
/*
   oper_renorm_opQ(superblock,site,qops1,qops2,qops,
   	              ksupp,int2e,int1e,debug);
   if(debug && ifcheck) oper_check_rbasis(icomb,icomb,p,qops,'Q',int2e,int1e);
*/
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opAll : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
   return qops;
}

/*
void oper_renorm_onedot(const comb& icomb, 
		             const directed_bond& dbond,
			     oper_dict& cqops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e,
			     const string scratch){
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type.at(p0) == 3 && p1.second == 1);
   cout << "ctns::oper_renorm_onedot" << endl;
   oper_dict qops;
   if(forward){
      if(!cturn){
	 qops = oper_renorm_opAll("lc", icomb, icomb, p, lqops, cqops, int2e, int1e);
      }else{
	 qops = oper_renorm_opAll("lr", icomb, icomb, p, lqops, rqops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      qops = oper_renorm_opAll("cr", icomb, icomb, p, cqops, rqops, int2e, int1e);
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}

void oper_renorm_twodot(const comb& icomb, 
		             const directed_bond& dbond,
			     oper_dict& c1qops,
			     oper_dict& c2qops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e, 
			     const string scratch){ 
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type.at(p0) == 3 && p1.second == 1);
   cout << "ctns::oper_renorm_twodot" << endl;
   oper_dict qops;
   if(forward){
      if(!cturn){
	 qops = oper_renorm_opAll("lc", icomb, icomb, p, lqops, c1qops, int2e, int1e);
      }else{
	 qops = oper_renorm_opAll("lr", icomb, icomb, p, lqops, rqops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      if(!cturn){
         qops = oper_renorm_opAll("cr", icomb, icomb, p, c2qops, rqops, int2e, int1e);
      }else{
         qops = oper_renorm_opAll("cr", icomb, icomb, p, c1qops, c2qops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}
*/

} // ctns

#endif
